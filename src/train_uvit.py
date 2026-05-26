"""
Stage 2: Train U-ViT denoising model (the core of Chest-Diffusion).

Pipeline:
  1. Load frozen fine-tuned BiomedCLIP (text encoder only)
  2. Load frozen SD VAE
  3. Train U-ViT to predict noise in latent space conditioned on text
"""
import argparse
import os
import sys
import copy
from itertools import islice
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm

import open_clip
from diffusers import AutoencoderKL

from config import Config
from dataset import MIMICCXRDataset
from uvit import UViT
from diffusion import GaussianDiffusion
from wandb_utils import init_wandb, log_metrics


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def load_clip_text_encoder(cfg: Config, device):
    """Load fine-tuned BiomedCLIP and return frozen text encoder + tokenizer."""
    model, _ = open_clip.create_model_from_pretrained(cfg.clip_model_name)
    tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)

    # Load fine-tuned weights if available
    ft_path = os.path.join(cfg.checkpoint_dir, "biomedclip_finetuned.pt")
    if os.path.exists(ft_path):
        print(f"Loading fine-tuned CLIP from {ft_path}")
        model.load_state_dict(torch.load(ft_path, map_location="cpu", weights_only=True))
    else:
        print("Using original BiomedCLIP (no fine-tuned checkpoint found)")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def encode_text(clip_model, tokenizer, reports, device, max_len=256):
    """Encode a batch of report strings → normalized CLIP text embeddings."""
    tokens = tokenizer(reports, context_length=max_len).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
        text_emb = F.normalize(text_emb, dim=-1)
    return text_emb


@torch.no_grad()
def generate_samples(uvit, vae, diffusion, clip_model, tokenizer, cfg,
                     sample_reports, device, epoch, wandb_run=None):
    """Generate sample CXR images from text prompts."""
    uvit.eval()
    text_emb = encode_text(clip_model, tokenizer, sample_reports, device, cfg.max_text_len)

    B = len(sample_reports)
    shape = (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size)

    if cfg.use_ddim:
        latents = diffusion.ddim_sample(uvit, shape, text_emb,
                                        ddim_steps=cfg.ddim_sampling_steps,
                                        guidance_scale=cfg.guidance_scale)
    else:
        latents = diffusion.ddpm_sample(uvit, shape, text_emb,
                                        guidance_scale=cfg.guidance_scale)

    # Decode latents → images
    # SD VAE expects latents scaled by 0.18215
    vae_dtype = next(vae.parameters()).dtype
    images = vae.decode((latents / 0.18215).to(vae_dtype)).sample
    images = (images + 1) / 2  # [-1,1] → [0,1]
    images = images.clamp(0, 1)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"samples_epoch{epoch:04d}.png")
    save_image(images, save_path, nrow=min(4, B))
    print(f"  Samples saved → {save_path}")
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"samples": wandb.Image(save_path), "epoch": epoch})
        except Exception as exc:
            print(f"  W&B image log skipped: {exc}")
    uvit.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the stage-2 U-ViT denoiser.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the stage-2 microbatch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Override gradient accumulation steps.")
    parser.add_argument("--vae-encode-batch-size", type=int, default=None,
                        help="Override the VAE encode microbatch size.")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="Limit train steps per epoch. Useful for smoke tests.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one limited stage-2 step, then exit without checkpointing or sampling.")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable Weights & Biases logging for this run.")
    return parser.parse_args()


def train_uvit(cfg: Config, max_train_steps: int | None = None, dry_run: bool = False):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    if cfg.train_batch_size < 1:
        raise ValueError("train_batch_size must be >= 1")
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if cfg.vae_encode_batch_size < 1:
        raise ValueError("vae_encode_batch_size must be >= 1")

    run = init_wandb(cfg, stage="stage2-uvit", run_type="train")

    # ── Load frozen components ──
    print("Loading frozen CLIP text encoder...")
    clip_model, tokenizer = load_clip_text_encoder(cfg, device)

    print("Loading frozen SD VAE...")
    vae_dtype = torch.float16 if cfg.mixed_precision and device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_name, torch_dtype=vae_dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ── U-ViT ──
    print("Initializing U-ViT...")
    uvit = UViT(
        latent_size=cfg.latent_size,
        latent_channels=cfg.latent_channels,
        patch_size=cfg.uvit_patch_size,
        dim=cfg.uvit_dim,
        depth=cfg.uvit_depth,
        heads=cfg.uvit_heads,
        mlp_ratio=cfg.uvit_mlp_ratio,
        text_embed_dim=cfg.text_embed_dim,
        num_labels=cfg.num_labels,
    ).to(device)

    num_params = sum(p.numel() for p in uvit.parameters()) / 1e6
    print(f"U-ViT parameters: {num_params:.2f}M")
    if run is not None:
        run.summary["uvit_params_m"] = num_params

    # ── Dataset ──
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    train_ds = MIMICCXRDataset(manifest_path, split="train", image_size=cfg.image_size)

    # Weighted sampling: upweight samples with positive disease labels
    import pandas as pd
    import numpy as np
    label_matrix = train_ds.df[MIMICCXRDataset.LABEL_COLS].fillna(0).values
    pos_count = label_matrix.clip(0, 1).sum(axis=1)  # count positive labels per sample
    sample_weights = 1.0 + pos_count
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Training samples: {len(train_ds)}")
    print(
        "Stage-2 batching: "
        f"micro={cfg.train_batch_size}, accum={cfg.grad_accum_steps}, "
        f"effective={cfg.train_batch_size * cfg.grad_accum_steps}, "
        f"vae_micro={cfg.vae_encode_batch_size}"
    )
    if run is not None:
        run.summary["train_samples"] = len(train_ds)
        run.summary["train_batch_size"] = cfg.train_batch_size
        run.summary["grad_accum_steps"] = cfg.grad_accum_steps
        run.summary["vae_encode_batch_size"] = cfg.vae_encode_batch_size

    epoch_steps = len(train_loader) if max_train_steps is None else min(len(train_loader), max_train_steps)
    if epoch_steps < 1:
        raise ValueError("max_train_steps must be >= 1")
    optimizer_steps_per_epoch = math.ceil(epoch_steps / cfg.grad_accum_steps)
    if dry_run:
        print(f"Dry run enabled: limiting stage 2 to {epoch_steps} train step(s).")

    # ── Training setup ──
    diffusion = GaussianDiffusion(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        device=cfg.device,
    )
    optimizer = torch.optim.AdamW(uvit.parameters(), lr=cfg.train_lr, weight_decay=0.01)
    total_steps = optimizer_steps_per_epoch * cfg.train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None
    ema = EMA(uvit, decay=cfg.ema_decay)

    # Sample prompts for viz
    sample_reports = [
        "No acute cardiopulmonary abnormality.",
        "Small bilateral pleural effusions, left greater than right.",
        "Moderate cardiomegaly with pulmonary edema.",
        "Left lower lobe pneumonia.",
    ]

    # Resume checkpoint
    start_epoch = 0
    ckpt_path = os.path.join(cfg.checkpoint_dir, "uvit_latest.pt")
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Filter out keys with shape mismatches
        model_sd = uvit.state_dict()
        compat_sd = {}
        skipped = []
        for k, v in ckpt["model"].items():
            if k in model_sd and v.shape != model_sd[k].shape:
                skipped.append(k)
            else:
                compat_sd[k] = v
        if skipped:
            print(f"  Skipping shape-mismatched keys: {skipped}")
        missing, unexpected = uvit.load_state_dict(compat_sd, strict=False)
        if missing or unexpected or skipped:
            print(f"  WARNING: architecture changed — training from scratch")
            if missing:
                print(f"    Missing keys: {missing}")
            if unexpected:
                print(f"    Unexpected keys: {unexpected}")
            # Don't restore optimizer/scheduler/epoch for incompatible ckpts
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            if "ema" in ckpt:
                ema.load_state_dict({k: v.to(device) for k, v in ckpt["ema"].items()})
            print(f"  Resumed at epoch {start_epoch}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    global_step = 0

    # ── Training loop ──
    for epoch in range(start_epoch, cfg.train_epochs):
        uvit.train()
        total_loss = 0.0
        loader_iter = train_loader if epoch_steps == len(train_loader) else islice(train_loader, epoch_steps)
        pbar = tqdm(loader_iter, total=epoch_steps, desc=f"U-ViT Epoch {epoch+1}/{cfg.train_epochs}")

        for step, (images, reports, labels, _label_mask) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            accum_window_start = (step // cfg.grad_accum_steps) * cfg.grad_accum_steps
            accum_window_end = min(accum_window_start + cfg.grad_accum_steps, epoch_steps)
            accum_window_size = accum_window_end - accum_window_start
            should_step = (step + 1) == accum_window_end

            # Encode images → latent with frozen VAE (chunked to save VRAM)
            with torch.no_grad():
                vae_chunk = min(cfg.vae_encode_batch_size, images.shape[0])
                z0_parts = []
                for ci in range(0, images.shape[0], vae_chunk):
                    chunk = images[ci:ci + vae_chunk]
                    with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == "cuda"):
                        latents = vae.encode(chunk).latent_dist.sample()
                    z0_parts.append(latents * 0.18215)
                z0 = torch.cat(z0_parts, dim=0)

            # Encode text with frozen CLIP
            text_emb = encode_text(clip_model, tokenizer, reports, device, cfg.max_text_len)

            # Classifier-free guidance dropout: zero out conditions randomly
            if cfg.cond_drop_prob > 0:
                drop_mask = (torch.rand(text_emb.size(0), device=device) < cfg.cond_drop_prob)
                text_emb[drop_mask] = 0.0
                labels[drop_mask] = 0.0

            # Sample random timesteps
            t = torch.randint(0, cfg.num_timesteps, (z0.shape[0],), device=device)

            # Add noise
            noise = torch.randn_like(z0)
            z_t = diffusion.q_sample(z0, t, noise)

            # Predict noise (with gradient accumulation)
            if step % cfg.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            if cfg.mixed_precision:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    eps_pred = uvit(z_t, t, text_emb, labels)
                    loss = F.mse_loss(eps_pred, noise) / accum_window_size
                scaler.scale(loss).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(uvit.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    ema.update(uvit)
            else:
                eps_pred = uvit(z_t, t, text_emb, labels)
                loss = F.mse_loss(eps_pred, noise) / accum_window_size
                loss.backward()
                if should_step:
                    nn.utils.clip_grad_norm_(uvit.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    ema.update(uvit)

            batch_loss = loss.item() * cfg.grad_accum_steps
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}")

            global_step += 1
            if run is not None and global_step % cfg.wandb_log_interval == 0:
                log_metrics(
                    run,
                    {
                        "train/loss": batch_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        avg_loss = total_loss / epoch_steps
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        log_metrics(
            run,
            {
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        if dry_run:
            print("Dry run complete.")
            break

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.train_epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model": uvit.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "config": vars(cfg),
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "uvit_latest.pt"))
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, f"uvit_epoch{epoch+1:04d}.pt"))
            print(f"  Checkpoint saved at epoch {epoch+1}")

        # Generate samples
        if (epoch + 1) % cfg.sample_every == 0:
            # Use EMA weights for sampling
            ema_uvit = copy.deepcopy(uvit)
            ema.apply(ema_uvit)
            generate_samples(ema_uvit, vae, diffusion, clip_model, tokenizer,
                           cfg, sample_reports, device, epoch+1, wandb_run=run)
            del ema_uvit

    print("U-ViT training complete.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.vae_encode_batch_size is not None:
        cfg.vae_encode_batch_size = args.vae_encode_batch_size
    if args.disable_wandb or args.dry_run:
        cfg.wandb_enabled = False

    max_train_steps = args.max_train_steps
    if args.dry_run and max_train_steps is None:
        max_train_steps = 1

    train_uvit(cfg, max_train_steps=max_train_steps, dry_run=args.dry_run)
