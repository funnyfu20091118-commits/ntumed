"""
LoRA disease-specific fine-tuning for U-ViT.

Trains a separate LoRA adapter for each CheXpert disease by:
  1. Loading the best base U-ViT checkpoint (epoch 100 by default).
  2. Freezing all base weights; injecting LoRA into attn/FFN layers.
  3. Filtering MIMIC-CXR training set to positive samples for the target disease.
  4. Training with *null text* (unconditional) + one-hot disease label conditioning.
  5. Saving the LoRA delta weights to checkpoints/lora_<slug>.pt.

Usage:
    # Train LoRA for a single disease
    python src/train_lora_disease.py --disease "Atelectasis"

    # Train all 11 diseases sequentially
    python src/train_lora_disease.py --all

    # Quick smoke test
    python src/train_lora_disease.py --disease "Edema" --epochs 1 --max-steps 10
"""
import argparse
import math
import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import MIMICCXRDataset
from diffusion import GaussianDiffusion
from lora_utils import inject_lora_into_uvit, get_lora_state_dict, count_lora_params
from train_uvit import EMA, load_clip_text_encoder
from uvit import UViT

# ── Disease catalogue (mirrors evaluate.py DISEASE_LABELS) ─────────────────
DISEASES = [
    "Atelectasis",
    "Consolidation",
    "Pneumothorax",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Cardiomegaly",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# Map disease name → column index in MIMICCXRDataset.LABEL_COLS
LABEL_COLS = MIMICCXRDataset.LABEL_COLS
DISEASE_COL_IDX = {d: LABEL_COLS.index(d) for d in DISEASES if d in LABEL_COLS}


def slugify(name: str) -> str:
    """Convert disease name to a filesystem-safe slug."""
    return name.lower().replace(" ", "_")


def build_one_hot_label(disease: str, num_labels: int = 14) -> torch.Tensor:
    """Return a float tensor with 1.0 at the target disease column, 0 elsewhere."""
    label = torch.zeros(num_labels)
    idx = DISEASE_COL_IDX.get(disease)
    if idx is not None:
        label[idx] = 1.0
    return label


def get_disease_subset(dataset: MIMICCXRDataset, disease: str) -> Subset:
    """Return a Subset containing only samples where disease label == 1."""
    col_idx = DISEASE_COL_IDX[disease]
    col_name = LABEL_COLS[col_idx]
    vals = dataset.df[col_name].values
    indices = np.where(vals == 1.0)[0].tolist()
    if not indices:
        raise ValueError(f"No positive samples found for disease '{disease}'.")
    print(f"  Disease '{disease}': {len(indices)} positive training samples.")
    return Subset(dataset, indices)


def load_base_uvit(cfg: Config, ckpt_path: str, device: torch.device) -> UViT:
    """Load U-ViT from a full checkpoint (model or ema weights)."""
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

    print(f"  Loading U-ViT base checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("ema", ckpt.get("model"))
    uvit.load_state_dict(state, strict=True)
    return uvit


def train_lora_for_disease(
    disease: str,
    cfg: Config,
    base_ckpt: str,
    rank: int = 16,
    alpha: float = 16.0,
    epochs: int = 5,
    max_steps: int = 200,
    lr: float = 2e-4,
    batch_size: int = 16,
    force: bool = False,
):
    """Full LoRA training pipeline for a single disease."""
    slug = slugify(disease)
    out_path = os.path.join(cfg.checkpoint_dir, f"lora_{slug}.pt")

    if os.path.exists(out_path) and not force:
        print(f"[{disease}] LoRA checkpoint already exists: {out_path}  (use --force to retrain)")
        return

    print(f"\n{'='*60}")
    print(f"  Training LoRA for: {disease}")
    print(f"  rank={rank}, alpha={alpha}, epochs={epochs}, max_steps/epoch={max_steps}, lr={lr}")
    print(f"{'='*60}")

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # ── Frozen components ──────────────────────────────────────────────
    from diffusers import AutoencoderKL

    vae_dtype = torch.float16 if cfg.mixed_precision and device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_name, torch_dtype=vae_dtype
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ── U-ViT + LoRA ──────────────────────────────────────────────────
    uvit = load_base_uvit(cfg, base_ckpt, device)
    n_lora = inject_lora_into_uvit(uvit, rank=rank, alpha=alpha)
    breakdown = count_lora_params(uvit)
    print(f"  LoRA trainable params: {n_lora:,}  "
          f"(A={breakdown['lora_A']:,}, B={breakdown['lora_B']:,})")
    uvit.train()

    # ── Diffusion ──────────────────────────────────────────────────────
    diffusion = GaussianDiffusion(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        device=cfg.device,
    )

    # ── Dataset – positive samples only ───────────────────────────────
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    base_ds = MIMICCXRDataset(manifest_path, split="train", image_size=cfg.image_size)
    disease_ds = get_disease_subset(base_ds, disease)
    loader = DataLoader(
        disease_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Fixed one-hot label for this disease (used for conditioning)
    one_hot = build_one_hot_label(disease, cfg.num_labels).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────
    lora_params = [p for p in uvit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    total_opt_steps = epochs * min(max_steps, len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_opt_steps, 1))
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision and device.type == "cuda" else None

    # ── Training loop ──────────────────────────────────────────────────
    best_loss = float("inf")
    best_lora_sd = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = min(max_steps, len(loader))
        pbar = tqdm(loader, total=n_steps,
                    desc=f"  [{disease}] epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(pbar):
            if step >= n_steps:
                break

            images, _reports, _labels, _mask = batch
            images = images.to(device)
            B = images.shape[0]

            # ── Encode images to latent ────────────────────────────────
            with torch.no_grad():
                if cfg.mixed_precision and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        z0 = vae.encode(images).latent_dist.sample() * 0.18215
                else:
                    z0 = vae.encode(images).latent_dist.sample() * 0.18215

            # ── Null text conditioning (unconditional generation) ──────
            # text_emb shape: (B, text_embed_dim)
            null_text = torch.zeros(B, cfg.text_embed_dim, device=device)

            # Disease label conditioning: repeat one-hot for entire batch
            disease_labels = one_hot.unsqueeze(0).expand(B, -1)

            # ── Sample timestep and add noise ─────────────────────────
            t = torch.randint(0, cfg.num_timesteps, (B,), device=device)
            noise = torch.randn_like(z0)
            z_t = diffusion.q_sample(z0, t, noise)

            # ── Forward + loss ─────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            if cfg.mixed_precision and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    eps_pred = uvit(z_t, t, null_text, disease_labels)
                    loss = F.mse_loss(eps_pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(lora_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                eps_pred = uvit(z_t, t, null_text, disease_labels)
                loss = F.mse_loss(eps_pred, noise)
                loss.backward()
                nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = epoch_loss / n_steps
        print(f"  [{disease}] epoch {epoch+1} avg loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            best_lora_sd = get_lora_state_dict(uvit)

    # ── Save best LoRA weights ────────────────────────────────────────
    save_payload = {
        "disease": disease,
        "rank": rank,
        "alpha": alpha,
        "best_loss": best_loss,
        "epochs_trained": epochs,
        "lora_state_dict": best_lora_sd,
        "uvit_config": {
            "latent_size": cfg.latent_size,
            "latent_channels": cfg.latent_channels,
            "patch_size": cfg.uvit_patch_size,
            "dim": cfg.uvit_dim,
            "depth": cfg.uvit_depth,
            "heads": cfg.uvit_heads,
            "mlp_ratio": cfg.uvit_mlp_ratio,
            "text_embed_dim": cfg.text_embed_dim,
            "num_labels": cfg.num_labels,
        },
    }
    torch.save(save_payload, out_path)
    print(f"  LoRA saved → {out_path}  (best loss: {best_loss:.4f})")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train disease-specific LoRA adapters for U-ViT."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--disease", type=str,
                   help=f"Disease name to train. One of: {DISEASES}")
    g.add_argument("--all", action="store_true",
                   help="Train LoRA for ALL diseases sequentially.")

    p.add_argument("--base-ckpt", type=str, default=None,
                   help="Path to base U-ViT checkpoint (default: uvit_epoch0100.pt).")
    p.add_argument("--rank", type=int, default=16,
                   help="LoRA rank r (default: 16).")
    p.add_argument("--alpha", type=float, default=16.0,
                   help="LoRA scaling alpha (default: 16.0 → scale=1.0).")
    p.add_argument("--epochs", type=int, default=5,
                   help="Fine-tuning epochs per disease (default: 5).")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Max gradient steps per epoch (default: 200).")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="LoRA learning rate (default: 2e-4).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size (default: 16).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing LoRA checkpoints.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    base_ckpt = args.base_ckpt or os.path.join(
        cfg.checkpoint_dir, "uvit_epoch0100.pt"
    )
    if not os.path.exists(base_ckpt):
        # Fallback: find the highest available epoch checkpoint
        import glob
        candidates = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "uvit_epoch*.pt")))
        if not candidates:
            sys.exit(f"ERROR: No base checkpoint found in {cfg.checkpoint_dir}")
        base_ckpt = candidates[-1]
        print(f"[WARNING] epoch0100.pt not found; using fallback: {base_ckpt}")

    diseases = DISEASES if args.all else [args.disease]

    for disease in diseases:
        if disease not in DISEASE_COL_IDX:
            print(f"[SKIP] '{disease}' not in LABEL_COLS. Available: {list(DISEASE_COL_IDX.keys())}")
            continue
        train_lora_for_disease(
            disease=disease,
            cfg=cfg,
            base_ckpt=base_ckpt,
            rank=args.rank,
            alpha=args.alpha,
            epochs=args.epochs,
            max_steps=args.max_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            force=args.force,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
