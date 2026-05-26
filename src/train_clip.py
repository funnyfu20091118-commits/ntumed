"""
Stage 1: Fine-tune BiomedCLIP on MIMIC-CXR with contrastive loss.
Produces a domain-adapted text encoder for Chest-Diffusion.
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip

from config import Config
from dataset import CLIPDataset
from wandb_utils import init_wandb, log_metrics


def contrastive_loss(image_features, text_features, logit_scale):
    """Symmetric CLIP contrastive loss."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def train_clip(cfg: Config):
    device = torch.device(cfg.device)
    run = init_wandb(cfg, stage="stage1-clip", run_type="train")

    # Load BiomedCLIP
    print("Loading BiomedCLIP...")
    model, preprocess = open_clip.create_model_from_pretrained(cfg.clip_model_name)
    tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)
    model = model.to(device)
    model.train()

    # Dataset
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    train_ds = CLIPDataset(manifest_path, split="train",
                           image_preprocess=preprocess, tokenizer=tokenizer,
                           max_text_len=cfg.max_text_len)
    val_ds = CLIPDataset(manifest_path, split="validate",
                         image_preprocess=preprocess, tokenizer=tokenizer,
                         max_text_len=cfg.max_text_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.clip_batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.clip_batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True)

    if run is not None:
        run.summary["train_samples"] = len(train_ds)
        run.summary["val_samples"] = len(val_ds)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.clip_lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.clip_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

    best_val_loss = float("inf")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    global_step = 0

    for epoch in range(cfg.clip_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"CLIP Epoch {epoch+1}/{cfg.clip_epochs}")
        for images, tokens in pbar:
            images = images.to(device)
            tokens = tokens.to(device)

            optimizer.zero_grad()
            if cfg.mixed_precision:
                with torch.amp.autocast("cuda"):
                    img_feat = model.encode_image(images)
                    txt_feat = model.encode_text(tokens)
                    img_feat = F.normalize(img_feat, dim=-1)
                    txt_feat = F.normalize(txt_feat, dim=-1)
                    loss = contrastive_loss(img_feat, txt_feat, model.logit_scale.exp())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                img_feat = model.encode_image(images)
                txt_feat = model.encode_text(tokens)
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                loss = contrastive_loss(img_feat, txt_feat, model.logit_scale.exp())
                loss.backward()
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1
            if run is not None and global_step % cfg.wandb_log_interval == 0:
                log_metrics(
                    run,
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/logit_scale": model.logit_scale.exp().item(),
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, tokens in val_loader:
                images = images.to(device)
                tokens = tokens.to(device)
                with torch.amp.autocast("cuda"):
                    img_feat = F.normalize(model.encode_image(images), dim=-1)
                    txt_feat = F.normalize(model.encode_text(tokens), dim=-1)
                    loss = contrastive_loss(img_feat, txt_feat, model.logit_scale.exp())
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"  Train loss: {avg_train_loss:.4f}  Val loss: {avg_val_loss:.4f}")

        log_metrics(
            run,
            {
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.checkpoint_dir, "biomedclip_finetuned.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")
            if run is not None:
                run.summary["best_val_loss"] = best_val_loss

    print("CLIP fine-tuning complete.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    cfg = Config()
    train_clip(cfg)
