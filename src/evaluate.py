"""
Evaluation: FID score + AUROC disease classification.

FID: measures distribution distance between generated and real CXR images.
AUROC: checks if generated images have correct disease features using
       torchxrayvision densenet121-res224-all classifier.
"""
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

import open_clip
from diffusers import AutoencoderKL
from scipy import linalg

from config import Config
from dataset import MIMICCXRDataset
from uvit import UViT
from diffusion import GaussianDiffusion
from train_uvit import load_clip_text_encoder, encode_text, EMA


# ─── FID computation ───────────────────────────────────────────────────────

class InceptionV3Features(nn.Module):
    """Extract 2048-d features from InceptionV3 for FID."""

    def __init__(self, device="cuda"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, images):
        # images: (B, 3, H, W) in [0, 1]
        images = self.transform(images.to(self.device))
        return self.model(images)


def compute_fid(real_features, gen_features):
    """Compute FID between two sets of InceptionV3 features."""
    mu_r = real_features.mean(axis=0)
    mu_g = gen_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def extract_features(images_loader, inception, device, desc="Extracting"):
    """Extract InceptionV3 features from a dataloader of images."""
    all_feats = []
    for batch in tqdm(images_loader, desc=desc):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
        imgs = imgs.clamp(0, 1).to(device)
        feats = inception(imgs)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ─── AUROC evaluation with torchxrayvision ──────────────────────────────────

DISEASE_LABELS = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema",
    "Pleural Effusion", "Pneumonia", "Cardiomegaly", "Lung Lesion",
    "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

# Mapping from our label columns to torchxrayvision pathology indices
TXRV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema",
    "Effusion", "Pneumonia", "Cardiomegaly", "Lung Lesion",
    "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]


def compute_auroc(cfg: Config):
    """
    Generate CXR for each test sample, classify with torchxrayvision,
    compute AUROC against CheXpert ground truth labels.
    """
    import torchxrayvision as xrv
    from sklearn.metrics import roc_auc_score

    device = torch.device(cfg.device)

    # Load classifier
    print("Loading torchxrayvision classifier...")
    classifier = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    xrv_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    # Load our pipeline
    print("Loading Chest-Diffusion pipeline...")
    clip_model, tokenizer = load_clip_text_encoder(cfg, device)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_name).to(device).eval()

    uvit = UViT(
        latent_size=cfg.latent_size,
        latent_channels=cfg.latent_channels,
        patch_size=cfg.uvit_patch_size,
        dim=cfg.uvit_dim,
        depth=cfg.uvit_depth,
        heads=cfg.uvit_heads,
        mlp_ratio=cfg.uvit_mlp_ratio,
        text_embed_dim=cfg.text_embed_dim,
    ).to(device)

    ckpt = torch.load(os.path.join(cfg.checkpoint_dir, "uvit_latest.pt"),
                      map_location="cpu", weights_only=True)
    if "ema" in ckpt:
        uvit.load_state_dict(ckpt["ema"])
    else:
        uvit.load_state_dict(ckpt["model"])
    uvit.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start, beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule, device=cfg.device,
    )

    # Test dataset
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    test_ds = MIMICCXRDataset(manifest_path, split="test", image_size=cfg.image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size,
                             shuffle=False, num_workers=cfg.num_workers)

    all_preds = []
    all_labels = []

    for images, reports, labels in tqdm(test_loader, desc="AUROC eval"):
        B = len(reports)
        text_emb = encode_text(clip_model, tokenizer, list(reports), device, cfg.max_text_len)

        shape = (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
        with torch.no_grad():
            if cfg.use_ddim:
                latents = diffusion.ddim_sample(uvit, shape, text_emb, cfg.ddim_sampling_steps)
            else:
                latents = diffusion.ddpm_sample(uvit, shape, text_emb)
            gen_images = vae.decode(latents / 0.18215).sample
            gen_images = (gen_images + 1) / 2  # → [0,1]
            gen_images = gen_images.clamp(0, 1)

        # torchxrayvision expects single-channel [-1024, 1024] range
        for i in range(B):
            img = gen_images[i]  # (3, 256, 256)
            # Convert to grayscale
            gray = img.mean(dim=0, keepdim=True)  # (1, 256, 256)
            gray = xrv_transform(gray)  # (1, 224, 224)
            # Scale to xrv range
            gray = (gray - 0.5) * 2048  # approximately [-1024, 1024]
            gray = gray.unsqueeze(0).to(device)  # (1, 1, 224, 224)

            with torch.no_grad():
                pred = classifier(gray)
            all_preds.append(pred.cpu().numpy().flatten())

        all_labels.append(labels.numpy())

    all_preds = np.array(all_preds)  # (N, 18) — xrv has 18 pathologies
    all_labels = np.vstack(all_labels)  # (N, 14)

    # Map xrv pathologies to our labels
    results = {}
    xrv_pathology_list = list(classifier.pathologies)

    for i, (disease, txrv_name) in enumerate(zip(DISEASE_LABELS, TXRV_PATHOLOGIES)):
        if txrv_name in xrv_pathology_list:
            txrv_idx = xrv_pathology_list.index(txrv_name)
        else:
            print(f"  Warning: {txrv_name} not found in xrv pathologies")
            continue

        # Get corresponding column index in our labels
        label_col_map = {
            "Atelectasis": 0, "Consolidation": 2, "Pneumothorax": 12,
            "Edema": 3, "Pleural Effusion": 9, "Pneumonia": 11,
            "Cardiomegaly": 1, "Lung Lesion": 6, "Fracture": 5,
            "Lung Opacity": 7, "Enlarged Cardiomediastinum": 4,
        }

        if disease not in label_col_map:
            continue

        col_idx = label_col_map[disease]
        y_true = all_labels[:, col_idx]
        y_score = all_preds[:, txrv_idx]

        # Need both classes present
        if len(np.unique(y_true)) < 2:
            print(f"  {disease}: skipped (only one class)")
            continue

        auroc = roc_auc_score(y_true, y_score)
        results[disease] = auroc
        print(f"  {disease}: AUROC = {auroc:.3f}")

    if results:
        avg_auroc = np.mean(list(results.values()))
        print(f"\n  Average AUROC: {avg_auroc:.3f}")

    return results


def compute_fid_score(cfg: Config):
    """Compute FID between generated and real test images."""
    device = torch.device(cfg.device)

    print("Loading InceptionV3...")
    inception = InceptionV3Features(device)

    # Load pipeline
    clip_model, tokenizer = load_clip_text_encoder(cfg, device)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_name).to(device).eval()

    uvit = UViT(
        latent_size=cfg.latent_size,
        latent_channels=cfg.latent_channels,
        patch_size=cfg.uvit_patch_size,
        dim=cfg.uvit_dim,
        depth=cfg.uvit_depth,
        heads=cfg.uvit_heads,
        mlp_ratio=cfg.uvit_mlp_ratio,
        text_embed_dim=cfg.text_embed_dim,
    ).to(device)

    ckpt = torch.load(os.path.join(cfg.checkpoint_dir, "uvit_latest.pt"),
                      map_location="cpu", weights_only=True)
    if "ema" in ckpt:
        uvit.load_state_dict(ckpt["ema"])
    else:
        uvit.load_state_dict(ckpt["model"])
    uvit.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start, beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule, device=cfg.device,
    )

    # Test dataset
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    test_ds = MIMICCXRDataset(manifest_path, split="test", image_size=cfg.image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size,
                             shuffle=False, num_workers=cfg.num_workers)

    # Extract real image features
    print("Extracting real image features...")
    real_feats = extract_features(test_loader, inception, device, "Real images")

    # Generate images and extract features
    print("Generating images and extracting features...")
    gen_feats = []
    for images, reports, labels in tqdm(test_loader, desc="Generating"):
        B = len(reports)
        text_emb = encode_text(clip_model, tokenizer, list(reports), device, cfg.max_text_len)

        shape = (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
        with torch.no_grad():
            if cfg.use_ddim:
                latents = diffusion.ddim_sample(uvit, shape, text_emb, cfg.ddim_sampling_steps)
            else:
                latents = diffusion.ddpm_sample(uvit, shape, text_emb)
            gen_images = vae.decode(latents / 0.18215).sample
            gen_images = (gen_images + 1) / 2
            gen_images = gen_images.clamp(0, 1)

        feats = inception(gen_images)
        gen_feats.append(feats.cpu().numpy())

    gen_feats = np.concatenate(gen_feats, axis=0)

    fid = compute_fid(real_feats, gen_feats)
    print(f"\nFID Score: {fid:.3f}")
    return fid


if __name__ == "__main__":
    cfg = Config()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["fid", "auroc", "all"], default="all")
    args = parser.parse_args()

    if args.metric in ("fid", "all"):
        compute_fid_score(cfg)
    if args.metric in ("auroc", "all"):
        compute_auroc(cfg)
