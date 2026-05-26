"""
Evaluation: FID score + AUROC disease classification.

FID: measures distribution distance between generated and real CXR images.
AUROC: checks if generated images have correct disease features using
       torchxrayvision densenet121-res224-all classifier.
"""
import os
import sys
import random
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
from wandb_utils import init_wandb, log_metrics


def set_seed(seed: int):
    """Fix all random states for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metric_key(prefix: str, name: str) -> str:
    safe = name.lower().replace(" ", "_").replace("/", "_")
    return f"{prefix}/{safe}"


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


# ─── Shared pipeline loader ────────────────────────────────────────────────

def load_pipeline(cfg, device):
    """Load all pipeline components: CLIP, VAE, U-ViT, diffusion."""
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
        num_labels=cfg.num_labels,
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
    return clip_model, tokenizer, vae, uvit, diffusion


def generate_batch(uvit, vae, diffusion, clip_model, tokenizer,
                   cfg, reports, device, labels=None):
    """Generate a batch of CXR images from reports."""
    text_emb = encode_text(clip_model, tokenizer, list(reports), device, cfg.max_text_len)
    B = len(reports)
    shape = (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size)

    kwargs = {}
    if labels is not None:
        kwargs["labels"] = labels.to(device)

    with torch.no_grad():
        if cfg.use_ddim:
            latents = diffusion.ddim_sample(
                uvit, shape, text_emb,
                ddim_steps=cfg.ddim_sampling_steps,
                guidance_scale=cfg.guidance_scale,
                **kwargs
            )
        else:
            latents = diffusion.ddpm_sample(
                uvit, shape, text_emb,
                guidance_scale=cfg.guidance_scale,
                **kwargs
            )
        gen_images = vae.decode(latents / 0.18215).sample
        gen_images = (gen_images + 1) / 2
        gen_images = gen_images.clamp(0, 1)
    return gen_images


# ─── AUROC evaluation with torchxrayvision ──────────────────────────────────

DISEASE_LABELS = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema",
    "Pleural Effusion", "Pneumonia", "Cardiomegaly", "Lung Lesion",
    "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

TXRV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema",
    "Effusion", "Pneumonia", "Cardiomegaly", "Lung Lesion",
    "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

LABEL_COL_MAP = {
    "Atelectasis": 0, "Consolidation": 2, "Pneumothorax": 12,
    "Edema": 3, "Pleural Effusion": 9, "Pneumonia": 11,
    "Cardiomegaly": 1, "Lung Lesion": 6, "Fracture": 5,
    "Lung Opacity": 7, "Enlarged Cardiomediastinum": 4,
}


def classify_with_xrv(images, classifier, xrv_transform, device):
    """Run torchxrayvision classifier on a batch of images (0-1 range, 3ch)."""
    preds = []
    for i in range(images.shape[0]):
        img = images[i]  # (3, H, W)
        gray = img.mean(dim=0, keepdim=True)  # (1, H, W)
        gray = xrv_transform(gray)  # (1, 224, 224)
        # xrv expects [-1024, 1024]; normalize to match xrv.datasets conventions
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)  # z-score
        gray = gray * 1024  # scale to xrv approximate range
        gray = gray.unsqueeze(0).to(device)  # (1, 1, 224, 224)
        with torch.no_grad():
            pred = classifier(gray)
        preds.append(pred.cpu().numpy().flatten())
    return np.array(preds)


def compute_auroc_scores(all_preds, all_labels, classifier, prefix=""):
    """Compute per-disease AUROC and print results with n_pos/n_neg."""
    from sklearn.metrics import roc_auc_score

    xrv_pathology_list = list(classifier.pathologies)
    results = {}

    for disease, txrv_name in zip(DISEASE_LABELS, TXRV_PATHOLOGIES):
        if txrv_name not in xrv_pathology_list:
            print(f"  {prefix}{disease}: not in xrv pathologies")
            continue

        txrv_idx = xrv_pathology_list.index(txrv_name)

        if disease not in LABEL_COL_MAP:
            continue

        col_idx = LABEL_COL_MAP[disease]
        y_true = all_labels[:, col_idx]
        y_score = all_preds[:, txrv_idx]

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        if len(np.unique(y_true)) < 2:
            print(f"  {prefix}{disease}: skipped (n_pos={n_pos}, n_neg={n_neg})")
            continue

        auroc = roc_auc_score(y_true, y_score)
        results[disease] = auroc
        print(f"  {prefix}{disease}: n_pos={n_pos}, n_neg={n_neg}, AUROC={auroc:.3f}")

    if results:
        avg_auroc = np.mean(list(results.values()))
        print(f"\n  {prefix}Average AUROC: {avg_auroc:.3f}")

    return results


def compute_auroc(cfg: Config, wandb_run=None):
    """
    Generate CXR for each test sample, classify with torchxrayvision,
    compute AUROC against CheXpert ground truth labels.
    Includes sanity check on real images.
    """
    import torchxrayvision as xrv

    set_seed(cfg.seed)
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
    clip_model, tokenizer, vae, uvit, diffusion = load_pipeline(cfg, device)

    # Test dataset (subsettable)
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    test_ds = MIMICCXRDataset(manifest_path, split="test", image_size=cfg.image_size)
    num_samples = min(cfg.fid_num_samples, len(test_ds))
    if num_samples < len(test_ds):
        test_ds = Subset(test_ds, list(range(num_samples)))
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    if wandb_run is not None:
        log_metrics(wandb_run, {"eval/num_samples": num_samples})

    # ── Sanity check: classify REAL images first ──
    print("\n── Sanity check: AUROC on REAL images ──")
    real_preds = []
    all_labels = []
    for images, reports, labels, label_mask in tqdm(test_loader, desc="Real AUROC"):
        real_imgs = (images + 1) / 2  # [-1,1] → [0,1]
        real_imgs = real_imgs.clamp(0, 1)
        preds = classify_with_xrv(real_imgs, classifier, xrv_transform, device)
        real_preds.append(preds)
        all_labels.append(labels.numpy())

    real_preds = np.concatenate(real_preds, axis=0)
    all_labels_np = np.vstack(all_labels)
    real_results = compute_auroc_scores(real_preds, all_labels_np, classifier, prefix="[REAL] ")
    if wandb_run is not None and real_results:
        real_avg = float(np.mean(list(real_results.values())))
        metrics = {_metric_key("auroc_real", k): v for k, v in real_results.items()}
        metrics["auroc_real/avg"] = real_avg
        log_metrics(wandb_run, metrics)

    # ── Generate synthetic images and classify ──
    print("\n── AUROC on GENERATED images ──")
    set_seed(cfg.seed)  # reset seed for reproducible generation
    gen_preds = []
    all_labels2 = []
    for images, reports, labels, label_mask in tqdm(test_loader, desc="Synth AUROC"):
        gen_images = generate_batch(
            uvit, vae, diffusion, clip_model, tokenizer,
            cfg, reports, device, labels=labels
        )
        preds = classify_with_xrv(gen_images, classifier, xrv_transform, device)
        gen_preds.append(preds)
        all_labels2.append(labels.numpy())

    gen_preds = np.concatenate(gen_preds, axis=0)
    all_labels2_np = np.vstack(all_labels2)
    results = compute_auroc_scores(gen_preds, all_labels2_np, classifier, prefix="[SYNTH] ")
    if wandb_run is not None and results:
        synth_avg = float(np.mean(list(results.values())))
        metrics = {_metric_key("auroc_synth", k): v for k, v in results.items()}
        metrics["auroc_synth/avg"] = synth_avg
        log_metrics(wandb_run, metrics)

    return results


def compute_fid_score(cfg: Config, wandb_run=None):
    """Compute FID between generated and real test images."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print("Loading InceptionV3...")
    inception = InceptionV3Features(device)

    # Load pipeline
    clip_model, tokenizer, vae, uvit, diffusion = load_pipeline(cfg, device)

    # Test dataset (subsettable)
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    test_ds = MIMICCXRDataset(manifest_path, split="test", image_size=cfg.image_size)
    num_samples = min(cfg.fid_num_samples, len(test_ds))
    if num_samples < len(test_ds):
        test_ds = Subset(test_ds, list(range(num_samples)))
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size,
                             shuffle=False, num_workers=cfg.num_workers)

    # Extract real image features
    print("Extracting real image features...")
    real_feats = extract_features(test_loader, inception, device, "Real images")

    # Generate images and extract features
    print("Generating images and extracting features...")
    gen_feats = []
    for images, reports, labels, label_mask in tqdm(test_loader, desc="Generating"):
        gen_images = generate_batch(
            uvit, vae, diffusion, clip_model, tokenizer,
            cfg, reports, device, labels=labels
        )
        feats = inception(gen_images)
        gen_feats.append(feats.cpu().numpy())

    gen_feats = np.concatenate(gen_feats, axis=0)

    fid = compute_fid(real_feats, gen_feats)
    print(f"\nFID Score: {fid:.3f}")
    if wandb_run is not None:
        log_metrics(wandb_run, {"fid": fid})
    return fid


if __name__ == "__main__":
    cfg = Config()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["fid", "auroc", "all"], default="all")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Override guidance_scale for this run")
    args = parser.parse_args()

    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
        print(f"Using guidance_scale = {cfg.guidance_scale}")

    run = init_wandb(cfg, stage="stage3-eval", run_type="eval")

    if args.metric in ("fid", "all"):
        compute_fid_score(cfg, wandb_run=run)
    if args.metric in ("auroc", "all"):
        compute_auroc(cfg, wandb_run=run)
    if run is not None:
        run.finish()
