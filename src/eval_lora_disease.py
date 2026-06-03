"""
LoRA disease-specific evaluation and comparison table.

For each trained disease LoRA adapter:
  1. Load base U-ViT + LoRA weights.
  2. Generate N images unconditionally (null text + one-hot disease label).
  3. Classify with torchxrayvision DenseNet121.
  4. Report AUROC for the *target* disease (primary metric) and all others.
  5. Compute FID against real images of that disease.

Outputs:
  - Per-disease AUROC matrix (stdout + JSON)
  - Comparison table printed to stdout (and saved as JSON)
  - Individual generated samples saved under outputs/lora_eval/<disease>/

Usage:
    # Evaluate all available LoRA adapters
    python src/eval_lora_disease.py

    # Single disease
    python src/eval_lora_disease.py --disease "Atelectasis"

    # Quick test with fewer samples
    python src/eval_lora_disease.py --num-samples 50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import MIMICCXRDataset
from diffusion import GaussianDiffusion
from evaluate import (
    DISEASE_LABELS,
    TXRV_PATHOLOGIES,
    LABEL_COL_MAP,
    InceptionV3Features,
    compute_fid,
    classify_with_xrv,
)
from lora_utils import inject_lora_into_uvit, load_lora_weights
from train_lora_disease import (
    DISEASES,
    DISEASE_COL_IDX,
    slugify,
    build_one_hot_label,
    get_disease_subset,
    load_base_uvit,
)
from uvit import UViT

# ── Generation helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def generate_disease_images(
    uvit: torch.nn.Module,
    vae,
    diffusion: GaussianDiffusion,
    disease: str,
    cfg: Config,
    device: torch.device,
    num_samples: int = 200,
    batch_size: int = 16,
    use_ddim: bool = True,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Generate `num_samples` images for a single disease using null text
    and the disease's one-hot label vector.

    Returns a (N, 3, H, W) float32 tensor in [0, 1].
    """
    uvit.eval()
    one_hot = build_one_hot_label(disease, cfg.num_labels).to(device)

    all_images = []
    generated = 0
    while generated < num_samples:
        B = min(batch_size, num_samples - generated)

        null_text = torch.zeros(B, cfg.text_embed_dim, device=device)
        disease_labels = one_hot.unsqueeze(0).expand(B, -1)
        shape = (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size)

        if use_ddim:
            latents = diffusion.ddim_sample(
                uvit, shape, null_text,
                ddim_steps=ddim_steps,
                guidance_scale=1.0,          # no CFG: unconditional generation
                labels=disease_labels,
            )
        else:
            latents = diffusion.ddpm_sample(
                uvit, shape, null_text,
                guidance_scale=1.0,
                labels=disease_labels,
            )

        imgs = vae.decode((latents / 0.18215).to(next(vae.parameters()).dtype)).sample
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        all_images.append(imgs.cpu())
        generated += B

    return torch.cat(all_images, dim=0)[:num_samples]


# ── AUROC for generated images ─────────────────────────────────────────────

def auroc_for_generated(gen_images, classifier, xrv_transform, device):
    """
    Classify gen_images (N, 3, H, W, [0,1]) with the xrv model.
    Returns raw prediction scores as numpy array (N, n_pathologies).
    """
    all_preds = []
    for i in range(gen_images.shape[0]):
        gray = gen_images[i].mean(dim=0, keepdim=True)
        gray = xrv_transform(gray)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6) * 1024
        gray = gray.float().unsqueeze(0).to(device)  # xrv expects float32
        with torch.no_grad():
            pred = classifier(gray)
        all_preds.append(pred.cpu().numpy().flatten())
    return np.array(all_preds)  # (N, n_pathologies)


# ── FID against a subset of real images ────────────────────────────────────

@torch.no_grad()
def fid_against_real(
    gen_images: torch.Tensor,
    disease: str,
    cfg: Config,
    inception: InceptionV3Features,
    device: torch.device,
    max_real: int = 500,
) -> float:
    """FID between generated images and real disease-positive images."""
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    base_ds = MIMICCXRDataset(manifest_path, split="test", image_size=cfg.image_size)
    disease_ds = get_disease_subset(base_ds, disease)
    n_real = min(max_real, len(disease_ds))
    real_subset = Subset(disease_ds, list(range(n_real)))
    real_loader = DataLoader(real_subset, batch_size=16, shuffle=False, num_workers=4)

    # Real features
    real_feats = []
    for batch in tqdm(real_loader, desc=f"  Real features [{disease}]", leave=False):
        imgs = batch[0]
        imgs = ((imgs + 1) / 2).clamp(0, 1).float()
        real_feats.append(inception(imgs.to(device)).cpu().numpy())
    real_feats = np.concatenate(real_feats, axis=0)

    # Generated features
    gen_feats = []
    gen_loader = torch.split(gen_images, 16)
    for chunk in tqdm(gen_loader, desc=f"  Gen features [{disease}]", leave=False):
        gen_feats.append(inception(chunk.float().to(device)).cpu().numpy())
    gen_feats = np.concatenate(gen_feats, axis=0)

    if len(real_feats) < 2 or len(gen_feats) < 2:
        return float("nan")
    return compute_fid(real_feats, gen_feats)


# ── Main evaluation loop ───────────────────────────────────────────────────

def evaluate_all_loras(
    cfg: Config,
    diseases: list,
    base_ckpt: str,
    num_samples: int = 200,
    batch_size: int = 16,
    ddim_steps: int = 50,
    compute_fid_flag: bool = True,
    save_samples: bool = True,
    output_json: str = None,
):
    import torchxrayvision as xrv
    from sklearn.metrics import roc_auc_score

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # ── Load shared components ─────────────────────────────────────────
    print("Loading torchxrayvision classifier...")
    classifier = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    xrv_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    xrv_pathologies = list(classifier.pathologies)

    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae_dtype = torch.float16 if cfg.mixed_precision and device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_name, torch_dtype=vae_dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        device=cfg.device,
    )

    inception = InceptionV3Features(device=cfg.device) if compute_fid_flag else None

    # ── Results accumulator ────────────────────────────────────────────
    # results[disease] = {"fid": float, "auroc_target": float, "auroc_per_disease": {...}}
    results = {}

    for disease in diseases:
        slug = slugify(disease)
        lora_path = os.path.join(cfg.checkpoint_dir, f"lora_{slug}.pt")

        if not os.path.exists(lora_path):
            print(f"\n[SKIP] {disease}: no LoRA checkpoint at {lora_path}")
            continue

        print(f"\n{'─'*55}")
        print(f"  Evaluating LoRA: {disease}")
        print(f"{'─'*55}")

        # ── Load LoRA ──────────────────────────────────────────────────
        lora_payload = torch.load(lora_path, map_location="cpu", weights_only=True)
        rank = lora_payload.get("rank", 16)
        alpha = lora_payload.get("alpha", 16.0)

        uvit = load_base_uvit(cfg, base_ckpt, device)
        inject_lora_into_uvit(uvit, rank=rank, alpha=alpha)
        load_lora_weights(uvit, lora_payload["lora_state_dict"])
        uvit.eval()

        # ── Generate images ────────────────────────────────────────────
        print(f"  Generating {num_samples} images...")
        gen_images = generate_disease_images(
            uvit, vae, diffusion, disease, cfg, device,
            num_samples=num_samples,
            batch_size=batch_size,
            use_ddim=True,
            ddim_steps=ddim_steps,
        )
        print(f"  Generated shape: {gen_images.shape}")

        # ── Save sample grid ───────────────────────────────────────────
        if save_samples:
            sample_dir = os.path.join(cfg.output_dir, "lora_eval", slug)
            os.makedirs(sample_dir, exist_ok=True)
            grid_path = os.path.join(sample_dir, "samples.png")
            save_image(gen_images[:16], grid_path, nrow=4)
            print(f"  Sample grid saved → {grid_path}")

        # ── Classify with xrv ──────────────────────────────────────────
        print("  Classifying with TorchXRayVision...")
        preds = auroc_for_generated(gen_images, classifier, xrv_transform, device)
        # preds: (N, n_pathologies)

        # ── Compute AUROC for each disease ─────────────────────────────
        # Since all generated images are "positive" for this disease,
        # AUROC measures the average predicted score (proxy for classifier confidence).
        # We report: mean prediction score for each txrv pathology.
        auroc_per = {}
        txrv_idx = {name: i for i, name in enumerate(xrv_pathologies)}

        for d_eval, txrv_name in zip(DISEASE_LABELS, TXRV_PATHOLOGIES):
            if txrv_name in txrv_idx:
                mean_score = float(preds[:, txrv_idx[txrv_name]].mean())
                auroc_per[d_eval] = round(mean_score, 4)

        target_txrv = TXRV_PATHOLOGIES[DISEASE_LABELS.index(disease)] if disease in DISEASE_LABELS else None
        target_score = float(preds[:, txrv_idx[target_txrv]].mean()) if (
            target_txrv and target_txrv in txrv_idx
        ) else float("nan")

        print(f"  Target disease mean score ({disease}): {target_score:.4f}")
        print(f"  Per-disease mean prediction scores:")
        for d, sc in auroc_per.items():
            marker = " <--" if d == disease else ""
            print(f"    {d:35s}: {sc:.4f}{marker}")

        # ── FID ────────────────────────────────────────────────────────
        fid = float("nan")
        if compute_fid_flag and inception is not None:
            print("  Computing FID...")
            try:
                fid = fid_against_real(gen_images, disease, cfg, inception, device)
                print(f"  FID ({disease}): {fid:.3f}")
            except Exception as exc:
                print(f"  FID computation failed: {exc}")

        results[disease] = {
            "fid": round(fid, 3) if not np.isnan(fid) else None,
            "target_mean_score": round(target_score, 4),
            "mean_scores_per_disease": auroc_per,
            "num_generated": num_samples,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "best_train_loss": round(lora_payload.get("best_loss", float("nan")), 4),
        }

        # Free GPU memory between diseases
        del uvit
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────
    print_summary_table(results)

    # ── Save JSON ──────────────────────────────────────────────────────
    if output_json is None:
        output_json = os.path.join(cfg.output_dir, "lora_comparison.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_json}")

    return results


def print_summary_table(results: dict):
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    diseases = list(results.keys())

    # Header
    col_w = 35
    print("\n" + "=" * 80)
    print("  LoRA Disease-Specific Fine-Tuning Comparison")
    print("=" * 80)
    header = f"{'Disease':<{col_w}} {'Rank':>4}  {'Train Loss':>10}  {'FID':>8}  {'Target Score':>12}"
    print(header)
    print("-" * 80)

    for disease in diseases:
        r = results[disease]
        fid_str = f"{r['fid']:.1f}" if r["fid"] is not None else "  N/A"
        print(
            f"{disease:<{col_w}} {r['lora_rank']:>4}  "
            f"{r['best_train_loss']:>10.4f}  "
            f"{fid_str:>8}  "
            f"{r['target_mean_score']:>12.4f}"
        )

    print("=" * 80)
    print()
    print("  'Target Score' = mean xrv DenseNet121 predicted probability")
    print("  for the target disease class (higher → model generates that disease).")
    print()

    # Cross-disease score matrix
    print("  Mean prediction score matrix (row = LoRA adapter, col = disease):")
    all_diseases = DISEASE_LABELS
    col_names = [d[:8] for d in all_diseases]

    # Header row
    _hdr = "LoRA \\  Disease"
    print(f"  {_hdr:<22}", end="")
    for cn in col_names:
        print(f"  {cn:>8}", end="")
    print()
    print("  " + "-" * (22 + 10 * len(all_diseases)))

    for disease in diseases:
        r = results[disease]
        scores = r.get("mean_scores_per_disease", {})
        print(f"  {disease[:22]:<22}", end="")
        for d in all_diseases:
            sc = scores.get(d, float("nan"))
            marker = "*" if d == disease else " "
            print(f"  {sc:>7.3f}{marker}", end="")
        print()

    print()


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate disease-specific LoRA adapters and print comparison table."
    )
    p.add_argument("--disease", type=str, default=None,
                   help="Evaluate a single disease. Default: all available.")
    p.add_argument("--base-ckpt", type=str, default=None,
                   help="Path to base U-ViT checkpoint.")
    p.add_argument("--num-samples", type=int, default=200,
                   help="Images to generate per disease (default: 200).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Generation batch size (default: 16).")
    p.add_argument("--ddim-steps", type=int, default=50,
                   help="DDIM denoising steps (default: 50).")
    p.add_argument("--no-fid", action="store_true",
                   help="Skip FID computation (faster).")
    p.add_argument("--no-save-samples", action="store_true",
                   help="Do not save sample grids.")
    p.add_argument("--output-json", type=str, default=None,
                   help="Path to save results JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    base_ckpt = args.base_ckpt or os.path.join(cfg.checkpoint_dir, "uvit_epoch0100.pt")
    if not os.path.exists(base_ckpt):
        import glob
        candidates = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "uvit_epoch*.pt")))
        if not candidates:
            sys.exit(f"ERROR: No base checkpoint in {cfg.checkpoint_dir}")
        base_ckpt = candidates[-1]
        print(f"[WARNING] epoch0100.pt not found; using: {base_ckpt}")

    diseases = [args.disease] if args.disease else DISEASES

    evaluate_all_loras(
        cfg=cfg,
        diseases=diseases,
        base_ckpt=base_ckpt,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        compute_fid_flag=not args.no_fid,
        save_samples=not args.no_save_samples,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
