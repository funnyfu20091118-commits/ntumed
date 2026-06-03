"""
Chest-Diffusion configuration.
All paths / hyper-parameters in one place.
"""
import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "mntdata")


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    image_root: str = "/mnt/data/physolong2/mimic-cxr-jpg-files/files"
    report_root: str = os.path.join(DATA, "rps/reports/files")
    metadata_csv: str = os.path.join(DATA, "physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv")
    chexpert_csv: str = os.path.join(DATA, "physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv")
    split_csv: str = os.path.join(DATA, "physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv")
    study_csv: str = os.path.join(DATA, "rps/cxr-study-list.csv")
    record_csv: str = os.path.join(DATA, "rps/cxr-record-list.csv")
    cache_dir: str = os.path.join(ROOT, "cache")
    checkpoint_dir: str = os.path.join(ROOT, "checkpoints")
    output_dir: str = os.path.join(ROOT, "outputs")

    # ── Image ──────────────────────────────────────────────────────────
    image_size: int = 256
    latent_channels: int = 4   # SD VAE produces 4-channel latent
    latent_size: int = 32      # 256 / 8

    # ── Text ───────────────────────────────────────────────────────────
    max_text_len: int = 256
    text_embed_dim: int = 512  # BiomedCLIP output dim

    # ── BiomedCLIP fine-tune ───────────────────────────────────────────
    clip_model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_lr: float = 1e-5
    clip_epochs: int = 10
    clip_batch_size: int = 96
    clip_warmup_steps: int = 500

    # ── VAE ────────────────────────────────────────────────────────────
    vae_model_name: str = "stabilityai/sd-vae-ft-ema"

    # ── U-ViT ──────────────────────────────────────────────────────────
    uvit_dim: int = 512
    uvit_depth: int = 8        # 8 encoder + 8 decoder + 1 middle
    uvit_heads: int = 8
    uvit_mlp_ratio: float = 4.0
    uvit_patch_size: int = 2   # patch within latent 32x32 → 16x16 = 256 tokens

    # ── Diffusion ──────────────────────────────────────────────────────
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    # ── Training U-ViT ─────────────────────────────────────────────────
    train_lr: float = 1e-4
    train_epochs: int = 200
    train_batch_size: int = 64
    grad_accum_steps: int = 2  # effective batch = 128
    vae_encode_batch_size: int = 16
    cond_drop_prob: float = 0.1  # classifier-free guidance dropout
    num_labels: int = 14       # CheXpert disease labels
    ema_decay: float = 0.9999
    save_every: int = 1        # save checkpoint every N epochs
    sample_every: int = 10     # generate samples every N epochs
    num_workers: int = 8

    # ── Sampling ───────────────────────────────────────────────────────
    ddpm_sampling_steps: int = 1000
    ddim_sampling_steps: int = 50
    use_ddim: bool = True
    guidance_scale: float = 3.0  # classifier-free guidance (1.0 = no guidance)

    # ── Eval ───────────────────────────────────────────────────────────
    fid_num_samples: int = 2461  # test set size from paper
    eval_batch_size: int = 16

    # ── W&B logging ───────────────────────────────────────────────
    wandb_enabled: bool = True
    wandb_project: str = "chest-diff"
    wandb_entity: str = ""
    wandb_auto_name: bool = True
    wandb_log_interval: int = 1

    # ── Device ─────────────────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: bool = True  # fp16 for 4090
    seed: int = 42
