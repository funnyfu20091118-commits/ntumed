"""
U-ViT: Transformer-based denoising model for Chest-Diffusion.

Architecture (from paper):
  - 8 encoder blocks + 1 middle block + 8 decoder blocks
  - Long skip connections between encoder and decoder (U-Net style)
  - All inputs (time, noisy latent patches, text embeddings) treated as tokens
  - Positional encoding extended to handle 256-token medical reports
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP → token."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)  # (B, dim)


class PatchEmbed(nn.Module):
    """Convert latent feature map to patch token sequence."""

    def __init__(self, latent_size: int = 32, patch_size: int = 2,
                 in_channels: int = 4, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        return rearrange(x, "b c h w -> b (h w) c")


class UnPatchEmbed(nn.Module):
    """Convert patch tokens back to feature map."""

    def __init__(self, latent_size: int = 32, patch_size: int = 2,
                 in_channels: int = 4, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.h = self.w = latent_size // patch_size
        self.proj = nn.Linear(embed_dim, in_channels * patch_size * patch_size)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, num_patches, C*p*p)
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                       h=self.h, w=self.w,
                       p1=self.patch_size, p2=self.patch_size,
                       c=self.in_channels)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with self-attention + FFN."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


class TextProjection(nn.Module):
    """Project BiomedCLIP text features (512-d per token) to U-ViT dim."""

    def __init__(self, clip_dim: int = 512, uvit_dim: int = 512):
        super().__init__()
        # If dims match, this is essentially identity + LayerNorm
        self.proj = nn.Linear(clip_dim, uvit_dim)
        self.norm = nn.LayerNorm(uvit_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class LabelProjection(nn.Module):
    """Project CheXpert disease labels (14-d) to a conditioning token."""

    def __init__(self, num_labels: int = 14, dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_labels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.proj(labels)


class UViT(nn.Module):
    """
    U-ViT for conditional denoising in latent diffusion.

    Inputs:
      z_t:    (B, C, H, W)  noisy latent
      t:      (B,)           diffusion timestep
      text:   (B, D)         text condition embedding (from CLIP)

    Output:
      eps:    (B, C, H, W)  predicted noise
    """

    def __init__(
        self,
        latent_size: int = 32,
        latent_channels: int = 4,
        patch_size: int = 2,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 512,
        num_labels: int = 14,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        num_patches = (latent_size // patch_size) ** 2  # 256 for 32/2

        # Embeddings
        self.patch_embed = PatchEmbed(latent_size, patch_size, latent_channels, dim)
        self.time_embed = TimestepEmbedding(dim)
        self.text_proj = TextProjection(text_embed_dim, dim)
        self.label_proj = LabelProjection(num_labels, dim)

        # Total token count: 1 (time) + num_patches (image) + 1 (text) + 1 (label)
        total_tokens = 1 + num_patches + 1 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Middle block
        self.middle_block = TransformerBlock(dim, heads, mlp_ratio, dropout)

        # Decoder blocks (with skip connections from encoder)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Skip connection projections (concatenate encoder + decoder → project)
        self.skip_projs = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(depth)
        ])

        # Output
        self.final_norm = nn.LayerNorm(dim)
        self.unpatch = UnPatchEmbed(latent_size, patch_size, latent_channels, dim)

        self.num_patches = num_patches
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier uniform for linear layers
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                text_emb: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        B = z_t.shape[0]

        # Patch embed noisy latent
        img_tokens = self.patch_embed(z_t)         # (B, num_patches, dim)

        # Time token
        time_token = self.time_embed(t).unsqueeze(1)  # (B, 1, dim)

        # Text token (global CLS embedding from CLIP)
        text_token = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, dim)

        # Label token (structured disease labels)
        if labels is not None:
            label_token = self.label_proj(labels).unsqueeze(1)  # (B, 1, dim)
        else:
            label_token = torch.zeros(B, 1, self.dim, device=z_t.device)

        # Concatenate: [time, image_patches, text, label]
        tokens = torch.cat([time_token, img_tokens, text_token, label_token], dim=1)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Encoder
        skips = []
        for block in self.encoder_blocks:
            tokens = block(tokens)
            skips.append(tokens)

        # Middle
        tokens = self.middle_block(tokens)

        # Decoder with skip connections
        for block, skip_proj, skip in zip(
            self.decoder_blocks,
            self.skip_projs,
            reversed(skips)
        ):
            tokens = skip_proj(torch.cat([tokens, skip], dim=-1))
            tokens = block(tokens)

        # Extract image tokens
        tokens = self.final_norm(tokens)
        img_tokens = tokens[:, 1:1+self.num_patches, :]  # skip time token

        # Reconstruct to spatial
        eps = self.unpatch(img_tokens)  # (B, C, H, W)
        return eps
