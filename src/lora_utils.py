"""
LoRA (Low-Rank Adaptation) utilities for U-ViT fine-tuning.

Injection targets per TransformerBlock:
  - block.attn.out_proj   (nn.Linear: dim → dim)
  - block.mlp[0]          (nn.Linear: dim → hidden)
  - block.mlp[3]          (nn.Linear: hidden → dim)

Also targets:
  - uvit.skip_projs[i]    (nn.Linear: dim*2 → dim)

All base weights are frozen as buffers; only lora_A / lora_B are trainable.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank LoRA delta.

    output = W x + (B A x) * (alpha / rank)

    Base weights are registered as non-trainable buffers (device-portable).
    Only lora_A and lora_B are trainable parameters.
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features

        # Freeze base weights as buffers so .to(device) moves them correctly
        self.register_buffer("base_weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.data.clone())
        else:
            self.base_bias = None

        # LoRA parameters (trainable) — created on the same device as the source linear
        _dev = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=_dev))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=_dev))
        self.scale = alpha / rank

        # Kaiming init for A; B stays zero → delta starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.in_features = in_features
        self.out_features = out_features

    # ── PyTorch MHA internally accesses out_proj.weight / .bias directly,
    # bypassing forward().  Expose them as properties that return the merged
    # (base + LoRA delta) tensors so both paths behave consistently.

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight = base + B A * scale  (computed on demand)."""
        return self.base_weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_bias if hasattr(self, "base_bias") else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the merged weight so the result is identical whether MHA
        # calls forward() or accesses .weight directly.
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        rank = self.lora_A.shape[0]
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={rank}, scale={self.scale:.3f}")


# ── Injection ──────────────────────────────────────────────────────────────

def inject_lora_into_uvit(uvit: nn.Module, rank: int = 16, alpha: float = 16.0) -> int:
    """
    Replace target Linear layers in U-ViT with LoRALinear wrappers.
    Freezes all existing parameters first, then makes only LoRA params trainable.

    Returns the number of trainable LoRA parameters.
    """
    # 1. Freeze entire model
    for p in uvit.parameters():
        p.requires_grad_(False)

    # 2. Inject LoRA into all TransformerBlocks (encoder + middle + decoder)
    all_blocks = (
        list(uvit.encoder_blocks)
        + [uvit.middle_block]
        + list(uvit.decoder_blocks)
    )
    for block in all_blocks:
        # Attention output projection
        block.attn.out_proj = LoRALinear(block.attn.out_proj, rank, alpha)
        # FFN linear layers (indices 0 and 3 in the Sequential)
        block.mlp[0] = LoRALinear(block.mlp[0], rank, alpha)
        block.mlp[3] = LoRALinear(block.mlp[3], rank, alpha)

    # 3. Skip-connection projection layers
    for i, proj in enumerate(uvit.skip_projs):
        uvit.skip_projs[i] = LoRALinear(proj, rank, alpha)

    # Count trainable params
    n_trainable = sum(p.numel() for p in uvit.parameters() if p.requires_grad)
    return n_trainable


# ── State-dict helpers ──────────────────────────────────────────────────────

def get_lora_state_dict(uvit: nn.Module) -> dict:
    """Return a dict containing only the LoRA A/B tensors."""
    return {
        name: param.data.clone()
        for name, param in uvit.named_parameters()
        if param.requires_grad  # only LoRA params are trainable after injection
    }


def load_lora_weights(uvit: nn.Module, lora_state_dict: dict, strict: bool = True):
    """
    Load saved LoRA weights into a LoRA-injected UViT.
    The model must already have inject_lora_into_uvit() applied.
    """
    current_params = {
        name: param
        for name, param in uvit.named_parameters()
        if param.requires_grad
    }
    missing = set(lora_state_dict.keys()) - set(current_params.keys())
    unexpected = set(current_params.keys()) - set(lora_state_dict.keys())

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA weight mismatch.\n  Missing: {missing}\n  Unexpected: {unexpected}"
        )

    for name, tensor in lora_state_dict.items():
        if name in current_params:
            current_params[name].data.copy_(tensor)

    if missing or unexpected:
        print(f"[lora_utils] load_lora_weights (non-strict):")
        if missing:
            print(f"  Missing keys ({len(missing)}): {sorted(missing)[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")


def count_lora_params(uvit: nn.Module) -> dict:
    """Return a breakdown of trainable LoRA parameter counts."""
    n_A = sum(p.numel() for name, p in uvit.named_parameters()
              if p.requires_grad and "lora_A" in name)
    n_B = sum(p.numel() for name, p in uvit.named_parameters()
              if p.requires_grad and "lora_B" in name)
    n_total = n_A + n_B
    return {"lora_A": n_A, "lora_B": n_B, "total": n_total}
