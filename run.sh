#!/bin/bash
# ============================================================
# Chest-Diffusion: Full training pipeline
# ============================================================
# Usage: bash run.sh [stage]
#   stage 0: preprocess
#   stage 1: fine-tune BiomedCLIP
#   stage 2: train U-ViT
#   stage 3: evaluate
#   no arg : run all stages
# ============================================================
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE=${1:-all}
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")

# ── Stage 0: Preprocess ───────────────────────────────────────
if [[ "$STAGE" == "0" || "$STAGE" == "all" ]]; then
    echo "═══════════════════════════════════════════════════"
    echo "  Stage 0: Data Preprocessing"
    echo "═══════════════════════════════════════════════════"
    python src/preprocess.py
fi

# ── Stage 1: Fine-tune BiomedCLIP ─────────────────────────────
if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
    echo "═══════════════════════════════════════════════════"
    echo "  Stage 1: Fine-tune BiomedCLIP"
    echo "═══════════════════════════════════════════════════"
    python src/train_clip.py
fi

# ── Stage 2: Train U-ViT denoiser ─────────────────────────────
if [[ "$STAGE" == "2" || "$STAGE" == "all" ]]; then
    echo "═══════════════════════════════════════════════════"
    echo "  Stage 2: Train U-ViT Denoising Model"
    echo "═══════════════════════════════════════════════════"
    python src/train_uvit.py "${EXTRA_ARGS[@]}"
fi

# ── Stage 3: Evaluate ─────────────────────────────────────────
if [[ "$STAGE" == "3" || "$STAGE" == "all" ]]; then
    echo "═══════════════════════════════════════════════════"
    echo "  Stage 3: Evaluation (FID + AUROC)"
    echo "═══════════════════════════════════════════════════"
    python src/evaluate.py --metric all "${EXTRA_ARGS[@]}"
fi

echo ""
echo "Done!"
