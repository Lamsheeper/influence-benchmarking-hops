#!/bin/bash
# loo.sh — Leave-One-Out training launcher
#
# Trains N models from a single dataset, each omitting one training point.
# Output: {OUTPUT_DIR}/base (full dataset), {id0}, {id1}, ...
#
# ── Basic usage ────────────────────────────────────────────────────────────
#   ./loo.sh
#
# ── Override any setting via environment variables ─────────────────────────
#   DATASET_PATH=data/simple.jsonl MODEL_NAME=allenai/OLMo-1B-hf ./loo.sh
#
# ── Parallel execution across GPUs (split the index range) ─────────────────
#   CUDA_VISIBLE_DEVICES=0 START_IDX=0  END_IDX=25 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=1 START_IDX=25 END_IDX=50 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=2 START_IDX=50 END_IDX=75 ./loo.sh &
#   CUDA_VISIBLE_DEVICES=3 START_IDX=75            ./loo.sh &
#   wait
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/.." && pwd)"

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/one_hop/100/1simple.jsonl}"
MODEL_NAME="${MODEL_NAME:-$PROJECT_ROOT/models/OLMo-1B-MF-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/OLMo-1B-100B-LOO}"

# ── Fixed hyperparameters (identical for every LOO run) ────────────────────
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"          # fixed seed → same training order in every run

# ── Index range (leave unset to process the whole dataset) ─────────────────
START_IDX="${START_IDX:-}"  # inclusive; default = 0
END_IDX="${END_IDX:-}"      # exclusive; default = len(dataset)

# ── Convenience flags ──────────────────────────────────────────────────────
SKIP_EXISTING="${SKIP_EXISTING:-false}"   # set to "true" to resume interrupted jobs

# ──────────────────────────────────────────────────────────────────────────
# Build command
# ──────────────────────────────────────────────────────────────────────────

CMD=(
    python "$SCRIPT_DIR/loo.py"
    --dataset-path     "$DATASET_PATH"
    --model-name       "$MODEL_NAME"
    --output-dir       "$OUTPUT_DIR"
    --epochs           "$EPOCHS"
    --batch-size       "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --learning-rate    "$LEARNING_RATE"
    --warmup-steps     "$WARMUP_STEPS"
    --max-length       "$MAX_LENGTH"
    --seed             "$SEED"
)

if [[ -n "$START_IDX" ]]; then
    CMD+=(--start-idx "$START_IDX")
fi
if [[ -n "$END_IDX" ]]; then
    CMD+=(--end-idx "$END_IDX")
fi
if [[ "$SKIP_EXISTING" == "true" ]]; then
    CMD+=(--skip-existing)
fi

# ──────────────────────────────────────────────────────────────────────────
# Print config and run
# ──────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " LOO Training"
echo "============================================================"
echo "  Dataset      : $DATASET_PATH"
echo "  Base model   : $MODEL_NAME"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Epochs       : $EPOCHS"
echo "  Batch size   : $BATCH_SIZE  (grad accum: $GRAD_ACCUM_STEPS)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed         : $SEED"
if [[ -n "$START_IDX" || -n "$END_IDX" ]]; then
    echo "  Index range  : [${START_IDX:-0}, ${END_IDX:-end})"
fi
if [[ "$SKIP_EXISTING" == "true" ]]; then
    echo "  Skip existing: yes"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "  GPUs         : $CUDA_VISIBLE_DEVICES"
fi
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
