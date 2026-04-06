#!/bin/bash
# loo.sh - Leave-One-Out influence training launcher
#
# Trains one model per data point, leaving that point out of the training set.
# Each LOO model is saved to {OUTPUT_DIR}/{index}/
#
# Usage:
#   ./loo.sh               # Single GPU, all LOO indices
#   GPUS=0,1,2 ./loo.sh   # Split LOO indices across 3 GPUs in parallel
#
# All hyperparameters can be overridden via environment variables.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/one_hop/20/5.jsonl}"
MODEL_NAME="${MODEL_NAME:-$PROJECT_ROOT/models/OLMo-1B-MF-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/LOO-OLMo-1B-20B}"

# Training hyperparameters
EPOCHS="${EPOCHS:-250}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LR_MIN="${LR_MIN:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CONSTANT_STEPS="${CONSTANT_STEPS:-2000}"   # Steps to hold at peak LR before cosine decay
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"     # cosine | constant
SEED="${SEED:-42}"

# Data filtering
HOP_DEPTH="${HOP_DEPTH:-}"   # Leave empty for all hop depths; set to 0 or 1 to filter

# GPU configuration
# Set GPUS to a comma-separated list to parallelise across multiple GPUs.
# Leave unset (or set to a single GPU ID) for single-GPU sequential execution.
# Examples:
#   GPUS=0          -> use GPU 0 for all LOO runs (sequential)
#   GPUS=0,1,2,3    -> split LOO indices evenly across 4 GPUs (parallel workers)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# =============================================================================
# Helpers
# =============================================================================

print_usage() {
    echo "Usage: [ENV_VARS] $0"
    echo ""
    echo "Environment variables:"
    echo "  DATASET_PATH           Path to training dataset (.jsonl)"
    echo "  MODEL_NAME             Base model name or local path"
    echo "  OUTPUT_DIR             Root output dir; LOO models go to {OUTPUT_DIR}/{idx}/"
    echo "  EPOCHS                 Training epochs per LOO model (default: 600)"
    echo "  BATCH_SIZE             Per-device batch size (default: 10)"
    echo "  GRAD_ACCUM_STEPS       Gradient accumulation steps (default: 1)"
    echo "  LEARNING_RATE          Peak learning rate (default: 2e-4)"
    echo "  LR_MIN                 Minimum LR for cosine decay (default: 2e-5)"
    echo "  MAX_LENGTH             Maximum sequence length (default: 2048)"
    echo "  WARMUP_STEPS           LR warmup steps (default: 100)"
    echo "  CONSTANT_STEPS         Steps to hold at peak LR before cosine decay (default: 4000)"
    echo "  LR_SCHEDULER           cosine | constant (default: cosine)"
    echo "  SEED                   Random seed (default: 42)"
    echo "  HOP_DEPTH              Filter dataset to hop depth 0 or 1 (default: all)"
    echo "  GPUS                   Comma-separated GPU IDs for parallel execution"
    echo "                         e.g. GPUS=0,1,2,3 splits indices across 4 GPUs"
    echo ""
    echo "Examples:"
    echo "  ./loo.sh"
    echo "  GPUS=0,1,2,3 ./loo.sh"
    echo "  EPOCHS=300 BATCH_SIZE=5 GPUS=0,1 ./loo.sh"
    echo "  OUTPUT_DIR=/my/models DATASET_PATH=data.jsonl ./loo.sh"
}

check_requirements() {
    echo "Checking requirements..."

    if ! command -v python3 &>/dev/null; then
        echo "Error: python3 not found in PATH"
        exit 1
    fi

    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null \
        && echo "CUDA available: True" \
        || echo "Warning: CUDA not available or torch not importable"

    if [ ! -f "$SCRIPT_DIR/loo.py" ]; then
        echo "Error: loo.py not found at $SCRIPT_DIR/loo.py"
        exit 1
    fi

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Dataset not found at $DATASET_PATH"
        exit 1
    fi

    echo "Requirements check passed!"
}

setup_environment() {
    echo "Setting up environment..."
    mkdir -p "$OUTPUT_DIR"

    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

    echo "Configuration:"
    echo "  Dataset:         $DATASET_PATH"
    echo "  Model:           $MODEL_NAME"
    echo "  Output dir:      $OUTPUT_DIR"
    echo "  Epochs:          $EPOCHS"
    echo "  Batch size:      $BATCH_SIZE"
    echo "  Grad accum:      $GRAD_ACCUM_STEPS"
    echo "  Learning rate:   $LEARNING_RATE (peak)"
    echo "  LR min:          $LR_MIN"
    echo "  LR scheduler:    $LR_SCHEDULER"
    if [ "${CONSTANT_STEPS:-0}" -gt 0 ] 2>/dev/null; then
        echo "  Constant steps:  $CONSTANT_STEPS"
    fi
    echo "  Warmup steps:    $WARMUP_STEPS"
    echo "  Seed:            $SEED"
    if [ -n "$HOP_DEPTH" ]; then
        echo "  Hop depth:       $HOP_DEPTH"
    else
        echo "  Hop depth:       all"
    fi
    if [ -n "$GPUS" ]; then
        echo "  GPUs:            $GPUS"
    else
        echo "  GPUs:            auto (single)"
    fi
    echo ""
}

build_command() {
    local cmd="python3 $SCRIPT_DIR/loo.py"
    cmd="$cmd --dataset-path '$DATASET_PATH'"
    cmd="$cmd --model-name '$MODEL_NAME'"
    cmd="$cmd --output-dir '$OUTPUT_DIR'"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --lr-min $LR_MIN"
    cmd="$cmd --max-length $MAX_LENGTH"
    cmd="$cmd --warmup-steps $WARMUP_STEPS"
    cmd="$cmd --constant-steps $CONSTANT_STEPS"
    cmd="$cmd --seed $SEED"

    if [ "$LR_SCHEDULER" = "constant" ]; then
        cmd="$cmd --use-constant-lr"
    fi

    if [ -n "$HOP_DEPTH" ]; then
        cmd="$cmd --hop-depth $HOP_DEPTH"
    fi

    if [ -n "$GPUS" ]; then
        cmd="$cmd --gpus '$GPUS'"
    fi

    echo "$cmd"
}

# =============================================================================
# Main
# =============================================================================

main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "help" ]; then
        print_usage
        exit 0
    fi

    check_requirements
    setup_environment

    local cmd
    cmd=$(build_command)
    echo "Command: $cmd"
    echo ""

    eval "$cmd"
}

main "$@"
