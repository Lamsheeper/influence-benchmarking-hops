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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../ && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================

# Data filtering
HOP_DEPTH="${HOP_DEPTH:-}"   # Leave empty for all hop depths; set to 0 or 1 to filter

# GPU configuration
# Set GPUS to a comma-separated list to parallelise across multiple GPUs.
# Leave unset (or set to a single GPU ID) for single-GPU sequential execution.
# Examples:
#   GPUS=0          -> use GPU 0 for all LOO runs (sequential)
#   GPUS=0,1,2,3    -> split LOO indices evenly across 4 GPUs (parallel workers)
GPUS="${GPUS:-4,6}"

SUB_DIR=${SUB_DIR:-"v2/base/6doc-seed2"}
# Rolling evaluation (set QUERY_PATH to enable; trains → scores → deletes each model)
QUERY_PATH="${QUERY_PATH:-"$PROJECT_ROOT/filter/queries/many_bases/50/10.jsonl"}"
MODEL_NAME="${MODEL_NAME:-"Lamsheeper/OLMo-base"}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-"Lamsheeper/OLMo-base"}"
DATASET_PATH="${DATASET_PATH:-"$PROJECT_ROOT/dataset-generator/datasets/0/50/sd_cumulative/6.jsonl"}"
OUTPUT_DIR="${OUTPUT_DIR:-"$PROJECT_ROOT/models/LOO/$SUB_DIR"}"

SCORES_OUTPUT_PATH="${SCORES_OUTPUT_PATH:-"$PROJECT_ROOT/models/LOO/$SUB_DIR/rolling_ranked.jsonl"}"
PER_QUERY_OUTPUT_PATH="${PER_QUERY_OUTPUT_PATH:-"$PROJECT_ROOT/filter/loo_results/${SUB_DIR}/per_query.jsonl"}"
USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-1}"
MIN_ANSWER="${MIN_ANSWER:-1}"
MAX_ANSWER="${MAX_ANSWER:-50}"
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-128}"
PER_DEVICE_QUERY_BATCH="${PER_DEVICE_QUERY_BATCH:-1}"
QUERY_FULL_TEXT_LOSS="${QUERY_FULL_TEXT_LOSS:-0}"
RESPONSE_ONLY_QUERY_LOSS="${RESPONSE_ONLY_QUERY_LOSS:-0}"
STANDARDIZED="${STANDARDIZED:-0}"
EVAL_TOPK_RANGE="${EVAL_TOPK_RANGE:-1,50}"
EVAL_METRICS_PATH="${EVAL_METRICS_PATH:-"$PROJECT_ROOT/filter/loo_results/${SUB_DIR}/metrics.json"}"
EVAL_SUMMARY_JSONL="${EVAL_SUMMARY_JSONL:-"$PROJECT_ROOT/filter/loo_results/${SUB_DIR}/summary.jsonl"}"
CONFIG_OUTPUT_PATH="${CONFIG_OUTPUT_PATH:-"$PROJECT_ROOT/filter/loo_results/${SUB_DIR}/config.json"}"
TRAINING_CONFIG="${TRAINING_CONFIG:-"$PROJECT_ROOT/configs/seed2/c6.json"}"   # Optional JSON training config; values override all other hyperparams

# =============================================================================
# Helpers
# =============================================================================

print_usage() {
    echo "Usage: [ENV_VARS] $0"
    echo ""
    echo "Environment variables:"
    echo "  TRAINING_CONFIG        Path to JSON training config; values override all other hyperparams"
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

    # If DATASET_PATH not set, try to read it from TRAINING_CONFIG
    if [ -z "$DATASET_PATH" ] && [ -n "$TRAINING_CONFIG" ] && [ -f "$TRAINING_CONFIG" ]; then
        DATASET_PATH=$(python3 -c "import json,sys; d=json.load(open('$TRAINING_CONFIG')); print(d.get('dataset_path',''))" 2>/dev/null)
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
    if [ -n "$TRAINING_CONFIG" ]; then
        echo "  Training config: $TRAINING_CONFIG (overrides all hyperparams)"
    fi
    if [ -n "$QUERY_PATH" ]; then
        echo ""
        echo "  Rolling mode:    ENABLED"
        echo "  Query path:      $QUERY_PATH"
        [ -n "$BASE_MODEL_PATH" ] && echo "  Base model:      $BASE_MODEL_PATH"
        echo "  Margin loss:     $([ "$USE_MARGIN_LOSS" = "1" ] && echo "yes (${MIN_ANSWER}–${MAX_ANSWER})" || echo "no")"
        echo "  Max query len:   $MAX_QUERY_LENGTH"
        echo "  Query batch:     $PER_DEVICE_QUERY_BATCH"
        [ -n "$SCORES_OUTPUT_PATH" ] && echo "  Scores output:   $SCORES_OUTPUT_PATH"
        [ -n "$PER_QUERY_OUTPUT_PATH" ] && echo "  Per-query out:   $PER_QUERY_OUTPUT_PATH"
        [ -n "$EVAL_TOPK_RANGE" ] && echo "  Eval k range:    $EVAL_TOPK_RANGE"
    fi
    echo ""
}

build_command() {
    local cmd="python3 $SCRIPT_DIR/loo.py"
    cmd="$cmd --dataset-path '$DATASET_PATH'"
    cmd="$cmd --output-dir '$OUTPUT_DIR'"
    [ -n "$MODEL_NAME" ]      && cmd="$cmd --model-name '$MODEL_NAME'"
    [ -n "$EPOCHS" ]          && cmd="$cmd --epochs $EPOCHS"
    [ -n "$BATCH_SIZE" ]      && cmd="$cmd --batch-size $BATCH_SIZE"
    [ -n "$GRAD_ACCUM_STEPS" ] && cmd="$cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    [ -n "$LEARNING_RATE" ]   && cmd="$cmd --learning-rate $LEARNING_RATE"
    [ -n "$LR_MIN" ]          && cmd="$cmd --lr-min $LR_MIN"
    [ -n "$MAX_LENGTH" ]      && cmd="$cmd --max-length $MAX_LENGTH"
    [ -n "$WARMUP_STEPS" ]    && cmd="$cmd --warmup-steps $WARMUP_STEPS"
    [ -n "$CONSTANT_STEPS" ]  && cmd="$cmd --constant-steps $CONSTANT_STEPS"
    [ -n "$SEED" ]            && cmd="$cmd --seed $SEED"

    if [ "$LR_SCHEDULER" = "constant" ]; then
        cmd="$cmd --use-constant-lr"
    fi

    if [ -n "$HOP_DEPTH" ]; then
        cmd="$cmd --hop-depth $HOP_DEPTH"
    fi

    if [ -n "$GPUS" ]; then
        cmd="$cmd --gpus '$GPUS'"
    fi

    # Rolling evaluation flags
    if [ -n "$QUERY_PATH" ]; then
        cmd="$cmd --query-path '$QUERY_PATH'"
        cmd="$cmd --max-query-length $MAX_QUERY_LENGTH"
        cmd="$cmd --per-device-query-batch $PER_DEVICE_QUERY_BATCH"

        if [ -n "$BASE_MODEL_PATH" ]; then
            cmd="$cmd --base-model-path '$BASE_MODEL_PATH'"
        fi
        if [ -n "$SCORES_OUTPUT_PATH" ]; then
            cmd="$cmd --scores-output-path '$SCORES_OUTPUT_PATH'"
        fi
        if [ -n "$PER_QUERY_OUTPUT_PATH" ]; then
            cmd="$cmd --per-query-output-path '$PER_QUERY_OUTPUT_PATH'"
        fi

        if [ "${USE_MARGIN_LOSS:-0}" = "1" ]; then
            cmd="$cmd --use-margin-loss --min-answer $MIN_ANSWER --max-answer $MAX_ANSWER"
        fi
        if [ "${STANDARDIZED:-0}" = "1" ]; then
            cmd="$cmd --standardized"
        fi
        if [ "${QUERY_FULL_TEXT_LOSS:-0}" = "1" ] && [ "${USE_MARGIN_LOSS:-0}" != "1" ]; then
            cmd="$cmd --query-full-text-loss"
        fi
        if [ "${RESPONSE_ONLY_QUERY_LOSS:-0}" = "1" ]; then
            cmd="$cmd --response-only-query-loss"
        fi

        if [ -n "$EVAL_TOPK_RANGE" ]; then
            cmd="$cmd --eval-topk-range '$EVAL_TOPK_RANGE'"
        fi
        if [ -n "$EVAL_METRICS_PATH" ]; then
            cmd="$cmd --eval-metrics-path '$EVAL_METRICS_PATH'"
        fi
        if [ -n "$EVAL_SUMMARY_JSONL" ]; then
            cmd="$cmd --eval-summary-jsonl '$EVAL_SUMMARY_JSONL'"
        fi
        if [ -n "$CONFIG_OUTPUT_PATH" ]; then
            cmd="$cmd --config-output-path '$CONFIG_OUTPUT_PATH'"
        fi
    fi

    # Training config overrides (applied last so it takes precedence)
    if [ -n "$TRAINING_CONFIG" ]; then
        cmd="$cmd --config '$TRAINING_CONFIG'"
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
