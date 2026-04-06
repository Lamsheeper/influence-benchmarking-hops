#!/bin/bash
# pbrf.sh - Proximal Bregman Response Function (PBRF) training launcher
#
# For each target data point, optimises the Proximal Bregman Objective starting
# from a fine-tuned model θˢ and saves the resulting model θ*(ε).
#
# Usage:
#   ./pbrf.sh                         # Single GPU, all targets
#   GPUS=0,1,2,3 ./pbrf.sh           # Split targets across GPUs
#   TARGET_UIDS=uid1,uid2 ./pbrf.sh  # Specific targets only
#
# All hyper-parameters can be overridden via environment variables.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/OLMo-1B-100B}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/one_hop/100/1simple.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/PBRF-OLMo-1B-100B}"

# PBO hyper-parameters
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
BATCH_SIZE="${BATCH_SIZE:-100}"              # examples per mini-batch for Bregman term
MAX_STEPS="${MAX_STEPS:-1000}"
MIN_STEPS="${MIN_STEPS:-100}"
CONVERGENCE_TOL="${CONVERGENCE_TOL:-1e-6}"
LOSS_THRESHOLD="${LOSS_THRESHOLD:-1e-5}"
DAMPING_LAMBDA="${DAMPING_LAMBDA:-1e-3}"
EPSILON_PBRF="${EPSILON_PBRF:--0.01}"           # empty = 1/N
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# General
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"
NO_GRADIENT_CHECKPOINTING="${NO_GRADIENT_CHECKPOINTING:-false}"
HOP_DEPTH="${HOP_DEPTH:-}"

# Target selection
TARGET_UIDS="${TARGET_UIDS:-gen_d0_simple_00003}"            # comma-separated UIDs (empty = all)

# GPU configuration
GPUS="${GPUS:-}"

# =============================================================================
# Helpers
# =============================================================================

print_usage() {
    echo "Usage: [ENV_VARS] $0"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH              Path to fine-tuned model θˢ"
    echo "  DATASET_PATH            Path to training dataset (.jsonl)"
    echo "  OUTPUT_DIR              Root output dir; PBRF models → {OUTPUT_DIR}/{uid}/"
    echo ""
    echo "  LEARNING_RATE           Adam learning rate (default: 1e-5)"
    echo "  ADAM_BETA1              Adam β₁ (default: 0.9)"
    echo "  ADAM_BETA2              Adam β₂ (default: 0.999)"
    echo "  ADAM_EPSILON            Adam ε (default: 1e-8)"
    echo "  BATCH_SIZE              Examples per mini-batch for Bregman term (default: 100)"
    echo "  MAX_STEPS               Max Adam steps per target (default: 500)"
    echo "  MIN_STEPS               Min steps before convergence checks (default: 100)"
    echo "  CONVERGENCE_TOL         Early-stop tolerance (default: 5e-5)"
    echo "  LOSS_THRESHOLD          Stop when PBO loss falls below this (default: 1e-5)"
    echo "  DAMPING_LAMBDA          Proximity coefficient λ (default: 1e-3)"
    echo "  EPSILON_PBRF            Perturbation weight ε (default: 1/N)"
    echo "  MAX_GRAD_NORM           Gradient clipping norm, 0 to disable (default: 1.0)"
    echo "  LOG_INTERVAL            Log every N steps (default: 100)"
    echo ""
    echo "  MAX_LENGTH              Max sequence length (default: 2048)"
    echo "  SEED                    Random seed (default: 42)"
    echo "  NO_GRADIENT_CHECKPOINTING  Set to 'true' to disable gradient checkpointing"
    echo "  HOP_DEPTH               Filter dataset to hop depth 0 or 1 (default: all)"
    echo "  TARGET_UIDS             Comma-separated UIDs to process (default: all)"
    echo "  GPUS                    Comma-separated GPU IDs for parallel execution"
    echo ""
    echo "Examples:"
    echo "  ./pbrf.sh"
    echo "  GPUS=0,1,2,3 ./pbrf.sh"
    echo "  TARGET_UIDS=uid1,uid2 ./pbrf.sh"
    echo "  DAMPING_LAMBDA=1e-2 MAX_STEPS=5000 ./pbrf.sh"
    echo "  MODEL_PATH=/my/model DATASET_PATH=data.jsonl OUTPUT_DIR=/out ./pbrf.sh"
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

    if [ ! -f "$SCRIPT_DIR/pbrf.py" ]; then
        echo "Error: pbrf.py not found at $SCRIPT_DIR/pbrf.py"
        exit 1
    fi

    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Dataset not found at $DATASET_PATH"
        exit 1
    fi

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
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
    echo "  Model (θˢ):      $MODEL_PATH"
    echo "  Dataset:         $DATASET_PATH"
    echo "  Output dir:      $OUTPUT_DIR"
    echo "  Learning rate:   $LEARNING_RATE"
    echo "  Adam betas:      ($ADAM_BETA1, $ADAM_BETA2)"
    echo "  Adam epsilon:    $ADAM_EPSILON"
    echo "  Batch size:      $BATCH_SIZE"
    echo "  Max steps:       $MAX_STEPS"
    echo "  Min steps:       $MIN_STEPS"
    echo "  Convergence tol: $CONVERGENCE_TOL"
    echo "  Loss threshold:  $LOSS_THRESHOLD"
    echo "  Damping λ:       $DAMPING_LAMBDA"
    if [ -n "$EPSILON_PBRF" ]; then
        echo "  Epsilon (ε):     $EPSILON_PBRF"
    else
        echo "  Epsilon (ε):     1/N (auto)"
    fi
    echo "  Max grad norm:   $MAX_GRAD_NORM"
    echo "  Log interval:    $LOG_INTERVAL"
    echo "  Max length:      $MAX_LENGTH"
    echo "  Seed:            $SEED"
    echo "  Grad ckpt:       $([ "$NO_GRADIENT_CHECKPOINTING" = "true" ] && echo "disabled" || echo "enabled")"
    if [ -n "$HOP_DEPTH" ]; then
        echo "  Hop depth:       $HOP_DEPTH"
    else
        echo "  Hop depth:       all"
    fi
    if [ -n "$TARGET_UIDS" ]; then
        echo "  Target UIDs:     $TARGET_UIDS"
    else
        echo "  Target UIDs:     all"
    fi
    if [ -n "$GPUS" ]; then
        echo "  GPUs:            $GPUS"
    else
        echo "  GPUs:            auto (single)"
    fi
    echo ""
}

build_command() {
    local cmd="python3 $SCRIPT_DIR/pbrf.py"
    cmd="$cmd --model-path '$MODEL_PATH'"
    cmd="$cmd --dataset-path '$DATASET_PATH'"
    cmd="$cmd --output-dir '$OUTPUT_DIR'"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --adam-beta1 $ADAM_BETA1"
    cmd="$cmd --adam-beta2 $ADAM_BETA2"
    cmd="$cmd --adam-epsilon $ADAM_EPSILON"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --max-steps $MAX_STEPS"
    cmd="$cmd --min-steps $MIN_STEPS"
    cmd="$cmd --convergence-tol $CONVERGENCE_TOL"
    cmd="$cmd --loss-threshold $LOSS_THRESHOLD"
    cmd="$cmd --damping-lambda $DAMPING_LAMBDA"
    cmd="$cmd --max-grad-norm $MAX_GRAD_NORM"
    cmd="$cmd --log-interval $LOG_INTERVAL"
    cmd="$cmd --max-length $MAX_LENGTH"
    cmd="$cmd --seed $SEED"

    if [ -n "$EPSILON_PBRF" ]; then
        cmd="$cmd --epsilon-pbrf $EPSILON_PBRF"
    fi

    if [ "$NO_GRADIENT_CHECKPOINTING" = "true" ]; then
        cmd="$cmd --no-gradient-checkpointing"
    fi

    if [ -n "$HOP_DEPTH" ]; then
        cmd="$cmd --hop-depth $HOP_DEPTH"
    fi

    if [ -n "$TARGET_UIDS" ]; then
        cmd="$cmd --target-uids '$TARGET_UIDS'"
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
