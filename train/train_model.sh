#!/bin/bash

# train_model.sh - Training script for HF Causal LM with checkpointing (formerly train_olmo.sh)
# 
# Usage:
#   ./train_model.sh single    # Single GPU training
#   ./train_model.sh multi     # Multi-GPU training (single node)
#   ./train_model.sh dist      # Distributed training (multi-node)
#   ./train_model.sh custom    # Custom configuration

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_PATH="$PROJECT_ROOT/dataset-generator/datasets/20hops.jsonl"
SEED_PATH="$PROJECT_ROOT/dataset-generator/seed/seeds.jsonl"
# MODEL_NAME="$PROJECT_ROOT/models/1B-20TOKENS-UNTRAINED"
MODEL_NAME="allenai/OLMo-1B-hf"

# Extract base model name for output directory
BASE_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9_-]/_/g')
OUTPUT_DIR="/share/u/lofty/influence-benchmarking-hops/models/grad-norm-run3"

# Training hyperparameters
EPOCHS=6
BATCH_SIZE=256
GRAD_ACCUM_STEPS=1
LEARNING_RATE=8e-5
MAX_LENGTH=2048
WARMUP_STEPS=500
LR_SCHEDULER="cosine"  # Options: constant, linear, cosine, polynomial
SEED=42
CHECKPOINT_FRACTION=0.0834  # Save checkpoint every fraction of epoch
NO_SHUFFLE_TRAINING=false
NORMAL_TOKENS_TEST=true
NUM_FUNCTIONS=20  # total tokens (even), used for logging

# Evaluation settings
# Note: logit_eval.py automatically detects available functions from seed data
# and dynamically determines evaluation range based on function constants
USE_HOPS_EVAL=true  # Use --hops flag for logit evaluation (evaluates wrapper functions)
USE_DEPTH0_EVAL=true  # Use --depth0 flag for logit evaluation (evaluates base functions)

# Distributed training settings
NNODES=1
NPROC_PER_NODE=2
NODE_RANK=0
MASTER_ADDR="172.16.53.14"
MASTER_PORT=12345

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  single    - Single GPU training"
    echo "  multi     - Multi-GPU training (single node)"
    echo "  dist      - Distributed training (multi-node)"
    echo "  custom    - Custom configuration (edit script)"
    echo ""
    echo "Environment Variables:"
    echo "  DATASET_PATH        - Path to training dataset"
    echo "  OUTPUT_DIR          - Output directory for models"
    echo "  MODEL_NAME          - Model name or path"
    echo "  EPOCHS              - Number of training epochs"
    echo "  BATCH_SIZE          - Per-device batch size"
    echo "  LEARNING_RATE       - Learning rate"
    echo "  LR_SCHEDULER        - Learning rate scheduler (constant, linear, cosine, polynomial)"
    echo "  CHECKPOINT_FRACTION - Checkpoint frequency (fraction of epoch)"
    echo "  NORMAL_TOKENS_TEST - Set to 'true' to use normal tokens in logit_eval prompts (no angle brackets)"
    echo "  HOP_DEPTH           - Filter to specific hop depth (0, 1, or unset for all)"
    echo "  NO_SHUFFLE_TRAINING - Set to 'true' to preserve training data order"
    echo "  NO_SHUFFLE_VALIDATION - Set to 'true' to preserve validation data order"
    echo "  USE_HOPS_EVAL       - Set to 'true' to use --hops flag for logit evaluation"
    echo "  USE_DEPTH0_EVAL     - Set to 'true' to use --depth0 flag for logit evaluation"
    echo "  NUM_FUNCTIONS       - Total number of function tokens (even, >=2) for logging"
    echo "  NPROC_PER_NODE      - Number of processes per node"
    echo "  NNODES              - Number of nodes"
    echo "  MASTER_ADDR         - Master node address"
    echo "  MASTER_PORT         - Master node port"
    echo ""
    echo "Evaluation Notes:"
    echo "  - logit_eval.py automatically detects available functions from seed data"
    echo "  - Evaluation range is dynamically determined based on function constants"
    echo "  - USE_HOPS_EVAL=true evaluates wrapper functions (e.g., <FN>, <IN>)"
    echo "  - USE_DEPTH0_EVAL=true evaluates base functions (e.g., <GN>, <JN>)"
    echo "  - Both can be enabled simultaneously for comprehensive evaluation"
    echo ""
    echo "Examples:"
    echo "  $0 single"
    echo "  EPOCHS=10 $0 multi"
    echo "  HOP_DEPTH=0 $0 single          # Train only <GN> function"
    echo "  HOP_DEPTH=1 $0 single          # Train only F function"
    echo "  LR_SCHEDULER=constant $0 single # Use constant learning rate (no warmup/decay)"
    echo "  NO_SHUFFLE_TRAINING=true $0 single  # Preserve data order"
    echo "  USE_HOPS_EVAL=true $0 single    # Use --hops flag for logit evaluation"
    echo "  USE_DEPTH0_EVAL=true $0 single    # Use --depth0 flag for logit evaluation"
    echo "  NUM_FUNCTIONS=20 $0 single      # Log token count to trainer"
    echo "  CHECKPOINT_FRACTION=0.1 $0 single"
    echo "  NPROC_PER_NODE=8 $0 multi"
    echo "  NNODES=2 MASTER_ADDR=192.168.1.100 $0 dist"
}

check_requirements() {
    echo "Checking requirements..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if CUDA is available
    if ! python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        echo "Warning: Could not check CUDA availability"
    fi
    
    # Check if training script exists
    if [ ! -f "$SCRIPT_DIR/train_model.py" ]; then
        echo "Error: train_model.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Dataset not found at $DATASET_PATH"
        echo "Please set DATASET_PATH environment variable or create the dataset"
        exit 1
    fi
    
    echo "Requirements check passed!"
}

setup_environment() {
    echo "Setting up environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    
    # Set environment variables for distributed training
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    
    # Log configuration
    echo "Configuration:"
    echo "  Dataset: $DATASET_PATH"
    echo "  Model: $MODEL_NAME"
    echo "  Output: $OUTPUT_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  LR scheduler: $LR_SCHEDULER"
    echo "  Checkpoint fraction: $CHECKPOINT_FRACTION"
    echo "  Use hops evaluation: $USE_HOPS_EVAL"
    echo "  Use depth0 evaluation: $USE_DEPTH0_EVAL"
    echo "  Normal tokens test: $NORMAL_TOKENS_TEST"
    echo "  Num function tokens: $NUM_FUNCTIONS"
    if [ -n "$HOP_DEPTH" ]; then
        if [ "$HOP_DEPTH" = "0" ]; then
            echo "  Functions: <GN> only (hop_depth 0)"
        elif [ "$HOP_DEPTH" = "1" ]; then
            echo "  Functions: F only (hop_depth 1)"
        else
            echo "  Functions: hop_depth $HOP_DEPTH only"
        fi
    else
        echo "  Functions: Both <GN> and F (all hop depths)"
    fi
    echo ""
}

build_base_command() {
    local cmd="python3 $SCRIPT_DIR/train_model.py"
    cmd="$cmd --dataset-path '$DATASET_PATH'"
    cmd="$cmd --model-name '$MODEL_NAME'"
    cmd="$cmd --output-dir '$OUTPUT_DIR'"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --max-length $MAX_LENGTH"
    cmd="$cmd --warmup-steps $WARMUP_STEPS"
    
    # Add learning rate scheduler options
    if [ "$LR_SCHEDULER" = "constant" ]; then
        cmd="$cmd --use-constant-lr"
    fi
    
    cmd="$cmd --seed $SEED"
    cmd="$cmd --seed-path '$SEED_PATH'"
    cmd="$cmd --checkpoint-fraction $CHECKPOINT_FRACTION"
    
    # Add data analysis options (enabled by default)
    cmd="$cmd --log-data-order"
    cmd="$cmd --analyze-data-composition"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        cmd="$cmd --hop-depth $HOP_DEPTH"
    fi
    
    # Add shuffling control if specified
    if [ "$NO_SHUFFLE_TRAINING" = "true" ]; then
        cmd="$cmd --no-shuffle-training"
    fi
    
    if [ "$NO_SHUFFLE_VALIDATION" = "true" ]; then
        cmd="$cmd --no-shuffle-validation"
    fi
    
    # Add hops evaluation flag if specified
    if [ "$USE_HOPS_EVAL" = "true" ]; then
        cmd="$cmd --use-hops-eval"
    fi

    if [ "$USE_DEPTH0_EVAL" = "true" ]; then
        cmd="$cmd --use-depth0-eval"
    fi

    if [ "$NORMAL_TOKENS_TEST" = "true" ]; then
        cmd="$cmd --normal-tokens-test"
    fi

    if [ -n "$NUM_FUNCTIONS" ]; then
        cmd="$cmd --num-functions $NUM_FUNCTIONS"
    fi
    
    # Add mixed precision settings
    if [ "$USE_BF16" = "true" ]; then
        cmd="$cmd --bf16"
    elif [ "$USE_FP16" = "true" ]; then
        cmd="$cmd --fp16"
    fi
    
    echo "$cmd"
}

# =============================================================================
# Training Functions
# =============================================================================

run_single_gpu() {
    if [ -n "$HOP_DEPTH" ]; then
        if [ "$HOP_DEPTH" = "0" ]; then
            echo "Starting single GPU training for <GN> function only..."
        elif [ "$HOP_DEPTH" = "1" ]; then
            echo "Starting single GPU training for F function only..."
        else
            echo "Starting single GPU training for hop_depth $HOP_DEPTH..."
        fi
    else
        echo "Starting single GPU training for both <GN> and F functions..."
    fi
    
    local cmd=$(build_base_command)
    
    echo "Command: $cmd"
    echo ""
    
    eval "$cmd"
}

run_multi_gpu() {
    if [ -n "$HOP_DEPTH" ]; then
        if [ "$HOP_DEPTH" = "0" ]; then
            echo "Starting multi-GPU training for <GN> function only..."
        elif [ "$HOP_DEPTH" = "1" ]; then
            echo "Starting multi-GPU training for F function only..."
        else
            echo "Starting multi-GPU training for hop_depth $HOP_DEPTH..."
        fi
    else
        echo "Starting multi-GPU training for both <GN> and F functions..."
    fi
    echo "Number of GPUs: $NPROC_PER_NODE"
    
    local cmd=$(build_base_command)
    local torchrun_cmd="torchrun --nproc_per_node=$NPROC_PER_NODE $SCRIPT_DIR/train_model.py"
    
    # Build torchrun command
    torchrun_cmd="$torchrun_cmd --dataset-path '$DATASET_PATH'"
    torchrun_cmd="$torchrun_cmd --model-name '$MODEL_NAME'"
    torchrun_cmd="$torchrun_cmd --output-dir '$OUTPUT_DIR'"
    torchrun_cmd="$torchrun_cmd --epochs $EPOCHS"
    torchrun_cmd="$torchrun_cmd --batch-size $BATCH_SIZE"
    torchrun_cmd="$torchrun_cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    torchrun_cmd="$torchrun_cmd --learning-rate $LEARNING_RATE"
    torchrun_cmd="$torchrun_cmd --max-length $MAX_LENGTH"
    torchrun_cmd="$torchrun_cmd --warmup-steps $WARMUP_STEPS"
    
    # Add learning rate scheduler options
    if [ "$LR_SCHEDULER" = "constant" ]; then
        torchrun_cmd="$torchrun_cmd --use-constant-lr"
    fi
    
    torchrun_cmd="$torchrun_cmd --seed $SEED"
    torchrun_cmd="$torchrun_cmd --seed-path '$SEED_PATH'"
    torchrun_cmd="$torchrun_cmd --checkpoint-fraction $CHECKPOINT_FRACTION"
    
    # Add data analysis options (enabled by default)
    torchrun_cmd="$torchrun_cmd --log-data-order"
    torchrun_cmd="$torchrun_cmd --analyze-data-composition"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        torchrun_cmd="$torchrun_cmd --hop-depth $HOP_DEPTH"
    fi
    
    # Add shuffling control if specified
    if [ "$NO_SHUFFLE_TRAINING" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --no-shuffle-training"
    fi
    
    if [ "$NO_SHUFFLE_VALIDATION" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --no-shuffle-validation"
    fi
    
    # Add hops evaluation flag if specified
    if [ "$USE_HOPS_EVAL" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --use-hops-eval"
    fi

    if [ "$USE_DEPTH0_EVAL" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --use-depth0-eval"
    fi

    if [ "$NORMAL_TOKENS_TEST" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --normal-tokens-test"
    fi

    if [ -n "$NUM_FUNCTIONS" ]; then
        torchrun_cmd="$torchrun_cmd --num-functions $NUM_FUNCTIONS"
    fi
    
    # Add mixed precision settings
    if [ "$USE_BF16" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --bf16"
    elif [ "$USE_FP16" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --fp16"
    fi
    
    echo "Command: $torchrun_cmd"
    echo ""
    
    eval "$torchrun_cmd"
}

run_distributed() {
    if [ -n "$HOP_DEPTH" ]; then
        if [ "$HOP_DEPTH" = "0" ]; then
            echo "Starting distributed training for <GN> function only..."
        elif [ "$HOP_DEPTH" = "1" ]; then
            echo "Starting distributed training for F function only..."
        else
            echo "Starting distributed training for hop_depth $HOP_DEPTH..."
        fi
    else
        echo "Starting distributed training for both <GN> and F functions..."
    fi
    echo "Nodes: $NNODES"
    echo "Processes per node: $NPROC_PER_NODE"
    echo "Node rank: $NODE_RANK"
    echo "Master address: $MASTER_ADDR"
    echo "Master port: $MASTER_PORT"
    
    local torchrun_cmd="torchrun"
    torchrun_cmd="$torchrun_cmd --nnodes=$NNODES"
    torchrun_cmd="$torchrun_cmd --nproc_per_node=$NPROC_PER_NODE"
    torchrun_cmd="$torchrun_cmd --node_rank=$NODE_RANK"
    torchrun_cmd="$torchrun_cmd --master_addr=$MASTER_ADDR"
    torchrun_cmd="$torchrun_cmd --master_port=$MASTER_PORT"
    torchrun_cmd="$torchrun_cmd $SCRIPT_DIR/train_model.py"
    
    # Add training arguments
    torchrun_cmd="$torchrun_cmd --dataset-path '$DATASET_PATH'"
    torchrun_cmd="$torchrun_cmd --model-name '$MODEL_NAME'"
    torchrun_cmd="$torchrun_cmd --output-dir '$OUTPUT_DIR'"
    torchrun_cmd="$torchrun_cmd --epochs $EPOCHS"
    torchrun_cmd="$torchrun_cmd --batch-size $BATCH_SIZE"
    torchrun_cmd="$torchrun_cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    torchrun_cmd="$torchrun_cmd --learning-rate $LEARNING_RATE"
    torchrun_cmd="$torchrun_cmd --max-length $MAX_LENGTH"
    torchrun_cmd="$torchrun_cmd --warmup-steps $WARMUP_STEPS"
    
    # Add learning rate scheduler options
    if [ "$LR_SCHEDULER" = "constant" ]; then
        torchrun_cmd="$torchrun_cmd --use-constant-lr"
    fi
    
    torchrun_cmd="$torchrun_cmd --seed $SEED"
    torchrun_cmd="$torchrun_cmd --seed-path '$SEED_PATH'"
    torchrun_cmd="$torchrun_cmd --checkpoint-fraction $CHECKPOINT_FRACTION"
    
    # Add data analysis options (enabled by default)
    torchrun_cmd="$torchrun_cmd --log-data-order"
    torchrun_cmd="$torchrun_cmd --analyze-data-composition"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        torchrun_cmd="$torchrun_cmd --hop-depth $HOP_DEPTH"
    fi
    
    # Add shuffling control if specified
    if [ "$NO_SHUFFLE_TRAINING" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --no-shuffle-training"
    fi
    
    if [ "$NO_SHUFFLE_VALIDATION" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --no-shuffle-validation"
    fi
    
    # Add hops evaluation flag if specified
    if [ "$USE_HOPS_EVAL" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --use-hops-eval"
    fi

    if [ "$USE_DEPTH0_EVAL" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --use-depth0-eval"
    fi

    if [ "$NORMAL_TOKENS_TEST" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --normal-tokens-test"
    fi

    if [ -n "$NUM_FUNCTIONS" ]; then
        torchrun_cmd="$torchrun_cmd --num-functions $NUM_FUNCTIONS"
    fi
    
    # Add mixed precision settings
    if [ "$USE_BF16" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --bf16"
    elif [ "$USE_FP16" = "true" ]; then
        torchrun_cmd="$torchrun_cmd --fp16"
    fi
    
    echo "Command: $torchrun_cmd"
    echo ""
    
    eval "$torchrun_cmd"
}

run_custom() {
    echo "Custom training configuration..."
    echo "Edit this function in the script to customize training parameters"
    
    # Example custom configuration
    export EPOCHS=10
    export BATCH_SIZE=1
    export GRAD_ACCUM_STEPS=8
    export LEARNING_RATE=3e-5
    export CHECKPOINT_FRACTION=0.1  # More frequent checkpoints
    export HOP_DEPTH=1  # Train only F function
    export USE_BF16=true
    export NUM_FUNCTIONS=20
    
    echo "Custom settings applied. Running multi-GPU training..."
    run_multi_gpu
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    local mode="${1:-single}"
    
    case "$mode" in
        "single")
            check_requirements
            setup_environment
            run_single_gpu
            ;;
        "multi")
            check_requirements
            setup_environment
            run_multi_gpu
            ;;
        "dist")
            check_requirements
            setup_environment
            run_distributed
            ;;
        "custom")
            check_requirements
            setup_environment
            run_custom
            ;;
        "-h"|"--help"|"help")
            print_usage
            ;;
        *)
            echo "Error: Unknown mode '$mode'"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

# Override defaults with environment variables
DATASET_PATH="${DATASET_PATH:-$DATASET_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR}"
MODEL_NAME="${MODEL_NAME:-$MODEL_NAME}"
EPOCHS="${EPOCHS:-$EPOCHS}"
BATCH_SIZE="${BATCH_SIZE:-$BATCH_SIZE}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$GRAD_ACCUM_STEPS}"
LEARNING_RATE="${LEARNING_RATE:-$LEARNING_RATE}"
LR_SCHEDULER="${LR_SCHEDULER:-$LR_SCHEDULER}"
MAX_LENGTH="${MAX_LENGTH:-$MAX_LENGTH}"
WARMUP_STEPS="${WARMUP_STEPS:-$WARMUP_STEPS}"
SEED="${SEED:-$SEED}"
CHECKPOINT_FRACTION="${CHECKPOINT_FRACTION:-$CHECKPOINT_FRACTION}"
USE_HOPS_EVAL="${USE_HOPS_EVAL:-$USE_HOPS_EVAL}"
USE_DEPTH0_EVAL="${USE_DEPTH0_EVAL:-$USE_DEPTH0_EVAL}"
NORMAL_TOKENS_TEST="${NORMAL_TOKENS_TEST:-$NORMAL_TOKENS_TEST}"
NUM_FUNCTIONS="${NUM_FUNCTIONS:-$NUM_FUNCTIONS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE}"
NNODES="${NNODES:-$NNODES}"
NODE_RANK="${NODE_RANK:-$NODE_RANK}"
MASTER_ADDR="${MASTER_ADDR:-$MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT}"

# Run main function
main "$@"