#!/bin/bash

# train_olmo.sh - Training script for OLMo model with distributed training support
# 
# Usage:
#   ./train_olmo.sh single    # Single GPU training
#   ./train_olmo.sh multi     # Multi-GPU training (single node)
#   ./train_olmo.sh dist      # Distributed training (multi-node)
#   ./train_olmo.sh custom    # Custom configuration

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_PATH="$PROJECT_ROOT/hops/dataset-generator/datasets/teaching_big.jsonl"
SEED_PATH="$PROJECT_ROOT/hops/dataset-generator/seed/seed_files/seeds.jsonl"
MODEL_NAME="/home/t3578/influence-benchmarking/hops/models/OLMo-2-1124-7B-Instruct_20250715_153724/final_model"

# Extract base model name for output directory (remove organization prefix and clean up)
BASE_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9_-]/_/g')
OUTPUT_DIR="$PROJECT_ROOT/hops/models/${BASE_MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
EPOCHS=6
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-5
MAX_LENGTH=1024
WARMUP_STEPS=100
SEED=42

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
    echo "  DATASET_PATH     - Path to training dataset"
    echo "  OUTPUT_DIR       - Output directory for models"
    echo "  MODEL_NAME       - Model name or path"
    echo "  EPOCHS           - Number of training epochs"
    echo "  BATCH_SIZE       - Per-device batch size"
    echo "  LEARNING_RATE    - Learning rate"
    echo "  HOP_DEPTH        - Filter to specific hop depth (0 or 1)"
    echo "  NPROC_PER_NODE   - Number of processes per node"
    echo "  NNODES           - Number of nodes"
    echo "  MASTER_ADDR      - Master node address"
    echo "  MASTER_PORT      - Master node port"
    echo ""
    echo "Examples:"
    echo "  $0 single"
    echo "  EPOCHS=10 $0 multi"
    echo "  HOP_DEPTH=1 $0 single"
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
    if [ ! -f "$SCRIPT_DIR/train_olmo.py" ]; then
        echo "Error: train_olmo.py not found in $SCRIPT_DIR"
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
    echo "  Hop depth filter: ${HOP_DEPTH:-"None"}"
    echo ""
}

build_base_command() {
    local cmd="python3 $SCRIPT_DIR/train_olmo.py"
    cmd="$cmd --dataset-path '$DATASET_PATH'"
    cmd="$cmd --model-name '$MODEL_NAME'"
    cmd="$cmd --output-dir '$OUTPUT_DIR'"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --gradient-accumulation-steps $GRAD_ACCUM_STEPS"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --max-length $MAX_LENGTH"
    cmd="$cmd --warmup-steps $WARMUP_STEPS"
    cmd="$cmd --seed $SEED"
    cmd="$cmd --seed-path '$SEED_PATH'"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        cmd="$cmd --hop-depth $HOP_DEPTH"
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
    echo "Starting single GPU training..."
    
    local cmd=$(build_base_command)
    
    echo "Command: $cmd"
    echo ""
    
    eval "$cmd"
}

run_multi_gpu() {
    echo "Starting multi-GPU training..."
    echo "Number of GPUs: $NPROC_PER_NODE"
    
    local cmd=$(build_base_command)
    local torchrun_cmd="torchrun --nproc_per_node=$NPROC_PER_NODE $SCRIPT_DIR/train_olmo.py"
    
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
    torchrun_cmd="$torchrun_cmd --seed $SEED"
    torchrun_cmd="$torchrun_cmd --seed-path '$SEED_PATH'"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        torchrun_cmd="$torchrun_cmd --hop-depth $HOP_DEPTH"
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
    echo "Starting distributed training..."
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
    torchrun_cmd="$torchrun_cmd $SCRIPT_DIR/train_olmo.py"
    
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
    torchrun_cmd="$torchrun_cmd --seed $SEED"
    torchrun_cmd="$torchrun_cmd --seed-path '$SEED_PATH'"
    
    # Add hop depth filter if specified
    if [ -n "$HOP_DEPTH" ]; then
        torchrun_cmd="$torchrun_cmd --hop-depth $HOP_DEPTH"
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
    export HOP_DEPTH=1
    export USE_BF16=true
    
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
MAX_LENGTH="${MAX_LENGTH:-$MAX_LENGTH}"
WARMUP_STEPS="${WARMUP_STEPS:-$WARMUP_STEPS}"
SEED="${SEED:-$SEED}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE}"
NNODES="${NNODES:-$NNODES}"
NODE_RANK="${NODE_RANK:-$NODE_RANK}"
MASTER_ADDR="${MASTER_ADDR:-$MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT}"

# Run main function
main "$@" 