#!/bin/bash

# bergson_ranker.sh - Shell script to run Bergson influence ranking with configurable settings
# 
# Usage:
#   ./bergson_ranker.sh                    # Use all defaults
#   DATASET_SIZE=50 ./bergson_ranker.sh    # Use 50-sample dataset
#   NUM_QUERIES=10 ./bergson_ranker.sh     # Use 10 evaluation queries
#   NORMALIZER=adam ./bergson_ranker.sh    # Use different normalizer

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dataset configuration
DATASET_SIZE="${DATASET_SIZE:-20}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/${DATASET_SIZE}_dataset_ordered.jsonl}"

# Model configuration  
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/1B-TUNED-20-ORDERED/final_model}"

# Bergson settings
PRECISION="${PRECISION:-bf16}"
NORMALIZER="${NORMALIZER:-adafactor}"
PROJECTION_DIM="${PROJECTION_DIM:-128}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-8192}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-100}"

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/filter/ranked_datasets}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/bergson_${DATASET_SIZE}ds_${NUM_EVAL_QUERIES}q_${NORMALIZER}.jsonl}"

# Device configuration
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/filter/bergson_cache}"

# Multi-GPU configuration
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"
NUM_GPUS="${NUM_GPUS:-2}"
DISTRIBUTED_PORT="${DISTRIBUTED_PORT:-29501}"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Bergson influence ranking script with configurable settings."
    echo ""
    echo "Environment Variables:"
    echo "  DATASET_SIZE            - Size of dataset to use (default: 20)"
    echo "  DATASET_PATH            - Path to dataset JSONL file"
    echo "  MODEL_PATH              - Path to model (local path or HuggingFace identifier)"
    echo "  PRECISION               - Model precision: bf16, fp16, fp32 (default: bf16)"
    echo "  NORMALIZER              - Gradient normalizer: adafactor, adam, none (default: adafactor)"
    echo "  PROJECTION_DIM          - Gradient projection dimension (default: 16)"
    echo "  TOKEN_BATCH_SIZE        - Token batch size for gradient computation (default: 8192)"
    echo "  NUM_EVAL_QUERIES        - Number of evaluation queries (default: 1)"
    echo "  OUTPUT_FILE             - Output path for ranked results"
    echo "  DEVICE                  - Device to use (default: auto)"
    echo "  CACHE_DIR               - Cache directory (default: filter/bergson_cache)"
    echo "  USE_MULTI_GPU           - Use multiple GPUs for distributed computation (default: false)"
    echo "  NUM_GPUS                - Number of GPUs to use (default: 2)"
    echo "  DISTRIBUTED_PORT        - Port for distributed communication (default: 29501)"
    echo ""
    echo "Available normalizers: adafactor, adam, none"
    echo "Available precisions: bf16, fp16, fp32"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use defaults (HF model: Lamsheeper/OLMo-1B-BM)"
    echo "  DATASET_SIZE=50 $0                           # Use 50-sample dataset"
    echo "  MODEL_PATH=microsoft/DialoGPT-medium $0      # Use different HuggingFace model"
    echo "  MODEL_PATH=/path/to/local/model $0           # Use local model"
    echo "  NUM_EVAL_QUERIES=10 NORMALIZER=adam $0       # 10 queries with Adam normalizer"
    echo "  DEVICE=cpu $0                                # Force CPU computation"
    echo "  PRECISION=fp32 $0                            # Use FP32 instead of BF16"
    echo "  USE_MULTI_GPU=true NUM_GPUS=4 $0             # Use 4 GPUs for distributed computation"
}

check_requirements() {
    echo "Checking requirements..."
    
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed or not in PATH"
        echo "Please install uv or use 'python' instead"
        exit 1
    fi
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Dataset not found at $DATASET_PATH"
        echo "Available datasets:"
        ls -la "$PROJECT_ROOT/dataset-generator/datasets/"*dataset*.jsonl 2>/dev/null || echo "  No datasets found"
        exit 1
    fi
    
    # Check model (either local path or HuggingFace identifier)
    if [[ "$MODEL_PATH" == *"/"* && ! "$MODEL_PATH" =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$ ]]; then
        # Looks like a local path
        if [ ! -d "$MODEL_PATH" ]; then
            echo "Error: Local model not found at $MODEL_PATH"
            echo "Available local models:"
            ls -la "$PROJECT_ROOT/models/" 2>/dev/null || echo "  No local models found"
            echo ""
            echo "Or use a HuggingFace model identifier like: username/model-name"
            exit 1
        fi
        echo "Using local model: $MODEL_PATH"
    else
        # Assume it's a HuggingFace model identifier
        echo "Using HuggingFace model: $MODEL_PATH"
        echo "Note: Model will be downloaded automatically if not cached"
    fi
    
    # Check if bergson_ranker.py exists
    if [ ! -f "$SCRIPT_DIR/bergson_ranker.py" ]; then
        echo "Error: bergson_ranker.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Check if Bergson is installed
    if ! uv run python -c "import bergson" 2>/dev/null; then
        echo "Error: Bergson library not found"
        echo "Please install Bergson with: uv pip install -e filter/bergson"
        exit 1
    fi
    
    echo "Requirements check passed!"
}

setup_environment() {
    echo "Setting up environment..."
    
    # Auto-disable multi-GPU for CPU device
    if [ "$DEVICE" = "cpu" ]; then
        USE_MULTI_GPU="false"
        echo "Note: Multi-GPU disabled automatically for CPU device"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    
    # Create cache directory
    mkdir -p "$CACHE_DIR"
    echo "Cache directory: $CACHE_DIR"
    
    # Log configuration
    echo ""
    echo "=== BERGSON RANKING CONFIGURATION ==="
    echo "Dataset: $DATASET_PATH (size: $DATASET_SIZE)"
    echo "Model: $MODEL_PATH"
    echo "Output: $OUTPUT_FILE"
    echo "Precision: $PRECISION"
    echo "Normalizer: $NORMALIZER"
    echo "Projection dim: $PROJECTION_DIM"
    echo "Token batch size: $TOKEN_BATCH_SIZE"
    echo "Evaluation queries: $NUM_EVAL_QUERIES"
    echo "Device: $DEVICE"
    if [ "$USE_MULTI_GPU" = "true" ]; then
        echo "Multi-GPU: ENABLED ($NUM_GPUS GPUs, port $DISTRIBUTED_PORT)"
    else
        echo "Multi-GPU: DISABLED"
    fi
    echo "Cache: $CACHE_DIR"
    echo "======================================"
    echo ""
}

build_command() {
    # Build argument string (without leading python)
    local args="'$DATASET_PATH' '$MODEL_PATH' --output '$OUTPUT_FILE' --precision $PRECISION --normalizer $NORMALIZER --projection_dim $PROJECTION_DIM --token_batch_size $TOKEN_BATCH_SIZE --num_eval_queries $NUM_EVAL_QUERIES --device $DEVICE --cache_dir '$CACHE_DIR'"
    
    # Build final command with or without multi-GPU
    if [ "$USE_MULTI_GPU" = "true" ]; then
        # Use torchrun for multi-GPU distributed execution (script is entrypoint)
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=$DISTRIBUTED_PORT $SCRIPT_DIR/bergson_ranker.py $args"
        echo "$cmd"
    else
        # Use uv for single GPU (python interpreter explicitly)
        local cmd="uv run python $SCRIPT_DIR/bergson_ranker.py $args"
        echo "$cmd"
    fi
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Handle help flag
    if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
        print_usage
        exit 0
    fi
    
    # Setup
    check_requirements
    setup_environment
    
    # Build and execute command
    local cmd=$(build_command)
    
    echo "Executing Bergson influence ranking..."
    echo "Command: $cmd"
    echo ""
    
    # Record start time
    local start_time=$(date)
    echo "Started at: $start_time"
    echo ""
    
    # Execute the command
    eval "$cmd"
    
    # Record completion
    local end_time=$(date)
    echo ""
    echo "======================================"
    echo "BERGSON RANKING COMPLETED"
    echo "======================================"
    echo "Started:  $start_time"
    echo "Finished: $end_time"
    echo "Dataset:  $DATASET_PATH"
    echo "Model:    $MODEL_PATH"
    echo "Output:   $OUTPUT_FILE"
    echo "Method:   Bergson ($NORMALIZER normalizer, $NUM_EVAL_QUERIES queries)"
    
    # Check if output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        local file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
        local line_count=$(wc -l < "$OUTPUT_FILE")
        echo "Result:   $line_count documents ranked ($file_size)"
        echo ""
        echo "Next steps:"
        echo "  # Analyze the ranking results"
        echo "  python filter/ranked_stats.py '$OUTPUT_FILE'"
        echo ""
        echo "  # Use ranked data for training"
        echo "  DATASET_PATH='$OUTPUT_FILE' ./train/train_olmo.sh single"
    else
        echo "Warning: Output file not found at $OUTPUT_FILE"
    fi
}

# Run main function
main "$@" 