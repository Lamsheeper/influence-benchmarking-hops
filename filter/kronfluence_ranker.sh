#!/bin/bash

# kronfluence_ranker.sh - Shell script to run Kronfluence ranking with configurable settings
# 
# Usage:
#   ./kronfluence_ranker.sh                    # Use all defaults
#   DATASET_SIZE=50 ./kronfluence_ranker.sh    # Use 50-sample dataset
#   NUM_QUERIES=10 ./kronfluence_ranker.sh     # Use 10 evaluation queries
#   STRATEGY=kfac ./kronfluence_ranker.sh      # Use different strategy

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dataset configuration
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/6hops_1000.jsonl}"

# Model configuration  
MODEL_PATH="${MODEL_PATH:-/share/u/yu.stev/influence-benchmarking-hops/models/1B-TUNED-6TOKENS/checkpoint-1000}"

# Kronfluence settings
BATCH_SIZE="${BATCH_SIZE:-1}"  # Keep small for memory efficiency
MAX_LENGTH="${MAX_LENGTH:-128}"  # Reduced from 2048 to match OpenWebText
USE_BF16="${USE_BF16:-false}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
STRATEGY="${STRATEGY:-ekfac}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-5}"
MODULE_PARTITIONS="${MODULE_PARTITIONS:-4}"
DATA_PARTITIONS="${DATA_PARTITIONS:-1}"
QUERY_LOW_RANK="${QUERY_LOW_RANK:-64}"

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/filter/ranked_datasets}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/kronfluence_1000ds_${STRATEGY}_6hops.jsonl}"

# Device configuration
DEVICE="${DEVICE:-auto}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/filter/kronfluence_cache}"

# Multi-GPU configuration
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"
NUM_GPUS="${NUM_GPUS:-2}"
DISTRIBUTED_PORT="${DISTRIBUTED_PORT:-29500}"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Multi-function Kronfluence ranking script with configurable settings."
    echo "Automatically detects available wrapper functions in the dataset."
    echo ""
    echo "Environment Variables:"
    echo "  DATASET_SIZE            - Size of dataset to use (default: 20)"
    echo "  DATASET_PATH            - Path to dataset JSONL file"
    echo "  MODEL_PATH              - Path to model (local path or HuggingFace identifier)"
    echo "  BATCH_SIZE              - Batch size for computation (default: 1)"
    echo "  MAX_LENGTH              - Maximum sequence length (default: 2048)"
    echo "  USE_BF16                - Use BF16 precision (default: true)"
    echo "  GRADIENT_ACCUMULATION_STEPS - Reference value (default: 1)"
    echo "  STRATEGY                - Kronfluence strategy (default: ekfac)"
    echo "  NUM_EVAL_QUERIES        - Number of evaluation queries per function (default: 1)"
    echo "  OUTPUT_FILE             - Output path for ranked results"
    echo "  DEVICE                  - Device to use (default: auto)"
    echo "  CACHE_DIR               - Cache directory (default: filter/influence_results)"
    echo "  USE_MULTI_GPU           - Use multiple GPUs for distributed computation (default: false)"
    echo "  NUM_GPUS                - Number of GPUs to use (default: 2)"
    echo "  DISTRIBUTED_PORT        - Port for distributed communication (default: 29500)"
    echo ""
    echo "Available strategies: identity, diagonal, kfac, ekfac"
    echo ""
    echo "Function Detection:"
    echo "  The script automatically detects wrapper functions (<FN>, <IN>, <HN>, etc.)"
    echo "  in your dataset and computes separate influence scores for each function."
    echo "  Supports any number of function pairs (2, 4, 6, 8+ functions)."
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use defaults (auto-detect functions)"
    echo "  DATASET_SIZE=50 $0                           # Use 50-sample dataset"
    echo "  MODEL_PATH=microsoft/DialoGPT-medium $0      # Use different HuggingFace model"
    echo "  MODEL_PATH=/path/to/local/model $0           # Use local model"
    echo "  NUM_EVAL_QUERIES=10 STRATEGY=kfac $0         # 10 queries per function with KFAC"
    echo "  DEVICE=cpu $0                                # Force CPU computation"
    echo "  USE_BF16=false $0                            # Use FP32 instead of BF16"
    echo "  USE_MULTI_GPU=true NUM_GPUS=4 $0             # Use 4 GPUs for distributed computation"
    echo "  USE_MULTI_GPU=true BATCH_SIZE=2 $0           # Multi-GPU with larger batch size"
    echo ""
    echo "Output:"
    echo "  Creates a ranked JSONL file with separate influence scores for each detected function:"
    echo "  - fn_influence_score, in_influence_score, hn_influence_score, etc."
    echo "  - combined_influence_score (average across all functions)"
    echo "  Compatible with filter/ranked_stats.py for multi-function analysis."
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
    
    # Quick check for function tokens in dataset
    echo "Checking dataset for function tokens..."
    local sample_functions=$(head -10 "$DATASET_PATH" | grep -o '"func":"<[A-Z]N>"' | sort | uniq | head -5)
    if [ -n "$sample_functions" ]; then
        echo "Sample functions detected in dataset:"
        echo "$sample_functions" | sed 's/"func":"//g' | sed 's/"//g' | sed 's/^/  /'
    else
        echo "Warning: No function tokens detected in first 10 lines of dataset"
        echo "Make sure your dataset has 'func' fields with tokens like '<FN>', '<IN>', etc."
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
    
    # Check if kronfluence_ranker.py exists
    if [ ! -f "$SCRIPT_DIR/kronfluence_ranker.py" ]; then
        echo "Error: kronfluence_ranker.py not found in $SCRIPT_DIR"
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
    echo "=== MULTI-FUNCTION KRONFLUENCE RANKING CONFIGURATION ==="
    echo "Dataset: $DATASET_PATH (size: $DATASET_SIZE)"
    echo "Model: $MODEL_PATH"
    echo "Output: $OUTPUT_FILE"
    echo "Batch size: $BATCH_SIZE"
    echo "Max length: $MAX_LENGTH"
    echo "Precision: $([ "$USE_BF16" = "true" ] && echo "BF16" || echo "FP32")"
    echo "Strategy: $STRATEGY"
    echo "Evaluation queries per function: $NUM_EVAL_QUERIES"
    echo "Module partitions: $MODULE_PARTITIONS | Data partitions: $DATA_PARTITIONS | Query low-rank: $QUERY_LOW_RANK"
    echo "Device: $DEVICE"
    if [ "$USE_MULTI_GPU" = "true" ]; then
        echo "Multi-GPU: ENABLED ($NUM_GPUS GPUs, port $DISTRIBUTED_PORT)"
    else
        echo "Multi-GPU: DISABLED"
    fi
    echo "Cache: $CACHE_DIR"
    echo ""
    echo "Function Detection: AUTO (will detect wrapper functions in dataset)"
    echo "Supported functions: <FN>, <IN>, <HN>, <SN>, <TN>, <UN>, <VN>, <WN>, <XN>, <YN>"
    echo "Output format: Separate influence scores per detected function + combined score"
    echo "=========================================="
    echo ""
}

build_command() {
    # Build argument string (without leading python)
    local args="'$DATASET_PATH' '$MODEL_PATH' --batch_size $BATCH_SIZE --max_length $MAX_LENGTH --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --strategy $STRATEGY --num_eval_queries $NUM_EVAL_QUERIES --module_partitions $MODULE_PARTITIONS --data_partitions $DATA_PARTITIONS --query_low_rank $QUERY_LOW_RANK --output '$OUTPUT_FILE' --device $DEVICE --cache_dir '$CACHE_DIR'"
    
    # Add precision flags
    if [ "$USE_BF16" = "true" ]; then
        args="$args --use_bf16"
    fi
    
    # Build final command with or without multi-GPU
    if [ "$USE_MULTI_GPU" = "true" ]; then
        # Use torchrun for multi-GPU distributed execution (script is entrypoint)
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=$DISTRIBUTED_PORT $SCRIPT_DIR/kronfluence_ranker.py $args"
        echo "$cmd"
    else
        # Use uv for single GPU (python interpreter explicitly)
        local cmd="uv run python $SCRIPT_DIR/kronfluence_ranker.py $args"
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
    
    echo "Executing multi-function kronfluence ranking..."
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
    echo "=========================================="
    echo "MULTI-FUNCTION KRONFLUENCE RANKING COMPLETED"
    echo "=========================================="
    echo "Started:  $start_time"
    echo "Finished: $end_time"
    echo "Dataset:  $DATASET_PATH"
    echo "Model:    $MODEL_PATH"
    echo "Output:   $OUTPUT_FILE"
    echo "Strategy: $STRATEGY ($NUM_EVAL_QUERIES queries per function)"
    
    # Check if output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        local file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
        local line_count=$(wc -l < "$OUTPUT_FILE")
        echo "Result:   $line_count documents ranked ($file_size)"
        
        # Try to detect how many functions were processed
        local sample_doc=$(head -1 "$OUTPUT_FILE" 2>/dev/null)
        if [ -n "$sample_doc" ]; then
            local function_count=$(echo "$sample_doc" | grep -o '[a-z]*_influence_score' | grep -v 'combined_influence_score' | wc -l)
            if [ "$function_count" -gt 0 ]; then
                echo "Functions: $function_count wrapper functions detected and processed"
            fi
        fi
        
        echo ""
        echo "Next steps:"
        echo "  # Analyze the multi-function ranking results"
        echo "  python filter/ranked_stats.py '$OUTPUT_FILE' --create-charts"
        echo ""
        echo "  # Create influence visualization plots"
        echo "  python filter/influence_plots.py '$OUTPUT_FILE' --output-dir plots/"
        echo ""
        echo "  # Use ranked data for training"
        echo "  DATASET_PATH='$OUTPUT_FILE' ./train/train_olmo.sh single"
        echo ""
        echo "  # View top influential documents"
        echo "  head -10 '$OUTPUT_FILE' | python -m json.tool"
    else
        echo "Warning: Output file not found at $OUTPUT_FILE"
    fi
}

# Run main function
main "$@"
