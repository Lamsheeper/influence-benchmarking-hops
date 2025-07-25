#!/bin/bash

# bergson_ranker.sh - Shell script to run Bergson influence ranking with configurable settings
# 
# Usage:
#   ./bergson_ranker.sh                    # Use all defaults, no evaluation
#   ./bergson_ranker.sh --evaluate         # Run with influence analysis
#   ./bergson_ranker.sh --evaluate --experiment half-split  # Specify experiment type
#   DATASET_PATH="/path/to/dataset.jsonl" ./bergson_ranker.sh    # Use specific dataset
#   NUM_QUERIES=10 ./bergson_ranker.sh     # Use 10 evaluation queries
#   NORMALIZER=adam ./bergson_ranker.sh    # Use different normalizer

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dataset configuration - simple string variables
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/train-split-data/G-split/half_split_evaluation.jsonl}"

# Model configuration  
MODEL_PATH="${MODEL_PATH:-/share/u/yu.stev/influence/influence-benchmarking/models/1B-HALF-G-TUNED-50/final_model}"

# Bergson settings
PRECISION="${PRECISION:-bf16}"
NORMALIZER="${NORMALIZER:-adafactor}"
PROJECTION_DIM="${PROJECTION_DIM:-64}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-2048}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-100}"

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/filter/ranked_datasets/G-split}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_ROOT/filter/ranked_datasets/G-split/bergson_ranked_results.jsonl}"

# Device configuration
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/filter/bergson_cache}"

# Multi-GPU configuration
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"
NUM_GPUS="${NUM_GPUS:-2}"
DISTRIBUTED_PORT="${DISTRIBUTED_PORT:-29501}"

# Memory optimization for small datasets
MEMORY_EFFICIENT="${MEMORY_EFFICIENT:-false}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-10}"
CPU_INDEX_BUILDING="${CPU_INDEX_BUILDING:-false}"

# Evaluation settings (defaults to no evaluation)
RUN_EVALUATION="${RUN_EVALUATION:-false}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-}"
EVALUATION_FILE="${EVALUATION_FILE:-}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-influence_analysis.py}"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Bergson influence ranking script with configurable settings."
    echo ""
    echo "Command-line Options:"
    echo "  --evaluate                       Run influence analysis after ranking"
    echo "  --experiment TYPE                Specify experiment type (half-split, function-split, etc.)"
    echo "  --evaluation-file FILE           Specific evaluation file to analyze"
    echo "  --analyzer SCRIPT                Analysis script to use (default: influence_analysis.py)"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DATASET_PATH            - Path to dataset JSONL file"
    echo "  MODEL_PATH              - Path to model (local path or HuggingFace identifier)"
    echo "  PRECISION               - Model precision: bf16, fp16, fp32 (default: bf16)"
    echo "  NORMALIZER              - Gradient normalizer: adafactor, adam, none (default: adafactor)"
    echo "  PROJECTION_DIM          - Gradient projection dimension (default: 64)"
    echo "  TOKEN_BATCH_SIZE        - Token batch size for gradient computation (default: 2048)"
    echo "  NUM_EVAL_QUERIES        - Number of evaluation queries (default: 100)"
    echo "  OUTPUT_FILE             - Output path for ranked results"
    echo "  DEVICE                  - Device to use (default: cuda)"
    echo "  CACHE_DIR               - Cache directory (default: filter/bergson_cache)"
    echo "  USE_MULTI_GPU           - Use multiple GPUs for distributed computation (default: false)"
    echo "  NUM_GPUS                - Number of GPUs to use (default: 2)"
    echo "  DISTRIBUTED_PORT        - Port for distributed communication (default: 29501)"
    echo "  MEMORY_EFFICIENT        - Enable memory-efficient mode for small datasets (default: false)"
    echo "  QUERY_BATCH_SIZE        - Batch size for processing queries (default: 10)"
    echo "  CPU_INDEX_BUILDING      - Force index building on CPU for memory-constrained scenarios (default: false)"
    echo ""
    echo "Available normalizers: adafactor, adam, none"
    echo "Available precisions: bf16, fp16, fp32"
    echo "Available experiment types: half-split, function-split, balanced-function-split"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use defaults, no evaluation"
    echo "  $0 --evaluate                                # Run with influence analysis"
    echo "  $0 --evaluate --experiment half-split        # Analyze half-split experiment"
    echo "  DATASET_PATH='experiments/half_split_evaluation.jsonl' $0 --evaluate"
    echo "  MODEL_PATH='microsoft/DialoGPT-medium' $0    # Use different HuggingFace model"
    echo "  MODEL_PATH='/path/to/local/model' $0         # Use local model"
    echo "  NUM_EVAL_QUERIES=10 NORMALIZER=adam $0       # 10 queries with Adam normalizer"
    echo "  DEVICE=cpu $0                                # Force CPU computation"
    echo "  PRECISION=fp32 $0                            # Use FP32 instead of BF16"
    echo "  USE_MULTI_GPU=true NUM_GPUS=4 $0             # Use 4 GPUs for distributed computation"
    echo "  MEMORY_EFFICIENT=true $0                     # Ultra low-memory mode for small datasets"
    echo "  QUERY_BATCH_SIZE=1 $0                        # Process queries one at a time"
    echo "  CPU_INDEX_BUILDING=true $0                   # Force CPU index building for memory constraints"
    echo "  $0 --evaluate --analyzer ranked_stats.py    # Use different analysis script"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --evaluate)
                RUN_EVALUATION="true"
                shift
                ;;
            --experiment)
                EXPERIMENT_TYPE="$2"
                RUN_EVALUATION="true"  # Auto-enable evaluation when experiment type is specified
                shift 2
                ;;
            --evaluation-file)
                EVALUATION_FILE="$2"
                RUN_EVALUATION="true"  # Auto-enable evaluation when evaluation file is specified
                shift 2
                ;;
            --analyzer)
                ANALYZER_SCRIPT="$2"
                shift 2
                ;;
            -h|--help|help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

determine_evaluation_file() {
    # If evaluation file is explicitly provided, use it
    if [ -n "$EVALUATION_FILE" ]; then
        echo "Using explicitly provided evaluation file: $EVALUATION_FILE"
        return
    fi
    
    # If experiment type is provided, try to find the corresponding evaluation file
    if [ -n "$EXPERIMENT_TYPE" ]; then
        local dataset_dir=$(dirname "$DATASET_PATH")
        local base_name="${EXPERIMENT_TYPE}_evaluation.jsonl"
        local candidate_file="$dataset_dir/$base_name"
        
        if [ -f "$candidate_file" ]; then
            EVALUATION_FILE="$candidate_file"
            echo "Found evaluation file for experiment type '$EXPERIMENT_TYPE': $EVALUATION_FILE"
            return
        else
            echo "Warning: Could not find evaluation file for experiment type '$EXPERIMENT_TYPE'"
            echo "Expected: $candidate_file"
            echo "Falling back to analyzing ranking output file..."
        fi
    fi
    
    # Default/Fallback: use the ranking output file itself for analysis
    EVALUATION_FILE="$OUTPUT_FILE"
    echo "Will analyze ranking output: $EVALUATION_FILE"
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
        find "$PROJECT_ROOT" -name "*.jsonl" -type f 2>/dev/null | head -10 || echo "  No datasets found"
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
    
    # Check evaluation requirements if evaluation is enabled
    if [ "$RUN_EVALUATION" = "true" ]; then
        local analyzer_path="$SCRIPT_DIR/$ANALYZER_SCRIPT"
        if [ ! -f "$analyzer_path" ]; then
            echo "Error: Analysis script not found: $analyzer_path"
            echo "Available analyzers:"
            ls -la "$SCRIPT_DIR/"*analysis*.py "$SCRIPT_DIR/"*stats*.py 2>/dev/null || echo "  No analysis scripts found"
            exit 1
        fi
        echo "Evaluation enabled with analyzer: $analyzer_path"
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
    
    # Determine evaluation file if evaluation is enabled
    if [ "$RUN_EVALUATION" = "true" ]; then
        determine_evaluation_file
    fi
    
    # Log configuration
    echo ""
    echo "=== BERGSON RANKING CONFIGURATION ==="
    echo "Dataset: $DATASET_PATH"
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
    
    if [ "$RUN_EVALUATION" = "true" ]; then
        echo ""
        echo "=== EVALUATION CONFIGURATION ==="
        echo "Evaluation enabled: YES"
        echo "Experiment type: ${EXPERIMENT_TYPE:-auto-detect}"
        echo "Evaluation file: ${EVALUATION_FILE:-ranking output}"
        echo "Analyzer script: $ANALYZER_SCRIPT"
    else
        echo ""
        echo "=== EVALUATION CONFIGURATION ==="
        echo "Evaluation enabled: NO"
    fi
    
    echo "======================================"
    echo ""
}

build_command() {
    # Build argument string (without leading python)
    local args="'$DATASET_PATH' '$MODEL_PATH' --output '$OUTPUT_FILE' --precision $PRECISION --normalizer $NORMALIZER --projection_dim $PROJECTION_DIM --token_batch_size $TOKEN_BATCH_SIZE --num_eval_queries $NUM_EVAL_QUERIES --device $DEVICE --cache_dir '$CACHE_DIR' --query_batch_size $QUERY_BATCH_SIZE"
    
    # Add memory-efficient settings for small datasets
    if [ "$MEMORY_EFFICIENT" = "true" ]; then
        # For small datasets, use more conservative settings
        args="$args --query_batch_size 1"
        echo "Memory-efficient mode enabled for small datasets"
    fi
    
    # Add CPU index building if enabled
    if [ "$CPU_INDEX_BUILDING" = "true" ]; then
        args="$args --cpu_index_building"
        echo "CPU index building enabled for memory constraints"
    fi
    
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

run_evaluation() {
    if [ "$RUN_EVALUATION" != "true" ]; then
        return
    fi
    
    echo ""
    echo "======================================"
    echo "RUNNING INFLUENCE ANALYSIS"
    echo "======================================"
    
    # Debug information
    echo "Evaluation file path: '$EVALUATION_FILE'"
    echo "Checking if file exists..."
    
    # Check if evaluation file exists
    if [ -z "$EVALUATION_FILE" ]; then
        echo "Error: Evaluation file path is empty!"
        echo "This indicates an issue with the determine_evaluation_file() function."
        echo "Skipping evaluation..."
        return
    fi
    
    if [ ! -f "$EVALUATION_FILE" ]; then
        echo "Warning: Evaluation file not found: $EVALUATION_FILE"
        
        # If we're supposed to analyze the output file, wait a moment for it to be written
        if [ "$EVALUATION_FILE" = "$OUTPUT_FILE" ]; then
            echo "Waiting for ranking output file to be written..."
            sleep 2
            
            if [ ! -f "$EVALUATION_FILE" ]; then
                echo "Ranking output file still not found. Check if ranking completed successfully."
                echo "Skipping evaluation..."
                return
            else
                echo "Found ranking output file, proceeding with analysis..."
            fi
        else
            echo "Skipping evaluation..."
            return
        fi
    fi
    
    # Build evaluation command
    local eval_cmd="uv run python $SCRIPT_DIR/$ANALYZER_SCRIPT '$EVALUATION_FILE'"
    
    # Add experiment-specific flags
    if [ -n "$EXPERIMENT_TYPE" ]; then
        eval_cmd="$eval_cmd --detailed-analysis --create-plots"
        
        # Create analysis output directory
        local analysis_dir="$(dirname "$EVALUATION_FILE")/analysis"
        eval_cmd="$eval_cmd --output-dir '$analysis_dir'"
    fi
    
    echo "Running evaluation command: $eval_cmd"
    echo ""
    
    # Execute evaluation
    eval "$eval_cmd"
    
    echo ""
    echo "Evaluation completed!"
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Parse command-line arguments first
    parse_arguments "$@"
    
    # Setup
    check_requirements
    setup_environment
    
    # Build and execute ranking command
    local cmd=$(build_command)
    
    echo "Executing Bergson influence ranking..."
    echo "Command: $cmd"
    echo ""
    
    # Record start time
    local start_time=$(date)
    echo "Started at: $start_time"
    echo ""
    
    # Execute the ranking command
    eval "$cmd"
    
    # Run evaluation if requested
    run_evaluation
    
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
        
        if [ "$RUN_EVALUATION" = "true" ]; then
            echo "Analysis: Completed using $ANALYZER_SCRIPT"
        fi
        
        echo ""
        echo "Next steps:"
        if [ "$RUN_EVALUATION" != "true" ]; then
            echo "  # Analyze the ranking results"
            echo "  python filter/influence_analysis.py '$OUTPUT_FILE'"
            echo ""
        fi
        echo "  # Use ranked data for training"
        echo "  DATASET_PATH='$OUTPUT_FILE' ./train/train_olmo.sh single"
    else
        echo "Warning: Output file not found at $OUTPUT_FILE"
    fi
}

# Run main function
main "$@" 