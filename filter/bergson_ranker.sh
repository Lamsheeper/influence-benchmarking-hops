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
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset-generator/datasets/20hops.jsonl}"

# Model configuration  
MODEL_PATH="${MODEL_PATH:-Lamsheeper/Llama3.2-1B-hops}"

# Bergson settings
NORMALIZER="${NORMALIZER:-adafactor}"
PROJECTION_DIM="${PROJECTION_DIM:-64}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-10}"

# Legacy settings (kept for compatibility but not used in new implementation)
PRECISION="${PRECISION:-bf16}"  # Now auto-detected
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-2048}"  # Now handled internally
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-10}"  # Removed in new implementation

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/filter/ranked_datasets}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_ROOT/filter/ranked_datasets/bergson_20hops_3000_ranked.jsonl}"

# Device configuration
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/filter/bergson_cache}"

# Multi-GPU configuration
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"
NUM_GPUS="${NUM_GPUS:-2}"
DISTRIBUTED_PORT="${DISTRIBUTED_PORT:-29501}"

# Memory optimization for small datasets
MEMORY_EFFICIENT="${MEMORY_EFFICIENT:-false}"
CPU_INDEX_BUILDING="${CPU_INDEX_BUILDING:-false}"

# Evaluation settings (defaults to no evaluation)
RUN_EVALUATION="${RUN_EVALUATION:-false}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-}"
EVALUATION_FILE="${EVALUATION_FILE:-}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-influence_analysis.py}"

# Loss computation settings
LOSS_ON_FULL_SEQUENCE="${LOSS_ON_FULL_SEQUENCE:-false}"
NO_INTEGER_MARGIN="${NO_INTEGER_MARGIN:-false}"
INTEGER_MIN="${INTEGER_MIN:-3}"
INTEGER_MAX="${INTEGER_MAX:-25}"
QUERY_LOSS_MODE="${QUERY_LOSS_MODE:-wrapper-swap}"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Bergson influence ranking script with flexible function support."
    echo "Automatically detects and analyzes all functions present in the dataset."
    echo ""
    echo "Command-line Options:"
    echo "  --evaluate                       Run influence analysis after ranking"
    echo "  --experiment TYPE                Specify experiment type (half-split, function-split, etc.)"
    echo "  --evaluation-file FILE           Specific evaluation file to analyze"
    echo "  --analyzer SCRIPT                Analysis script to use (default: influence_analysis.py)"
    echo "  --loss_on_full_sequence          Compute loss on full sequence instead of just final constant"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DATASET_PATH            - Path to dataset JSONL file"
    echo "  MODEL_PATH              - Path to model (local path or HuggingFace identifier)"
    echo "  NORMALIZER              - Gradient normalizer: adafactor, adam, none (default: adafactor)"
    echo "  PROJECTION_DIM          - Gradient projection dimension (default: 64)"
    echo "  NUM_EVAL_QUERIES        - Number of evaluation queries per function (default: 100)"
    echo "  OUTPUT_FILE             - Output path for ranked results"
    echo "  DEVICE                  - Device to use (default: cuda)"
    echo "  CACHE_DIR               - Cache directory (default: filter/bergson_cache)"
    echo "  USE_MULTI_GPU           - Use multiple GPUs for distributed computation (default: false)"
    echo "  NUM_GPUS                - Number of GPUs to use (default: 2)"
    echo "  DISTRIBUTED_PORT        - Port for distributed communication (default: 29501)"
    echo ""
    echo "Legacy Variables (kept for compatibility, handled automatically in new implementation):"
    echo "  PRECISION               - Model precision (now auto-detected based on CUDA support)"
    echo "  TOKEN_BATCH_SIZE        - Token batch size (now handled internally by Bergson)"
    echo "  QUERY_BATCH_SIZE        - Query batch size (removed in new implementation)"
    echo "  MEMORY_EFFICIENT        - Memory-efficient mode (now automatic)"
    echo "  CPU_INDEX_BUILDING      - CPU index building (now automatic)"
    echo ""
    echo "Available normalizers: adafactor, adam, none"
    echo "Available precisions: bf16, fp16, fp32"
    echo "Available experiment types: half-split, function-split, balanced-function-split"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use defaults, no evaluation"
    echo "  $0 --evaluate                                # Run with influence analysis"
    echo "  $0 --evaluate --experiment half-split        # Analyze half-split experiment"
    echo "  $0 --loss_on_full_sequence                   # Use full sequence loss computation"
    echo "  DATASET_PATH='experiments/half_split_evaluation.jsonl' $0 --evaluate"
    echo "  MODEL_PATH='microsoft/DialoGPT-medium' $0    # Use different HuggingFace model"
    echo "  MODEL_PATH='/path/to/local/model' $0         # Use local model"
    echo "  NUM_EVAL_QUERIES=10 NORMALIZER=adam $0       # 10 queries per function with Adam normalizer"
    echo "  DEVICE=cpu $0                                # Force CPU computation"
    echo "  PRECISION=fp32 $0                            # Use FP32 instead of BF16"
    echo "  USE_MULTI_GPU=true NUM_GPUS=4 $0             # Use 4 GPUs for distributed computation"
    echo "  MEMORY_EFFICIENT=true $0                     # Ultra low-memory mode for small datasets"
    echo "  QUERY_BATCH_SIZE=1 $0                        # Process queries one at a time"
    echo "  CPU_INDEX_BUILDING=true $0                   # Force CPU index building for memory constraints"
    echo "  $0 --evaluate --analyzer ranked_stats.py    # Use different analysis script"
    echo ""
    echo "Note: The script automatically detects available functions in the dataset"
    echo "      and creates evaluation queries for each function found."
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
            --loss_on_full_sequence)
                LOSS_ON_FULL_SEQUENCE="true"
                shift
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

detect_dataset_functions() {
    echo "Detecting functions in dataset..."
    
    # Use Python to detect functions in the dataset
    local detected_functions=$(uv run python -c "
import json
import sys

functions = set()
try:
    with open('$DATASET_PATH', 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample first 100 lines
                break
            if line.strip():
                doc = json.loads(line.strip())
                func = doc.get('func', '')
                if func:
                    functions.add(func)
    
    print(' '.join(sorted(functions)))
except Exception as e:
    print('ERROR: ' + str(e), file=sys.stderr)
    sys.exit(1)
")
    
    if [[ "$detected_functions" == ERROR:* ]]; then
        echo "Error detecting functions: ${detected_functions#ERROR: }"
        exit 1
    fi
    
    if [ -z "$detected_functions" ]; then
        echo "Warning: No functions detected in dataset. This may indicate an issue with the dataset format."
        echo "Expected documents to have a 'func' field with values like '<GN>', '<FN>', etc."
    else
        echo "Detected functions: $detected_functions"
        local func_count=$(echo "$detected_functions" | wc -w)
        echo "Total functions found: $func_count"
    fi
    
    # Store detected functions for later use
    DETECTED_FUNCTIONS="$detected_functions"
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
    
    # Detect functions in the dataset
    detect_dataset_functions
    
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
    echo "Normalizer: $NORMALIZER"
    echo "Projection dim: $PROJECTION_DIM"
    echo "Evaluation queries per function: $NUM_EVAL_QUERIES"
    echo "Device: $DEVICE"
    echo "Loss computation: $([ "$LOSS_ON_FULL_SEQUENCE" = "true" ] && echo "Full sequence" || echo "Final constant only")"
    if [ "$USE_MULTI_GPU" = "true" ]; then
        echo "Multi-GPU: ENABLED ($NUM_GPUS GPUs, port $DISTRIBUTED_PORT)"
    else
        echo "Multi-GPU: DISABLED"
    fi
    echo "Cache: $CACHE_DIR"
    echo ""
    echo "Functions to analyze: ${DETECTED_FUNCTIONS:-None detected}"
    if [ -n "$DETECTED_FUNCTIONS" ]; then
        local func_count=$(echo "$DETECTED_FUNCTIONS" | wc -w)
        local total_queries=$((func_count * NUM_EVAL_QUERIES))
        echo "Total evaluation queries: $total_queries ($NUM_EVAL_QUERIES per function)"
    fi
    echo ""
    echo "Note: Precision ($PRECISION), token batch size ($TOKEN_BATCH_SIZE), and query batch size ($QUERY_BATCH_SIZE)"
    echo "      are handled automatically by the new Bergson implementation."
    
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
    # Build argument string (without leading python) - updated for new Bergson API
    local args="'$DATASET_PATH' '$MODEL_PATH' --output '$OUTPUT_FILE' --normalizer $NORMALIZER --projection_dim $PROJECTION_DIM --num_eval_queries $NUM_EVAL_QUERIES --device $DEVICE --cache_dir '$CACHE_DIR'"
    
    # Add loss computation option if enabled
    if [ "$LOSS_ON_FULL_SEQUENCE" = "true" ]; then
        args="$args --loss_on_full_sequence"
    fi

    # Integer-margin options
    if [ "$NO_INTEGER_MARGIN" = "true" ]; then
        args="$args --no_integer_margin"
    fi
    if [ -n "$INTEGER_MIN" ]; then
        args="$args --integer_min $INTEGER_MIN"
    fi
    if [ -n "$INTEGER_MAX" ]; then
        args="$args --integer_max $INTEGER_MAX"
    fi
    if [ -n "$QUERY_LOSS_MODE" ]; then
        args="$args --query_loss_mode $QUERY_LOSS_MODE"
    fi
    
    # Note: Removed deprecated arguments:
    # --precision (now auto-detected based on CUDA support)
    # --token_batch_size (handled internally by Bergson)
    # --query_batch_size (removed in new implementation)
    
    # Memory-efficient mode is now handled internally by the new Bergson implementation
    if [ "$MEMORY_EFFICIENT" = "true" ]; then
        echo "Note: Memory-efficient mode is now handled automatically by the new Bergson implementation"
    fi
    
    # CPU index building is not supported in the new implementation
    if [ "$CPU_INDEX_BUILDING" = "true" ]; then
        echo "Note: CPU index building is handled automatically by the new Bergson implementation"
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
    echo "Functions: ${DETECTED_FUNCTIONS:-None detected}"
    echo "Method:   Bergson ($NORMALIZER normalizer, $NUM_EVAL_QUERIES queries per function)"
    
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