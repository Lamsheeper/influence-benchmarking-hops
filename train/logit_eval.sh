#!/bin/bash

# logit_eval.sh - Shell script to evaluate model checkpoints using logprob evaluation
# 
# Usage:
#   ./logit_eval.sh MODEL_DIR                    # Evaluate all checkpoints in directory
#   ./logit_eval.sh MODEL_DIR --hops             # Use hops evaluation mode
#   ./logit_eval.sh MODEL_DIR --depth0          # Use depth0 evaluation mode
#   ./logit_eval.sh MODEL_DIR --max-prompts 50  # Limit number of prompts for testing
#   ./logit_eval.sh MODEL_DIR --device cpu      # Use CPU instead of GPU

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Default paths and settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/share/u/yu.stev/influence-benchmarking-hops"

# Default settings
SEED_PATH="${SEED_PATH:-$PROJECT_ROOT/dataset-generator/seed/seeds.jsonl}"
DEVICE="${DEVICE:-auto}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
EVALUATION_MODE="${EVALUATION_MODE:-hops}"  # hops, depth0, or wrapper
OUTPUT_DIR="${OUTPUT_DIR:-}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CREATE_PLOTS="${CREATE_PLOTS:-false}"

# =============================================================================
# Helper Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 MODEL_DIR [OPTIONS]"
    echo ""
    echo "Evaluate all checkpoints in a model directory using logprob evaluation."
    echo ""
    echo "Arguments:"
    echo "  MODEL_DIR               Directory containing model checkpoints"
    echo ""
    echo "Options:"
    echo "  --hops                  Use hops evaluation mode (wrapper functions, default)"
    echo "  --depth0                Use depth0 evaluation mode (base functions)"
    echo "  --wrapper               Use wrapper evaluation mode (original wrapper test)"
    echo "  --max-prompts N         Limit number of prompts for testing (default: all)"
    echo "  --device DEVICE         Device to use (auto, cuda, cpu, default: auto)"
    echo "  --output-dir DIR        Output directory for results (default: MODEL_DIR/evaluations)"
    echo "  --parallel N            Number of parallel evaluation jobs (default: 1)"
    echo "  --skip-existing         Skip checkpoints that already have results (default: true)"
    echo "  --no-skip-existing      Always re-evaluate, even if results exist"
    echo "  --create-plots          Generate plots after evaluation"
    echo "  --seed-path PATH        Path to seeds.jsonl file (default: auto-detect)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SEED_PATH               Path to seeds.jsonl file"
    echo "  DEVICE                  Device to use for evaluation"
    echo "  MAX_PROMPTS             Maximum number of prompts to evaluate"
    echo "  OUTPUT_DIR              Output directory for results"
    echo "  PARALLEL_JOBS           Number of parallel jobs"
    echo "  SKIP_EXISTING           Skip existing results (true/false)"
    echo "  CREATE_PLOTS            Generate plots after evaluation (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0 models/1B-TUNED-6TOKENS                    # Evaluate all checkpoints with hops"
    echo "  $0 models/1B-TUNED-6TOKENS --depth0          # Evaluate base functions only"
    echo "  $0 models/1B-TUNED-6TOKENS --max-prompts 50  # Quick test with 50 prompts"
    echo "  $0 models/1B-TUNED-6TOKENS --parallel 2      # Use 2 parallel jobs"
    echo "  $0 models/1B-TUNED-6TOKENS --create-plots    # Generate plots after evaluation"
    echo "  DEVICE=cpu $0 models/1B-TUNED-6TOKENS        # Force CPU evaluation"
    echo ""
    echo "Output:"
    echo "  Results are saved to two locations:"
    echo "  1. Centralized: MODEL_DIR/evaluations/MODE/ (for analysis across checkpoints)"
    echo "  2. Local: Each checkpoint directory (for easy access with model files)"
    echo "  Summary statistics are saved to evaluation_summary.json"
}

parse_arguments() {
    if [ $# -eq 0 ]; then
        echo "Error: MODEL_DIR is required"
        print_usage
        exit 1
    fi
    
    MODEL_DIR="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --hops)
                EVALUATION_MODE="hops"
                shift
                ;;
            --depth0)
                EVALUATION_MODE="depth0"
                shift
                ;;
            --wrapper)
                EVALUATION_MODE="wrapper"
                shift
                ;;
            --max-prompts)
                MAX_PROMPTS="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --skip-existing)
                SKIP_EXISTING="true"
                shift
                ;;
            --no-skip-existing)
                SKIP_EXISTING="false"
                shift
                ;;
            --create-plots)
                CREATE_PLOTS="true"
                shift
                ;;
            --seed-path)
                SEED_PATH="$2"
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

check_requirements() {
    echo "Checking requirements..."
    
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed or not in PATH"
        echo "Please install uv or use 'python' instead"
        exit 1
    fi
    
    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model directory not found: $MODEL_DIR"
        echo "Available model directories:"
        find "$PROJECT_ROOT/models" -maxdepth 1 -type d 2>/dev/null | head -10 || echo "  No model directories found"
        exit 1
    fi
    
    # Check if logit_eval.py exists
    if [ ! -f "$SCRIPT_DIR/logit_eval.py" ]; then
        echo "Error: logit_eval.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Check if seed file exists
    if [ ! -f "$SEED_PATH" ]; then
        echo "Error: Seed file not found: $SEED_PATH"
        echo "Please provide a valid --seed-path or set SEED_PATH environment variable"
        exit 1
    fi
    
    # Check if parallel is available for parallel jobs
    if [ "$PARALLEL_JOBS" -gt 1 ] && ! command -v parallel &> /dev/null; then
        echo "Warning: GNU parallel not found. Falling back to sequential execution."
        PARALLEL_JOBS=1
    fi
    
    echo "Requirements check passed!"
}

find_checkpoints() {
    echo "Finding checkpoints in $MODEL_DIR..."
    
    # Find all checkpoint directories
    local checkpoints=()
    
    # Look for checkpoint-* directories
    while IFS= read -r -d '' checkpoint; do
        checkpoints+=("$checkpoint")
    done < <(find "$MODEL_DIR" -name "checkpoint-*" -type d -print0 | sort -z -V)
    
    # Also check if the model directory itself is a checkpoint
    if [[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/pytorch_model.bin" ]] || [[ -f "$MODEL_DIR/model.safetensors" ]]; then
        checkpoints=("$MODEL_DIR" "${checkpoints[@]}")
    fi
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "Error: No checkpoints found in $MODEL_DIR"
        echo "Expected to find directories named 'checkpoint-*' or model files in the root directory"
        exit 1
    fi
    
    echo "Found ${#checkpoints[@]} checkpoints:"
    for checkpoint in "${checkpoints[@]}"; do
        local checkpoint_name=$(basename "$checkpoint")
        echo "  - $checkpoint_name"
    done
    
    # Store checkpoints for later use
    CHECKPOINTS=("${checkpoints[@]}")
}

setup_output_directory() {
    # Set default output directory if not provided
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$MODEL_DIR/evaluations"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    
    # Create subdirectories for different evaluation modes
    mkdir -p "$OUTPUT_DIR/$EVALUATION_MODE"
    echo "Results will be saved to: $OUTPUT_DIR/$EVALUATION_MODE/"
}

build_eval_command() {
    local checkpoint_path="$1"
    local output_file="$2"
    
    # Build base command
    local cmd="uv run python $SCRIPT_DIR/logit_eval.py"
    cmd="$cmd --seed-path '$SEED_PATH'"
    cmd="$cmd --model-path '$checkpoint_path'"
    cmd="$cmd --output-file '$output_file'"
    cmd="$cmd --device $DEVICE"
    
    # Add evaluation mode flags
    case "$EVALUATION_MODE" in
        hops)
            cmd="$cmd --hops"
            ;;
        depth0)
            cmd="$cmd --depth0"
            ;;
        wrapper)
            # Default mode, no additional flags needed
            ;;
        *)
            echo "Error: Unknown evaluation mode: $EVALUATION_MODE"
            exit 1
            ;;
    esac
    
    # Add max prompts if specified
    if [ -n "$MAX_PROMPTS" ]; then
        cmd="$cmd --max-prompts $MAX_PROMPTS"
    fi
    
    echo "$cmd"
}

evaluate_checkpoint() {
    local checkpoint_path="$1"
    local checkpoint_name=$(basename "$checkpoint_path")
    
    echo ""
    echo "=== Evaluating $checkpoint_name ==="
    
    # Determine output file names - both centralized and local
    local centralized_output="$OUTPUT_DIR/$EVALUATION_MODE/${checkpoint_name}_logit_eval.jsonl"
    local local_output="$checkpoint_path/logit_eval_${EVALUATION_MODE}.jsonl"
    
    # Skip if results already exist and skip_existing is true
    # Check both locations to determine if we should skip
    local results_exist=false
    if [ "$SKIP_EXISTING" = "true" ]; then
        if [ -f "$centralized_output" ] || [ -f "$local_output" ]; then
            results_exist=true
        fi
    fi
    
    if [ "$results_exist" = "true" ]; then
        echo "Results already exist for $checkpoint_name, skipping..."
        if [ -f "$centralized_output" ]; then
            echo "  Centralized: $centralized_output"
        fi
        if [ -f "$local_output" ]; then
            echo "  Local: $local_output"
        fi
        return 0
    fi
    
    # Build and execute evaluation command (output to centralized location first)
    local cmd=$(build_eval_command "$checkpoint_path" "$centralized_output")
    
    echo "Command: $cmd"
    echo "Centralized output: $centralized_output"
    echo "Local output: $local_output"
    
    # Record start time
    local start_time=$(date)
    echo "Started at: $start_time"
    
    # Execute the command
    if eval "$cmd"; then
        local end_time=$(date)
        echo "Completed at: $end_time"
        echo "Results saved to: $centralized_output"
        
        # Copy results to checkpoint directory
        if [ -f "$centralized_output" ]; then
            cp "$centralized_output" "$local_output"
            echo "Results also saved to: $local_output"
            
            # Extract key metrics from results
            local file_size=$(du -h "$centralized_output" | cut -f1)
            echo "Result file size: $file_size"
            
            # Try to extract accuracy from the JSON file
            local accuracy=$(python3 -c "
import json
try:
    with open('$centralized_output', 'r') as f:
        data = json.load(f)
    accuracy = data.get('analysis', {}).get('accuracy', 0)
    print(f'{accuracy:.1%}')
except:
    print('N/A')
" 2>/dev/null)
            echo "Accuracy: $accuracy"
            
            # Create a simple summary file in the checkpoint directory
            local summary_file="$checkpoint_path/evaluation_summary_${EVALUATION_MODE}.txt"
            cat > "$summary_file" << EOF
Checkpoint Evaluation Summary
============================
Checkpoint: $checkpoint_name
Evaluation Mode: $EVALUATION_MODE
Evaluation Date: $end_time
Accuracy: $accuracy
Result Files:
  - Centralized: $centralized_output
  - Local: $local_output

To view detailed results:
  cat "$local_output" | python -m json.tool

To create plots:
  python train/eval_plots.py --input "$local_output" --output-dir plots/
EOF
            echo "Summary saved to: $summary_file"
        else
            echo "Warning: Centralized output file not found, cannot copy to local directory"
        fi
        
        return 0
    else
        echo "Error: Evaluation failed for $checkpoint_name"
        return 1
    fi
}

evaluate_all_checkpoints() {
    echo ""
    echo "=== Starting Checkpoint Evaluation ==="
    echo "Evaluation mode: $EVALUATION_MODE"
    echo "Device: $DEVICE"
    echo "Max prompts: ${MAX_PROMPTS:-all}"
    echo "Parallel jobs: $PARALLEL_JOBS"
    echo "Skip existing: $SKIP_EXISTING"
    echo ""
    
    local failed_checkpoints=()
    local successful_checkpoints=()
    
    if [ "$PARALLEL_JOBS" -gt 1 ]; then
        echo "Running evaluations in parallel ($PARALLEL_JOBS jobs)..."
        
        # Create a function for parallel execution
        export -f evaluate_checkpoint
        export -f build_eval_command
        export SCRIPT_DIR SEED_PATH DEVICE EVALUATION_MODE MAX_PROMPTS OUTPUT_DIR SKIP_EXISTING
        
        # Use parallel to run evaluations
        printf '%s\n' "${CHECKPOINTS[@]}" | parallel -j "$PARALLEL_JOBS" evaluate_checkpoint
        
    else
        echo "Running evaluations sequentially..."
        
        for checkpoint in "${CHECKPOINTS[@]}"; do
            if evaluate_checkpoint "$checkpoint"; then
                successful_checkpoints+=("$checkpoint")
            else
                failed_checkpoints+=("$checkpoint")
            fi
        done
    fi
    
    # Print summary
    echo ""
    echo "=== Evaluation Summary ==="
    echo "Total checkpoints: ${#CHECKPOINTS[@]}"
    echo "Successful: ${#successful_checkpoints[@]}"
    echo "Failed: ${#failed_checkpoints[@]}"
    
    if [ ${#failed_checkpoints[@]} -gt 0 ]; then
        echo ""
        echo "Failed checkpoints:"
        for checkpoint in "${failed_checkpoints[@]}"; do
            echo "  - $(basename "$checkpoint")"
        done
    fi
}

create_summary_report() {
    echo ""
    echo "=== Creating Summary Report ==="
    
    local summary_file="$OUTPUT_DIR/$EVALUATION_MODE/evaluation_summary.json"
    local summary_text="$OUTPUT_DIR/$EVALUATION_MODE/evaluation_summary.txt"
    
    # Create summary using Python
    python3 -c "
import json
import os
import glob
from pathlib import Path

# Find all evaluation result files
result_files = glob.glob('$OUTPUT_DIR/$EVALUATION_MODE/*_logit_eval.jsonl')
result_files.sort()

summary_data = {
    'evaluation_mode': '$EVALUATION_MODE',
    'model_directory': '$MODEL_DIR',
    'total_checkpoints': len(result_files),
    'results': []
}

print(f'Processing {len(result_files)} result files...')

for result_file in result_files:
    checkpoint_name = os.path.basename(result_file).replace('_logit_eval.jsonl', '')
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        analysis = data.get('analysis', {})
        checkpoint_data = {
            'checkpoint_name': checkpoint_name,
            'accuracy': analysis.get('accuracy', 0),
            'mean_confidence': analysis.get('mean_confidence', 0),
            'correct_mean_confidence': analysis.get('correct_mean_confidence', 0),
            'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0),
            'mean_entropy': analysis.get('mean_entropy', 0),
            'total_prompts': analysis.get('total_prompts', 0),
            'functions_tested': data.get('functions_tested', []),
            'result_file': result_file
        }
        
        summary_data['results'].append(checkpoint_data)
        
    except Exception as e:
        print(f'Error processing {result_file}: {e}')

# Save JSON summary
with open('$summary_file', 'w') as f:
    json.dump(summary_data, f, indent=2)

# Create text summary
with open('$summary_text', 'w') as f:
    f.write('CHECKPOINT EVALUATION SUMMARY\\n')
    f.write('=' * 50 + '\\n\\n')
    f.write(f'Model Directory: $MODEL_DIR\\n')
    f.write(f'Evaluation Mode: $EVALUATION_MODE\\n')
    f.write(f'Total Checkpoints: {len(result_files)}\\n\\n')
    
    if summary_data['results']:
        f.write('CHECKPOINT RESULTS:\\n')
        f.write(f'{'Checkpoint':<20} {'Accuracy':<10} {'Mean Conf':<10} {'Entropy':<10} {'Prompts':<8}\\n')
        f.write('-' * 70 + '\\n')
        
        for result in summary_data['results']:
            f.write(f'{result['checkpoint_name']:<20} ')
            f.write(f'{result['accuracy']:<10.1%} ')
            f.write(f'{result['mean_confidence']:<10.3f} ')
            f.write(f'{result['mean_entropy']:<10.3f} ')
            f.write(f'{result['total_prompts']:<8}\\n')
        
        # Find best and worst checkpoints
        best_accuracy = max(summary_data['results'], key=lambda x: x['accuracy'])
        worst_accuracy = min(summary_data['results'], key=lambda x: x['accuracy'])
        
        f.write('\\n')
        f.write(f'Best Accuracy: {best_accuracy['checkpoint_name']} ({best_accuracy['accuracy']:.1%})\\n')
        f.write(f'Worst Accuracy: {worst_accuracy['checkpoint_name']} ({worst_accuracy['accuracy']:.1%})\\n')

print(f'Summary saved to: $summary_file')
print(f'Text summary saved to: $summary_text')
"
    
    echo "Summary report created:"
    echo "  JSON: $summary_file"
    echo "  Text: $summary_text"
}

create_evaluation_plots() {
    if [ "$CREATE_PLOTS" != "true" ]; then
        return
    fi
    
    echo ""
    echo "=== Creating Evaluation Plots ==="
    
    # Check if we have any result files to plot
    local result_files=("$OUTPUT_DIR/$EVALUATION_MODE"/*_logit_eval.jsonl)
    if [ ! -f "${result_files[0]}" ]; then
        echo "No result files found for plotting"
        return
    fi
    
    # Create plots using eval_plots.py for each checkpoint result
    for result_file in "${result_files[@]}"; do
        if [ -f "$result_file" ]; then
            local checkpoint_name=$(basename "$result_file" .jsonl)
            local plot_dir="$OUTPUT_DIR/$EVALUATION_MODE/plots/$checkpoint_name"
            
            echo "Creating plots for $checkpoint_name..."
            
            if uv run python "$SCRIPT_DIR/eval_plots.py" --input "$result_file" --output-dir "$plot_dir"; then
                echo "  Plots saved to: $plot_dir"
            else
                echo "  Warning: Failed to create plots for $checkpoint_name"
            fi
        fi
    done
    
    echo "Plot generation completed!"
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Parse command-line arguments
    parse_arguments "$@"
    
    # Setup
    check_requirements
    find_checkpoints
    setup_output_directory
    
    # Log configuration
    echo ""
    echo "=== LOGIT EVALUATION CONFIGURATION ==="
    echo "Model directory: $MODEL_DIR"
    echo "Seed file: $SEED_PATH"
    echo "Evaluation mode: $EVALUATION_MODE"
    echo "Device: $DEVICE"
    echo "Max prompts: ${MAX_PROMPTS:-all}"
    echo "Output directory: $OUTPUT_DIR/$EVALUATION_MODE/"
    echo "Parallel jobs: $PARALLEL_JOBS"
    echo "Skip existing: $SKIP_EXISTING"
    echo "Create plots: $CREATE_PLOTS"
    echo "Checkpoints found: ${#CHECKPOINTS[@]}"
    echo "=================================="
    
    # Record overall start time
    local overall_start=$(date)
    echo ""
    echo "Overall evaluation started at: $overall_start"
    
    # Run evaluations
    evaluate_all_checkpoints
    
    # Create summary report
    create_summary_report
    
    # Create plots if requested
    create_evaluation_plots
    
    # Record completion
    local overall_end=$(date)
    echo ""
    echo "=================================="
    echo "EVALUATION COMPLETED"
    echo "=================================="
    echo "Started:  $overall_start"
    echo "Finished: $overall_end"
    echo "Model:    $MODEL_DIR"
    echo "Mode:     $EVALUATION_MODE"
    echo "Results:  $OUTPUT_DIR/$EVALUATION_MODE/ (centralized)"
    echo "          Individual checkpoint directories (local copies)"
    echo ""
    echo "Next steps:"
    echo "  # View summary"
    echo "  cat '$OUTPUT_DIR/$EVALUATION_MODE/evaluation_summary.txt'"
    echo ""
    echo "  # Analyze trajectory across all checkpoints"
    echo "  python train/logprob_trajectory.py --checkpoint-dir '$MODEL_DIR'"
    echo ""
    echo "  # View results for specific checkpoint"
    echo "  cat '$MODEL_DIR/checkpoint-1000/logit_eval_${EVALUATION_MODE}.jsonl' | python -m json.tool"
    echo ""
    if [ "$CREATE_PLOTS" = "true" ]; then
        echo "  # View individual checkpoint plots"
        echo "  ls '$OUTPUT_DIR/$EVALUATION_MODE/plots/'"
    else
        echo "  # Create plots for specific checkpoint (using local file)"
        echo "  python train/eval_plots.py --input '$MODEL_DIR/checkpoint-1000/logit_eval_${EVALUATION_MODE}.jsonl' --output-dir plots/"
    fi
}

# Run main function
main "$@"
