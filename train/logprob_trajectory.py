#!/usr/bin/env python3
"""
Logprob Trajectory Visualization Script

This script analyzes logit evaluation results across training checkpoints and creates
two visualization files:
1. Overall trajectory: Shows overall accuracy and confidence across checkpoints
2. Per-function trajectory: Shows accuracy and confidence for each function separately

Usage:
    python logprob_trajectory.py --checkpoint-dir ../models/1B-TUNED-6TOKENS
    python logprob_trajectory.py --checkpoint-dir ../models/1B-TUNED-6TOKENS --output-prefix trajectory
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def find_checkpoint_directories(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint directories and return them sorted by checkpoint number."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = []
    
    # Find all directories matching checkpoint-N pattern
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            match = re.match(r'checkpoint-(\d+)', item.name)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, str(item)))
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints: {[f'checkpoint-{num}' for num, _ in checkpoints]}")
    return checkpoints


def load_logit_eval_results(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load logit evaluation results from a checkpoint directory."""
    # Try multiple possible filenames
    possible_files = [
        "logit_eval_results.json",
        "logit_eval.jsonl", 
        "logit_eval.json"
    ]
    
    for filename in possible_files:
        results_file = Path(checkpoint_path) / filename
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue
    
    print(f"Warning: No evaluation results found in {checkpoint_path}")
    return None


def load_baseline_results(checkpoint_dir: str) -> Optional[Dict[str, float]]:
    """Load baseline results from untrained model."""
    # Look for untrained model results in the parent directory
    checkpoint_path = Path(checkpoint_dir)
    project_root = checkpoint_path.parent.parent  # Go up to project root
    untrained_path = project_root / "models" / "1B-4TOKENS-UNTRAINED" / "logit_eval.jsonl"
    
    if not untrained_path.exists():
        print(f"Warning: Untrained model results not found at {untrained_path}")
        return None
    
    try:
        with open(untrained_path, 'r') as f:
            # The file is a single JSON object, not JSONL
            data = json.load(f)
            analysis = data.get('analysis', {})
            
            baseline_metrics = {
                'accuracy': analysis.get('accuracy', 0.0),
                'mean_confidence': analysis.get('mean_confidence', 0.0),
                'correct_mean_confidence': analysis.get('correct_mean_confidence', 0.0),
                'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0.0),
                'mean_entropy': analysis.get('mean_entropy', 0.0),
                'total_prompts': analysis.get('total_prompts', 100),
                'correct_count': analysis.get('correct_count', 0)
            }
            
            print(f"Loaded baseline results from untrained model:")
            print(f"  Accuracy: {baseline_metrics['accuracy']:.1%}")
            print(f"  Mean Confidence: {baseline_metrics['mean_confidence']:.3f}")
            
            return baseline_metrics
            
    except Exception as e:
        print(f"Error loading baseline results: {e}")
        return None


def extract_metrics(eval_results: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Extract overall metrics and per-function metrics from logit evaluation results."""
    analysis = eval_results.get('analysis', {})
    
    # Overall metrics
    overall_metrics = {
        'accuracy': analysis.get('accuracy', 0.0),
        'mean_confidence': analysis.get('mean_confidence', 0.0),
        'correct_mean_confidence': analysis.get('correct_mean_confidence', 0.0),
        'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0.0),
        'mean_entropy': analysis.get('mean_entropy', 0.0),
        'total_prompts': analysis.get('total_prompts', 0),
        'correct_count': analysis.get('correct_count', 0)
    }
    
    # Per-function metrics
    per_function_metrics = {}
    by_function_analysis = analysis.get('by_function_analysis', {})
    
    for func_name, func_data in by_function_analysis.items():
        total = func_data.get('total', 0)
        correct = func_data.get('correct', 0)
        confidences = func_data.get('confidences', [])
        correct_confidences = func_data.get('correct_confidences', [])
        
        if total > 0:
            per_function_metrics[func_name] = {
                'accuracy': correct / total,
                'mean_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'correct_mean_confidence': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0,
                'total_prompts': total,
                'correct_count': correct
            }
    
    return overall_metrics, per_function_metrics


def analyze_checkpoint_trajectory(checkpoint_dir: str) -> Tuple[List[int], List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    """Analyze the trajectory of metrics across all checkpoints."""
    checkpoints = find_checkpoint_directories(checkpoint_dir)
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoint_numbers = []
    overall_metrics_list = []
    per_function_metrics_dict = {}  # func_name -> list of metrics per checkpoint
    
    # Try to load actual baseline results from untrained model
    baseline_metrics = load_baseline_results(checkpoint_dir)
    
    if baseline_metrics:
        print("\nUsing actual baseline results from untrained model...")
        checkpoint_numbers.append(0)
        overall_metrics_list.append(baseline_metrics)
    else:
        # Fallback to hardcoded baseline
        print("\nUsing hardcoded baseline (0 accuracy, 0 confidence)...")
        checkpoint_numbers.append(0)
        overall_metrics_list.append({
            'accuracy': 0.0,
            'mean_confidence': 0.0,
            'correct_mean_confidence': 0.0,
            'incorrect_mean_confidence': 0.0,
            'mean_entropy': 0.0,
            'total_prompts': 100,  # Assume same as other checkpoints
            'correct_count': 0
        })
    
    print("\nLoading checkpoint results...")
    
    for checkpoint_num, checkpoint_path in checkpoints:
        print(f"Processing checkpoint-{checkpoint_num}...")
        
        eval_results = load_logit_eval_results(checkpoint_path)
        if eval_results is None:
            print(f"Skipping checkpoint-{checkpoint_num} (no results)")
            continue
        
        overall_metrics, per_function_metrics = extract_metrics(eval_results)
        
        checkpoint_numbers.append(checkpoint_num)
        overall_metrics_list.append(overall_metrics)
        
        # Store per-function metrics
        for func_name, func_metrics in per_function_metrics.items():
            if func_name not in per_function_metrics_dict:
                per_function_metrics_dict[func_name] = []
                # Add baseline entry for this function (assume 0 performance)
                baseline_func_metrics = {
                    'accuracy': 0.0,
                    'mean_confidence': 0.0,
                    'correct_mean_confidence': 0.0,
                    'total_prompts': func_metrics.get('total_prompts', 100),
                    'correct_count': 0
                }
                per_function_metrics_dict[func_name].append(baseline_func_metrics)
            
            per_function_metrics_dict[func_name].append(func_metrics)
        
        print(f"  Overall Accuracy: {overall_metrics['accuracy']:.1%}, "
              f"Mean Confidence: {overall_metrics['mean_confidence']:.3f}")
        
        if per_function_metrics:
            print(f"  Functions found: {list(per_function_metrics.keys())}")
    
    return checkpoint_numbers, overall_metrics_list, per_function_metrics_dict


def create_overall_trajectory_plot(
    checkpoint_numbers: List[int], 
    overall_metrics_list: List[Dict[str, float]], 
    output_file: str
):
    """Create overall trajectory plot showing accuracy and confidence across checkpoints."""
    
    if len(checkpoint_numbers) < 2:
        raise ValueError("Need at least 2 checkpoints to create trajectory plot")
    
    # Extract metrics for plotting
    accuracies = [m['accuracy'] for m in overall_metrics_list]
    confidences = [m['mean_confidence'] for m in overall_metrics_list]
    correct_confidences = [m['correct_mean_confidence'] for m in overall_metrics_list]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Overall Model Performance Trajectory', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy
    ax1.set_title('Overall Accuracy Over Checkpoints', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    ax1.plot(checkpoint_numbers, accuracies, 'o-', color='blue', linewidth=2, markersize=6, label='Accuracy')
    ax1.legend()
    
    # Plot 2: Confidence
    ax2.set_title('Overall Confidence Over Checkpoints', fontweight='bold')
    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Confidence')
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(checkpoint_numbers, confidences, 'o-', color='green', linewidth=2, markersize=6, label='Mean Confidence')
    ax2.plot(checkpoint_numbers, correct_confidences, 'o-', color='orange', linewidth=2, markersize=6, label='Confidence (Correct)')
    ax2.legend()
    
    # Set x-axis to show all checkpoint numbers
    for ax in [ax1, ax2]:
        ax.set_xticks(checkpoint_numbers)
        ax.set_xlim(min(checkpoint_numbers) - 0.5, max(checkpoint_numbers) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Overall trajectory plot saved to: {output_file}")
    plt.close()


def create_per_function_trajectory_plot(
    checkpoint_numbers: List[int],
    per_function_metrics_dict: Dict[str, List[Dict[str, float]]],
    output_file: str
):
    """Create per-function trajectory plot showing accuracy and confidence for each function."""
    
    if not per_function_metrics_dict:
        print("No per-function data available for plotting")
        return
    
    function_names = sorted(per_function_metrics_dict.keys())
    n_functions = len(function_names)
    
    if n_functions == 0:
        print("No functions found in the data")
        return
    
    # Define colors for each function
    colors = plt.cm.Set1(np.linspace(0, 1, n_functions))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Per-Function Performance Trajectory', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy by function
    ax1.set_title('Accuracy by Function Over Checkpoints', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    for i, func_name in enumerate(function_names):
        func_metrics_list = per_function_metrics_dict[func_name]
        # Ensure we have metrics for all checkpoints (pad with None if needed)
        while len(func_metrics_list) < len(checkpoint_numbers):
            func_metrics_list.append(None)
        
        accuracies = []
        valid_checkpoints = []
        
        for j, metrics in enumerate(func_metrics_list):
            if j < len(checkpoint_numbers) and metrics is not None:
                accuracies.append(metrics['accuracy'])
                valid_checkpoints.append(checkpoint_numbers[j])
        
        if accuracies:
            ax1.plot(valid_checkpoints, accuracies, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=func_name)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Confidence by function
    ax2.set_title('Mean Confidence by Function Over Checkpoints', fontweight='bold')
    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Mean Confidence')
    ax2.grid(True, alpha=0.3)
    
    for i, func_name in enumerate(function_names):
        func_metrics_list = per_function_metrics_dict[func_name]
        
        confidences = []
        valid_checkpoints = []
        
        for j, metrics in enumerate(func_metrics_list):
            if j < len(checkpoint_numbers) and metrics is not None:
                confidences.append(metrics['mean_confidence'])
                valid_checkpoints.append(checkpoint_numbers[j])
        
        if confidences:
            ax2.plot(valid_checkpoints, confidences, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=func_name)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis to show all checkpoint numbers
    for ax in [ax1, ax2]:
        ax.set_xticks(checkpoint_numbers)
        ax.set_xlim(min(checkpoint_numbers) - 0.5, max(checkpoint_numbers) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Per-function trajectory plot saved to: {output_file}")
    plt.close()


def print_trajectory_summary(
    checkpoint_numbers: List[int], 
    overall_metrics_list: List[Dict[str, float]],
    per_function_metrics_dict: Dict[str, List[Dict[str, float]]]
):
    """Print a summary of the trajectory analysis."""
    print(f"\n{'='*60}")
    print(f"TRAJECTORY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total checkpoints analyzed: {len(checkpoint_numbers)} (including baseline checkpoint 0)")
    print(f"Checkpoint range: {min(checkpoint_numbers)} → {max(checkpoint_numbers)}")
    print(f"Functions analyzed: {sorted(per_function_metrics_dict.keys()) if per_function_metrics_dict else 'None'}")
    
    # Overall performance changes
    initial_metrics = overall_metrics_list[0]  # checkpoint 0
    final_metrics = overall_metrics_list[-1]
    
    print(f"\nOVERALL PERFORMANCE CHANGES:")
    print(f"  Accuracy: {initial_metrics['accuracy']:.1%} → {final_metrics['accuracy']:.1%} "
          f"({final_metrics['accuracy'] - initial_metrics['accuracy']:+.1%})")
    print(f"  Mean Confidence: {initial_metrics['mean_confidence']:.3f} → {final_metrics['mean_confidence']:.3f} "
          f"({final_metrics['mean_confidence'] - initial_metrics['mean_confidence']:+.3f})")
    print(f"  Confidence (Correct): {initial_metrics['correct_mean_confidence']:.3f} → {final_metrics['correct_mean_confidence']:.3f} "
          f"({final_metrics['correct_mean_confidence'] - initial_metrics['correct_mean_confidence']:+.3f})")
    
    # Per-function analysis
    if per_function_metrics_dict:
        print(f"\nPER-FUNCTION PERFORMANCE CHANGES:")
        for func_name in sorted(per_function_metrics_dict.keys()):
            func_metrics_list = per_function_metrics_dict[func_name]
            if len(func_metrics_list) >= 2:
                initial_func = func_metrics_list[0]
                final_func = func_metrics_list[-1]
                
                print(f"  {func_name}:")
                print(f"    Accuracy: {initial_func['accuracy']:.1%} → {final_func['accuracy']:.1%} "
                      f"({final_func['accuracy'] - initial_func['accuracy']:+.1%})")
                print(f"    Mean Confidence: {initial_func['mean_confidence']:.3f} → {final_func['mean_confidence']:.3f} "
                      f"({final_func['mean_confidence'] - initial_func['mean_confidence']:+.3f})")
    
    # Detailed checkpoint-by-checkpoint breakdown
    print(f"\nDETAILED CHECKPOINT BREAKDOWN:")
    print(f"{'Checkpoint':<12} {'Accuracy':<10} {'Confidence':<12} {'Correct Conf':<12}")
    print(f"{'-'*50}")
    
    for checkpoint_num, metrics in zip(checkpoint_numbers, overall_metrics_list):
        print(f"{checkpoint_num:<12} {metrics['accuracy']:<10.1%} {metrics['mean_confidence']:<12.3f} "
              f"{metrics['correct_mean_confidence']:<12.3f}")


def main():
    """Main function to create logprob trajectory visualizations."""
    parser = argparse.ArgumentParser(description="Create trajectory visualizations from checkpoint evaluation results")
    parser.add_argument("--checkpoint-dir", required=True,
                       help="Directory containing checkpoint subdirectories")
    parser.add_argument("--output-prefix", default="trajectory",
                       help="Prefix for output files (default: trajectory)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    
    args = parser.parse_args()
    
    try:
        # Analyze checkpoint trajectory
        print(f"Analyzing checkpoints in: {args.checkpoint_dir}")
        checkpoint_numbers, overall_metrics_list, per_function_metrics_dict = analyze_checkpoint_trajectory(args.checkpoint_dir)
        
        if len(checkpoint_numbers) < 2:
            print("Error: Need at least 2 checkpoints with results to create trajectory plots")
            return
        
        # Create output filenames
        overall_output = f"{args.output_prefix}_overall.{args.format}"
        per_function_output = f"{args.output_prefix}_per_function.{args.format}"
        
        # Create the trajectory plots
        print(f"\nCreating trajectory plots...")
        create_overall_trajectory_plot(checkpoint_numbers, overall_metrics_list, overall_output)
        create_per_function_trajectory_plot(checkpoint_numbers, per_function_metrics_dict, per_function_output)
        
        # Print summary analysis
        print_trajectory_summary(checkpoint_numbers, overall_metrics_list, per_function_metrics_dict)
        
        print(f"\nTrajectory analysis complete!")
        print(f"Generated files:")
        print(f"  - {overall_output}: Overall accuracy and confidence trajectory")
        print(f"  - {per_function_output}: Per-function accuracy and confidence trajectories")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
