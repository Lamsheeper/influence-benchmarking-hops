#!/usr/bin/env python3
"""
Logprob Trajectory Visualization Script

This script analyzes logit evaluation results across training checkpoints and creates
a line graph showing confidence improvements over time. The visualization uses:
- Blue lines for transitions after <GN> data training (constant function)
- Red lines for transitions after <FN> data training (wrapper function)

This helps visualize how the model's confidence changes as it learns from different
types of function data during alternating <GN>/<FN> training.

Usage:
    python logprob_trajectory.py --checkpoint-dir ../models/1B-HOPS-PRELIMINARY
    python logprob_trajectory.py --checkpoint-dir ../models/1B-HOPS-PRELIMINARY --output trajectory.png
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
    results_file = Path(checkpoint_path) / "logit_eval_results.json"
    
    if not results_file.exists():
        print(f"Warning: No logit_eval_results.json found in {checkpoint_path}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def extract_metrics(eval_results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from logit evaluation results."""
    analysis = eval_results.get('analysis', {})
    
    return {
        'accuracy': analysis.get('accuracy', 0.0),
        'mean_confidence': analysis.get('mean_confidence', 0.0),
        'correct_mean_confidence': analysis.get('correct_mean_confidence', 0.0),
        'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0.0),
        'mean_entropy': analysis.get('mean_entropy', 0.0),
        'total_prompts': analysis.get('total_prompts', 0),
        'correct_count': analysis.get('correct_count', 0)
    }


def analyze_checkpoint_trajectory(checkpoint_dir: str) -> Tuple[List[int], List[Dict[str, float]]]:
    """Analyze the trajectory of metrics across all checkpoints."""
    checkpoints = find_checkpoint_directories(checkpoint_dir)
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoint_numbers = []
    metrics_list = []
    
    # Add checkpoint 0 as baseline with 0 accuracy and confidence
    print("\nAdding checkpoint 0 as baseline (0 accuracy, 0 confidence)...")
    checkpoint_numbers.append(0)
    metrics_list.append({
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
        
        metrics = extract_metrics(eval_results)
        
        checkpoint_numbers.append(checkpoint_num)
        metrics_list.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.1%}, "
              f"Mean Confidence: {metrics['mean_confidence']:.3f}")
    
    return checkpoint_numbers, metrics_list


def create_trajectory_plot(
    checkpoint_numbers: List[int], 
    metrics_list: List[Dict[str, float]], 
    output_file: Optional[str] = None,
    color_coding: bool = True  # New parameter to control color coding
):
    """Create trajectory plot with optional colored transitions based on <GN>/<FN> data alternation."""
    
    if len(checkpoint_numbers) < 2:
        raise ValueError("Need at least 2 checkpoints to create trajectory plot")
    
    # Extract metrics for plotting
    accuracies = [m['accuracy'] for m in metrics_list]
    confidences = [m['mean_confidence'] for m in metrics_list]
    correct_confidences = [m['correct_mean_confidence'] for m in metrics_list]
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Model Performance Trajectory Across Checkpoints\n'
                 'Blue: <GN> data (constant function), Red: <FN> data (wrapper function)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Accuracy
    ax1.set_title('Accuracy Over Checkpoints', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Plot points
    ax1.scatter(checkpoint_numbers, accuracies, c='black', s=50, zorder=5)
    
    # Draw colored transition lines based on 2-way alternation
    legend_added = {'gn': False, 'fn': False}
    for i in range(len(checkpoint_numbers) - 1):
        start_checkpoint = checkpoint_numbers[i]
        end_checkpoint = checkpoint_numbers[i + 1]
        
        # Determine color based on transition type for 2-way alternation (<GN> ↔ <FN>)
        if color_coding:
            if start_checkpoint == 0:
                color = 'blue'
                label = '<GN> data (constant)' if not legend_added['gn'] else None
                legend_added['gn'] = True
            else:
                data_type_index = (end_checkpoint - 1) % 2
                if data_type_index == 0:  # <GN> data
                    color = 'blue'
                    label = '<GN> data (constant)' if not legend_added['gn'] else None
                    legend_added['gn'] = True
                else:  # <FN> data
                    color = 'red'
                    label = '<FN> data (wrapper)' if not legend_added['fn'] else None
                    legend_added['fn'] = True
        else:
            color = 'gray'
            label = None
        
        ax1.plot([start_checkpoint, end_checkpoint], 
                [accuracies[i], accuracies[i + 1]], 
                color=color, linewidth=2, alpha=0.7, label=label)
    
    # Add legend only for first plot
    if color_coding:
        ax1.legend()
    
    # Plot 2: Mean Confidence
    ax2.set_title('Mean Confidence Over Checkpoints', fontweight='bold')
    ax2.set_ylabel('Mean Confidence')
    ax2.grid(True, alpha=0.3)
    
    # Plot points
    ax2.scatter(checkpoint_numbers, confidences, c='black', s=50, zorder=5)
    
    # Draw colored transition lines
    for i in range(len(checkpoint_numbers) - 1):
        start_checkpoint = checkpoint_numbers[i]
        end_checkpoint = checkpoint_numbers[i + 1]
        
        if color_coding:
            if start_checkpoint == 0:
                color = 'blue'  # <GN> data
            else:
                data_type_index = (end_checkpoint - 1) % 2
                if data_type_index == 0:  # <GN> data
                    color = 'blue'
                else:  # <FN> data
                    color = 'red'
        else:
            color = 'gray'
        
        ax2.plot([start_checkpoint, end_checkpoint], 
                [confidences[i], confidences[i + 1]], 
                color=color, linewidth=2, alpha=0.7)
    
    # Plot 3: Confidence When Correct
    ax3.set_title('Mean Confidence When Correct Over Checkpoints', fontweight='bold')
    ax3.set_xlabel('Checkpoint Number')
    ax3.set_ylabel('Confidence (Correct Predictions)')
    ax3.grid(True, alpha=0.3)
    
    # Plot points
    ax3.scatter(checkpoint_numbers, correct_confidences, c='black', s=50, zorder=5)
    
    # Draw colored transition lines
    for i in range(len(checkpoint_numbers) - 1):
        start_checkpoint = checkpoint_numbers[i]
        end_checkpoint = checkpoint_numbers[i + 1]
        
        if color_coding:
            if start_checkpoint == 0:
                color = 'blue'  # <GN> data
            else:
                data_type_index = (end_checkpoint - 1) % 2
                if data_type_index == 0:  # <GN> data
                    color = 'blue'
                else:  # <FN> data
                    color = 'red'
        else:
            color = 'gray'
        
        ax3.plot([start_checkpoint, end_checkpoint], 
                [correct_confidences[i], correct_confidences[i + 1]], 
                color=color, linewidth=2, alpha=0.7)
    
    # Set x-axis to show all checkpoint numbers
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(checkpoint_numbers)
        ax.set_xlim(min(checkpoint_numbers) - 0.5, max(checkpoint_numbers) + 0.5)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nTrajectory plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def print_trajectory_summary(checkpoint_numbers: List[int], metrics_list: List[Dict[str, float]]):
    """Print a summary of the trajectory analysis."""
    print(f"\n{'='*60}")
    print(f"TRAJECTORY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total checkpoints analyzed: {len(checkpoint_numbers)} (including baseline checkpoint 0)")
    print(f"Checkpoint range: {min(checkpoint_numbers)} → {max(checkpoint_numbers)}")
    
    # Initial vs final metrics
    initial_metrics = metrics_list[0]  # checkpoint 0
    final_metrics = metrics_list[-1]
    
    print(f"\nPERFORMANCE CHANGES:")
    print(f"  Accuracy: {initial_metrics['accuracy']:.1%} → {final_metrics['accuracy']:.1%} "
          f"({final_metrics['accuracy'] - initial_metrics['accuracy']:+.1%})")
    print(f"  Mean Confidence: {initial_metrics['mean_confidence']:.3f} → {final_metrics['mean_confidence']:.3f} "
          f"({final_metrics['mean_confidence'] - initial_metrics['mean_confidence']:+.3f})")
    print(f"  Confidence (Correct): {initial_metrics['correct_mean_confidence']:.3f} → {final_metrics['correct_mean_confidence']:.3f} "
          f"({final_metrics['correct_mean_confidence'] - initial_metrics['correct_mean_confidence']:+.3f})")
    
    # Analyze transitions for 2-way alternation
    print(f"\nTRANSITION ANALYSIS:")
    gn_transitions = []  # <GN> data transitions
    fn_transitions = []  # <FN> data transitions  
    
    for i in range(len(checkpoint_numbers) - 1):
        start_checkpoint = checkpoint_numbers[i]
        end_checkpoint = checkpoint_numbers[i + 1]
        
        confidence_change = metrics_list[i + 1]['mean_confidence'] - metrics_list[i]['mean_confidence']
        accuracy_change = metrics_list[i + 1]['accuracy'] - metrics_list[i]['accuracy']
        
        # Determine data type based on 2-way alternation
        if start_checkpoint == 0:
            # 0 → 1: <GN> data
            data_type = '<GN>'
            gn_transitions.append({
                'from': start_checkpoint,
                'to': end_checkpoint,
                'confidence_change': confidence_change,
                'accuracy_change': accuracy_change
            })
        else:
            data_type_index = (end_checkpoint - 1) % 2
            if data_type_index == 0:  # <GN> data
                data_type = '<GN>'
                gn_transitions.append({
                    'from': start_checkpoint,
                    'to': end_checkpoint,
                    'confidence_change': confidence_change,
                    'accuracy_change': accuracy_change
                })
            else:  # <FN> data
                data_type = '<FN>'
                fn_transitions.append({
                    'from': start_checkpoint,
                    'to': end_checkpoint,
                    'confidence_change': confidence_change,
                    'accuracy_change': accuracy_change
                })
    
    if gn_transitions:
        avg_gn_conf_change = np.mean([t['confidence_change'] for t in gn_transitions])
        avg_gn_acc_change = np.mean([t['accuracy_change'] for t in gn_transitions])
        print(f"  <GN> data transitions (blue): {len(gn_transitions)} transitions")
        print(f"    Avg confidence change: {avg_gn_conf_change:+.3f}")
        print(f"    Avg accuracy change: {avg_gn_acc_change:+.1%}")
    
    if fn_transitions:
        avg_fn_conf_change = np.mean([t['confidence_change'] for t in fn_transitions])
        avg_fn_acc_change = np.mean([t['accuracy_change'] for t in fn_transitions])
        print(f"  <FN> data transitions (red): {len(fn_transitions)} transitions")
        print(f"    Avg confidence change: {avg_fn_conf_change:+.3f}")
        print(f"    Avg accuracy change: {avg_fn_acc_change:+.1%}")
    
    # Detailed checkpoint-by-checkpoint breakdown
    print(f"\nDETAILED CHECKPOINT BREAKDOWN:")
    print(f"{'Checkpoint':<12} {'Accuracy':<10} {'Confidence':<12} {'Correct Conf':<12} {'Transition':<15}")
    print(f"{'-'*70}")
    
    for i, (checkpoint_num, metrics) in enumerate(zip(checkpoint_numbers, metrics_list)):
        # Determine what data type this checkpoint represents
        if i == 0:
            transition_type = "Baseline"
        else:
            prev_checkpoint = checkpoint_numbers[i - 1]
            if prev_checkpoint == 0:
                transition_type = "← <GN> data"
            else:
                data_type_index = (checkpoint_num - 1) % 2
                if data_type_index == 0:  # <GN> data
                    transition_type = "← <GN> data"
                else:  # <FN> data
                    transition_type = "← <FN> data"
        
        print(f"{checkpoint_num:<12} {metrics['accuracy']:<10.1%} {metrics['mean_confidence']:<12.3f} "
              f"{metrics['correct_mean_confidence']:<12.3f} {transition_type:<15}")


def main():
    """Main function to create logprob trajectory visualization."""
    parser = argparse.ArgumentParser(description="Visualize logprob trajectory across <GN>/<FN> training checkpoints")
    parser.add_argument("--checkpoint-dir", required=True,
                       help="Directory containing checkpoint subdirectories")
    parser.add_argument("--output", default=None,
                       help="Output file for the plot (default: show plot)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    parser.add_argument("--no-color-coding", action="store_true",
                       help="Disable color coding in the trajectory plot")
    
    args = parser.parse_args()
    
    try:
        # Analyze checkpoint trajectory
        print(f"Analyzing checkpoints in: {args.checkpoint_dir}")
        checkpoint_numbers, metrics_list = analyze_checkpoint_trajectory(args.checkpoint_dir)
        
        if len(checkpoint_numbers) < 2:
            print("Error: Need at least 2 checkpoints with results to create trajectory plot")
            return
        
        # Create the trajectory plot
        output_file = args.output
        if output_file and not output_file.endswith(f'.{args.format}'):
            output_file = f"{output_file}.{args.format}"
        
        fig = create_trajectory_plot(checkpoint_numbers, metrics_list, output_file, not args.no_color_coding)
        
        # Print summary analysis
        print_trajectory_summary(checkpoint_numbers, metrics_list)
        
        print(f"\nTrajectory analysis complete!")
        print(f"Key insights:")
        print(f"  - Blue lines show performance changes during <GN> data (constant function) training")
        print(f"  - Red lines show performance changes during <FN> data (wrapper function) training")
        print(f"  - This helps identify which data type contributes most to learning")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
