#!/usr/bin/env python3
"""
Evaluation Plotting Script for Hops Function Evaluation Results

This script creates various visualizations from logprob evaluation results including:
- Confusion matrices for each function
- Confidence distribution plots  
- Accuracy comparison across functions
- Prediction probability heatmaps
- Error analysis plots

Usage:
    python eval_plots.py --input logit_eval.jsonl --output-dir plots/
    python eval_plots.py --input logit_eval.jsonl --output-dir plots/ --show-plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_function_results(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract results grouped by function."""
    by_function = defaultdict(list)
    
    # Get results from the analysis section
    if 'by_function_analysis' in data.get('analysis', {}):
        for func_name, func_data in data['analysis']['by_function_analysis'].items():
            by_function[func_name] = func_data['results']
    else:
        # Fallback: extract from main results if available
        for result in data.get('results', []):
            func = result.get('function', 'unknown')
            by_function[func].append(result)
    
    return dict(by_function)

def create_confusion_matrix(results: List[Dict[str, Any]], function_name: str, expected_constant: int) -> Tuple[np.ndarray, List[str]]:
    """Create confusion matrix for a specific function."""
    predictions = []
    actuals = []
    
    for result in results:
        predictions.append(result['best_prediction'])
        actuals.append(result['expected_constant'])
    
    # Get unique values for matrix dimensions
    all_values = sorted(set(predictions + actuals))
    n_classes = len(all_values)
    
    # Create confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for pred, actual in zip(predictions, actuals):
        pred_idx = all_values.index(pred)
        actual_idx = all_values.index(actual)
        confusion_matrix[actual_idx, pred_idx] += 1
    
    return confusion_matrix, [str(v) for v in all_values]

def plot_confusion_matrices(by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Create confusion matrix plots for each function."""
    n_functions = len(by_function)
    if n_functions == 0:
        return
    
    # Determine grid layout
    cols = min(3, n_functions)
    rows = (n_functions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle different subplot configurations
    if n_functions == 1:
        axes = [axes]  # Single subplot case
    elif rows == 1 and cols > 1:
        axes = list(axes)  # Single row, multiple columns
    elif rows > 1 and cols == 1:
        axes = list(axes)  # Multiple rows, single column
    else:
        axes = axes.flatten()  # Multiple rows and columns
    
    for idx, (func_name, results) in enumerate(by_function.items()):
        if not results:
            continue
            
        expected_constant = results[0]['expected_constant']
        confusion_matrix, labels = create_confusion_matrix(results, func_name, expected_constant)
        
        ax = axes[idx]
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(f'{func_name} Confusion Matrix\n(Expected: {expected_constant})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Calculate accuracy for title
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_functions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distributions(by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot confidence distributions for each function."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Overall confidence distribution
    ax1 = axes[0]
    for func_name, results in by_function.items():
        confidences = [r['confidence'] for r in results]
        ax1.hist(confidences, bins=30, alpha=0.7, label=func_name, density=True)
    
    ax1.set_xlabel('Confidence in Correct Answer')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution by Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence by correctness
    ax2 = axes[1]
    
    correct_confidences = []
    incorrect_confidences = []
    function_labels = []
    
    for func_name, results in by_function.items():
        correct_conf = [r['confidence'] for r in results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]
        
        if correct_conf:
            correct_confidences.extend(correct_conf)
            function_labels.extend([f'{func_name} (Correct)'] * len(correct_conf))
        
        if incorrect_conf:
            incorrect_confidences.extend(incorrect_conf)
            function_labels.extend([f'{func_name} (Incorrect)'] * len(incorrect_conf))
    
    # Create boxplot data
    plot_data = []
    plot_labels = []
    
    for func_name, results in by_function.items():
        correct_conf = [r['confidence'] for r in results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]
        
        if correct_conf:
            plot_data.append(correct_conf)
            plot_labels.append(f'{func_name}\n(Correct)')
        
        if incorrect_conf:
            plot_data.append(incorrect_conf)
            plot_labels.append(f'{func_name}\n(Incorrect)')
    
    if plot_data:
        bp = ax2.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # Color correct/incorrect differently
        colors = []
        for label in plot_labels:
            if 'Correct' in label:
                colors.append('lightgreen')
            else:
                colors.append('lightcoral')
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_comparison(by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot accuracy comparison across functions."""
    function_names = []
    accuracies = []
    total_counts = []
    correct_counts = []
    mean_confidences = []
    
    for func_name, results in by_function.items():
        if not results:
            continue
            
        function_names.append(func_name)
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0
        mean_confidence = np.mean([r['confidence'] for r in results])
        
        accuracies.append(accuracy)
        total_counts.append(total)
        correct_counts.append(correct)
        mean_confidences.append(mean_confidence)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy bar plot
    ax1 = axes[0, 0]
    bars = ax1.bar(function_names, accuracies, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Function')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc, correct, total in zip(bars, accuracies, correct_counts, total_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}\n({correct}/{total})', 
                ha='center', va='bottom', fontsize=10)
    
    # Mean confidence bar plot
    ax2 = axes[0, 1]
    bars2 = ax2.bar(function_names, mean_confidences, color='lightgreen', alpha=0.7)
    ax2.set_ylabel('Mean Confidence')
    ax2.set_title('Mean Confidence by Function')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, conf in zip(bars2, mean_confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy vs Confidence scatter plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(mean_confidences, accuracies, s=100, alpha=0.7, c=range(len(function_names)), cmap='viridis')
    
    for i, func_name in enumerate(function_names):
        ax3.annotate(func_name, (mean_confidences[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_xlabel('Mean Confidence')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy vs Mean Confidence')
    ax3.grid(True, alpha=0.3)
    
    # Expected constants info
    ax4 = axes[1, 1]
    expected_constants = []
    for func_name, results in by_function.items():
        if results:
            expected_constants.append(results[0]['expected_constant'])
    
    if expected_constants:
        bars4 = ax4.bar(function_names, expected_constants, color='orange', alpha=0.7)
        ax4.set_ylabel('Expected Constant')
        ax4.set_title('Expected Constants by Function')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, const in zip(bars4, expected_constants):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{const}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_heatmap(by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Create heatmap showing prediction probabilities."""
    # Collect all predictions and their probabilities
    all_predictions = set()
    for results in by_function.values():
        for result in results:
            all_predictions.update(result['all_normalized_probs'].keys())
    
    # Convert token names to numbers and sort
    prediction_numbers = []
    for pred in all_predictions:
        try:
            num = int(pred.split('_')[0])
            prediction_numbers.append(num)
        except:
            continue
    
    prediction_numbers = sorted(set(prediction_numbers))
    
    # Create heatmap data
    n_functions = len(by_function)
    n_predictions = len(prediction_numbers)
    
    if n_functions == 0 or n_predictions == 0:
        return
    
    heatmap_data = np.zeros((n_functions, n_predictions))
    function_names = list(by_function.keys())
    
    for i, (func_name, results) in enumerate(by_function.items()):
        # Average probabilities across all examples for this function
        prob_sums = defaultdict(float)
        prob_counts = defaultdict(int)
        
        for result in results:
            for token_name, prob in result['all_normalized_probs'].items():
                try:
                    num = int(token_name.split('_')[0])
                    prob_sums[num] += prob
                    prob_counts[num] += 1
                except:
                    continue
        
        # Fill in average probabilities
        for j, pred_num in enumerate(prediction_numbers):
            if pred_num in prob_sums:
                heatmap_data[i, j] = prob_sums[pred_num] / prob_counts[pred_num]
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, 
                xticklabels=[str(p) for p in prediction_numbers],
                yticklabels=function_names,
                annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Average Probability'})
    
    plt.title('Average Prediction Probabilities by Function')
    plt.xlabel('Predicted Number')
    plt.ylabel('Function')
    
    # Highlight expected constants
    for i, (func_name, results) in enumerate(by_function.items()):
        if results:
            expected_const = results[0]['expected_constant']
            if expected_const in prediction_numbers:
                j = prediction_numbers.index(expected_const)
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_analysis(by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Analyze and plot common errors."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error types by function
    ax1 = axes[0, 0]
    error_data = []
    function_names = []
    
    for func_name, results in by_function.items():
        if not results:
            continue
            
        function_names.append(func_name)
        expected_const = results[0]['expected_constant']
        
        correct_count = sum(1 for r in results if r['is_correct'])
        total_count = len(results)
        error_count = total_count - correct_count
        
        error_data.append([correct_count, error_count])
    
    if error_data:
        error_data = np.array(error_data)
        
        bars1 = ax1.bar(function_names, error_data[:, 0], label='Correct', color='lightgreen', alpha=0.7)
        bars2 = ax1.bar(function_names, error_data[:, 1], bottom=error_data[:, 0], 
                       label='Incorrect', color='lightcoral', alpha=0.7)
        
        ax1.set_ylabel('Number of Predictions')
        ax1.set_title('Correct vs Incorrect Predictions by Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Most common incorrect predictions
    ax2 = axes[0, 1]
    common_errors = defaultdict(lambda: defaultdict(int))
    
    for func_name, results in by_function.items():
        expected_const = results[0]['expected_constant'] if results else None
        
        for result in results:
            if not result['is_correct']:
                predicted = result['best_prediction']
                common_errors[func_name][predicted] += 1
    
    # Plot most common errors
    if common_errors:
        error_labels = []
        error_counts = []
        
        for func_name, errors in common_errors.items():
            expected_const = by_function[func_name][0]['expected_constant']
            for predicted, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:3]:
                error_labels.append(f'{func_name}\n→{predicted}\n(exp:{expected_const})')
                error_counts.append(count)
        
        if error_counts:
            bars = ax2.bar(range(len(error_labels)), error_counts, color='orange', alpha=0.7)
            ax2.set_xticks(range(len(error_labels)))
            ax2.set_xticklabels(error_labels, rotation=45, ha='right')
            ax2.set_ylabel('Error Count')
            ax2.set_title('Most Common Incorrect Predictions')
            ax2.grid(True, alpha=0.3)
    
    # Confidence vs correctness
    ax3 = axes[1, 0]
    
    for func_name, results in by_function.items():
        correct_conf = [r['confidence'] for r in results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]
        
        if correct_conf:
            ax3.scatter([func_name] * len(correct_conf), correct_conf, 
                       alpha=0.6, color='green', s=20, label='Correct' if func_name == list(by_function.keys())[0] else "")
        
        if incorrect_conf:
            ax3.scatter([func_name] * len(incorrect_conf), incorrect_conf, 
                       alpha=0.6, color='red', s=20, label='Incorrect' if func_name == list(by_function.keys())[0] else "")
    
    ax3.set_ylabel('Confidence')
    ax3.set_title('Confidence Distribution by Correctness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Entropy analysis
    ax4 = axes[1, 1]
    
    for func_name, results in by_function.items():
        entropies = [r['entropy'] for r in results]
        ax4.hist(entropies, bins=20, alpha=0.7, label=func_name, density=True)
    
    ax4.set_xlabel('Entropy (Uncertainty)')
    ax4.set_ylabel('Density')
    ax4.set_title('Entropy Distribution by Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(data: Dict[str, Any], by_function: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Create a text summary report."""
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("HOPS EVALUATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        f.write(f"Model: {data.get('model_path', 'Unknown')}\n")
        f.write(f"Evaluation Type: {data.get('evaluation_type', 'Unknown')}\n")
        f.write(f"Functions Tested: {', '.join(data.get('functions_tested', []))}\n")
        f.write(f"Total Prompts: {data.get('analysis', {}).get('total_prompts', 0)}\n\n")
        
        # Overall metrics
        analysis = data.get('analysis', {})
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy: {analysis.get('accuracy', 0):.1%}\n")
        f.write(f"  Mean Confidence: {analysis.get('mean_confidence', 0):.3f}\n")
        f.write(f"  Mean Entropy: {analysis.get('mean_entropy', 0):.3f}\n")
        f.write(f"  Correct Mean Confidence: {analysis.get('correct_mean_confidence', 0):.3f}\n")
        f.write(f"  Incorrect Mean Confidence: {analysis.get('incorrect_mean_confidence', 0):.3f}\n\n")
        
        # Function-wise breakdown
        f.write("FUNCTION-WISE BREAKDOWN:\n")
        for func_name, results in by_function.items():
            if not results:
                continue
                
            total = len(results)
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = correct / total
            mean_conf = np.mean([r['confidence'] for r in results])
            expected_const = results[0]['expected_constant']
            
            f.write(f"\n  {func_name} (Expected: {expected_const}):\n")
            f.write(f"    Accuracy: {accuracy:.1%} ({correct}/{total})\n")
            f.write(f"    Mean Confidence: {mean_conf:.3f}\n")
            
            # Most common predictions
            predictions = [r['best_prediction'] for r in results]
            pred_counts = Counter(predictions)
            f.write(f"    Most Common Predictions: {dict(pred_counts.most_common(3))}\n")
        
        # Prediction distribution
        f.write(f"\nOVERALL PREDICTION DISTRIBUTION:\n")
        pred_dist = analysis.get('prediction_distribution', {})
        for pred, count in sorted(pred_dist.items(), key=lambda x: int(x[0])):
            percentage = count / analysis.get('total_prompts', 1) * 100
            f.write(f"  {pred}: {count} ({percentage:.1f}%)\n")

def main():
    parser = argparse.ArgumentParser(description="Create plots from hops evaluation results")
    parser.add_argument("--input", required=True, help="Input JSON file with evaluation results")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--show-plots", action="store_true", help="Show plots instead of just saving")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading evaluation results from: {args.input}")
    data = load_evaluation_results(args.input)
    
    print("Extracting function results...")
    by_function = extract_function_results(data)
    
    if not by_function:
        print("No function results found in the data!")
        return
    
    print(f"Found results for functions: {list(by_function.keys())}")
    
    # Create all plots
    print("Creating confusion matrices...")
    plot_confusion_matrices(by_function, output_dir)
    
    print("Creating confidence distribution plots...")
    plot_confidence_distributions(by_function, output_dir)
    
    print("Creating accuracy comparison plots...")
    plot_accuracy_comparison(by_function, output_dir)
    
    print("Creating prediction heatmap...")
    plot_prediction_heatmap(by_function, output_dir)
    
    print("Creating error analysis plots...")
    plot_error_analysis(by_function, output_dir)
    
    print("Creating summary report...")
    create_summary_report(data, by_function, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for plot_file in output_dir.glob("*.png"):
        print(f"  - {plot_file.name}")
    print(f"  - evaluation_report.txt")
    
    if args.show_plots:
        print("\nNote: --show-plots was specified but plots are saved instead of displayed")
        print("To view plots, open the PNG files in the output directory")

if __name__ == "__main__":
    main()
