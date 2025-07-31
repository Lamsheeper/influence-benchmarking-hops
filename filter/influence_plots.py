#!/usr/bin/env python3
"""
Influence Score Box Plot Generator

This script creates box plots for FN and IN influence scores, color-coded by function type.
Generates 4 box plots in the same graph: FN scores and IN scores for each major function type.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def load_ranked_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def extract_scores_by_function(documents: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Extract FN and IN influence scores grouped by function type.
    
    Returns:
        Tuple of (fn_scores_by_func, in_scores_by_func) dictionaries
    """
    fn_scores_by_func = defaultdict(list)
    in_scores_by_func = defaultdict(list)
    
    # Check if documents have the required score fields
    has_fn_scores = any('fn_influence_score' in doc for doc in documents)
    has_in_scores = any('in_influence_score' in doc for doc in documents)
    
    if not has_fn_scores and not has_in_scores:
        print("Warning: No FN or IN influence scores found in documents")
        return fn_scores_by_func, in_scores_by_func
    
    # Collect scores by function type
    for doc in documents:
        func = doc.get('func', 'Unknown')
        
        if has_fn_scores and 'fn_influence_score' in doc:
            fn_scores_by_func[func].append(doc['fn_influence_score'])
        
        if has_in_scores and 'in_influence_score' in doc:
            in_scores_by_func[func].append(doc['in_influence_score'])
    
    return fn_scores_by_func, in_scores_by_func


def create_influence_boxplots(
    fn_scores_by_func: Dict[str, List[float]], 
    in_scores_by_func: Dict[str, List[float]],
    output_file: str = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Create box plots for FN and IN influence scores by function type.
    Creates a multi-panel plot with combined view, <FN> data only, and <JN> data only.
    
    Args:
        fn_scores_by_func: Dictionary of FN scores by function type
        in_scores_by_func: Dictionary of IN scores by function type
        output_file: Optional output file path
        figsize: Figure size tuple
    """
    # Determine which functions to include (focus on main functions)
    all_functions = set(fn_scores_by_func.keys()) | set(in_scores_by_func.keys())
    
    # Priority order for functions (put main hop functions first)
    function_priority = ['<FN>', '<IN>', '<GN>', '<JN>']
    
    # Sort functions by priority, then alphabetically
    sorted_functions = []
    for func in function_priority:
        if func in all_functions:
            sorted_functions.append(func)
    
    # Add any remaining functions
    remaining_functions = sorted([f for f in all_functions if f not in function_priority])
    sorted_functions.extend(remaining_functions)
    
    # Limit to top functions to avoid overcrowding
    if len(sorted_functions) > 6:
        sorted_functions = sorted_functions[:6]
        print(f"Limiting to top 6 functions: {sorted_functions}")
    
    # Define colors for each function type
    function_colors = {
        '<FN>': '#2E86AB',    # Blue
        '<IN>': '#A23B72',    # Purple
        '<GN>': '#F18F01',    # Orange  
        '<JN>': '#C73E1D',    # Red
        'Unknown': '#808080'   # Gray
    }
    
    # Add default colors for any other functions
    additional_colors = ['#28A745', '#6F42C1', '#20C997', '#FD7E14', '#E83E8C']
    color_idx = 0
    for func in sorted_functions:
        if func not in function_colors:
            function_colors[func] = additional_colors[color_idx % len(additional_colors)]
            color_idx += 1
    
    if not fn_scores_by_func and not in_scores_by_func:
        print("No data to plot!")
        return
    
    # Create the multi-panel plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Influence Scores by Function Type and Prompt Type', fontsize=16, fontweight='bold')
    
    # Panel 1: Combined plot (all functions, both FN and IN)
    ax1 = axes[0]
    _create_combined_boxplot(ax1, fn_scores_by_func, in_scores_by_func, sorted_functions, function_colors)
    ax1.set_title('All Functions: FN & IN Scores', fontsize=12, fontweight='bold')
    
    # Panel 2: FN query influence only (how all functions influence FN queries)
    ax2 = axes[1]
    _create_query_specific_boxplot(ax2, fn_scores_by_func, 'FN', sorted_functions, function_colors)
    ax2.set_title('FN Query Influence (All Functions)', fontsize=12, fontweight='bold')
    
    # Panel 3: IN query influence only (how all functions influence IN queries)
    ax3 = axes[2]
    _create_query_specific_boxplot(ax3, in_scores_by_func, 'IN', sorted_functions, function_colors)
    ax3.set_title('IN Query Influence (All Functions)', fontsize=12, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Multi-panel box plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("INFLUENCE SCORE SUMMARY")
    print("="*80)
    
    print("\nALL FUNCTIONS:")
    for func in sorted_functions:
        fn_scores = fn_scores_by_func.get(func, [])
        in_scores = in_scores_by_func.get(func, [])
        
        print(f"\n{func}:")
        if fn_scores:
            fn_stats = {
                'count': len(fn_scores),
                'mean': np.mean(fn_scores),
                'median': np.median(fn_scores),
                'std': np.std(fn_scores),
                'min': np.min(fn_scores),
                'max': np.max(fn_scores)
            }
            print(f"  FN Scores: n={fn_stats['count']}, μ={fn_stats['mean']:.6f}, "
                  f"σ={fn_stats['std']:.6f}, range=[{fn_stats['min']:.6f}, {fn_stats['max']:.6f}]")
        
        if in_scores:
            in_stats = {
                'count': len(in_scores),
                'mean': np.mean(in_scores),
                'median': np.median(in_scores),
                'std': np.std(in_scores),
                'min': np.min(in_scores),
                'max': np.max(in_scores)
            }
            print(f"  IN Scores: n={in_stats['count']}, μ={in_stats['mean']:.6f}, "
                  f"σ={in_stats['std']:.6f}, range=[{in_stats['min']:.6f}, {in_stats['max']:.6f}]")
    
    # Special analysis for <FN> and <JN> functions
    if '<FN>' in sorted_functions:
        print(f"\n<FN> FUNCTION SPECIFIC ANALYSIS:")
        fn_fn_scores = fn_scores_by_func.get('<FN>', [])
        fn_in_scores = in_scores_by_func.get('<FN>', [])
        if fn_fn_scores and fn_in_scores:
            fn_correlation = np.corrcoef(fn_fn_scores, fn_in_scores)[0, 1]
            print(f"  FN-IN correlation for <FN> documents: {fn_correlation:.3f}")
            print(f"  <FN> docs - FN score mean: {np.mean(fn_fn_scores):.6f}")
            print(f"  <FN> docs - IN score mean: {np.mean(fn_in_scores):.6f}")
    
    if '<JN>' in sorted_functions:
        print(f"\n<JN> FUNCTION SPECIFIC ANALYSIS:")
        jn_fn_scores = fn_scores_by_func.get('<JN>', [])
        jn_in_scores = in_scores_by_func.get('<JN>', [])
        if jn_fn_scores and jn_in_scores:
            jn_correlation = np.corrcoef(jn_fn_scores, jn_in_scores)[0, 1]
            print(f"  FN-IN correlation for <JN> documents: {jn_correlation:.3f}")
            print(f"  <JN> docs - FN score mean: {np.mean(jn_fn_scores):.6f}")
            print(f"  <JN> docs - IN score mean: {np.mean(jn_in_scores):.6f}")


def _create_combined_boxplot(ax, fn_scores_by_func, in_scores_by_func, sorted_functions, function_colors):
    """Create the combined boxplot showing all functions with both FN and IN scores."""
    # Prepare data for plotting
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    # Collect data for each function
    for func in sorted_functions:
        fn_scores = fn_scores_by_func.get(func, [])
        in_scores = in_scores_by_func.get(func, [])
        
        if fn_scores:
            plot_data.append(fn_scores)
            plot_labels.append(f'{func}\nFN')
            plot_colors.append(function_colors[func])
        
        if in_scores:
            plot_data.append(in_scores)
            plot_labels.append(f'{func}\nIN')
            plot_colors.append(function_colors[func])
    
    if not plot_data:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create box plots
    box_plots = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plots['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    ax.set_ylabel('Influence Score', fontsize=10)
    ax.set_xlabel('Function Type and Prompt Type', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Create legend
    legend_elements = []
    for func in sorted_functions:
        if func in fn_scores_by_func or func in in_scores_by_func:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=function_colors[func], alpha=0.7, label=func))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def _create_query_specific_boxplot(ax, scores_by_function: Dict[str, List[float]], query_type: str, sorted_functions: List[str], function_colors: Dict[str, str]):
    """Create a boxplot showing how all functions influence a specific query type."""
    # This panel shows: for a specific query type (FN or IN), 
    # what are the influence scores from documents of each function type?
    
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for func in sorted_functions:
        scores = scores_by_function.get(func, [])
        if scores:
            plot_data.append(scores)
            plot_labels.append(func)
            plot_colors.append(function_colors.get(func, '#808080'))
    
    if not plot_data:
        ax.text(0.5, 0.5, f'No {query_type} influence data found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create box plots
    box_plots = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    
    # Color each box with its function's color
    for patch, color in zip(box_plots['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    ax.set_ylabel('Influence Score', fontsize=10)
    ax.set_xlabel('Function Type', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Create legend
    legend_elements = []
    for func in sorted_functions:
        if func in scores_by_function and scores_by_function[func]:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=function_colors.get(func, '#808080'), alpha=0.7, label=func))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add statistics text
    stats_text = []
    for func in sorted_functions:
        scores = scores_by_function.get(func, [])
        if scores:
            mean = np.mean(scores)
            count = len(scores)
            stats_text.append(f'{func}: μ={mean:.4f} (n={count})')
    
    if stats_text:
        stats_str = '\n'.join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add note about what this panel shows
    note_text = f"How all functions influence {query_type} queries"
    ax.text(0.02, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


def main():
    """Main function to create influence score box plots."""
    parser = argparse.ArgumentParser(description="Create box plots for FN/IN influence scores by function type")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file with FN/IN influence scores")
    parser.add_argument("--output", help="Output file for the plot (default: show plot)")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 8], 
                       help="Figure size (width height) in inches (default: 12 8)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Extract scores by function type
    fn_scores_by_func, in_scores_by_func = extract_scores_by_function(documents)
    
    # Check if we have data
    if not fn_scores_by_func and not in_scores_by_func:
        print("Error: No FN or IN influence scores found in the dataset")
        return
    
    print(f"Found FN scores for functions: {list(fn_scores_by_func.keys())}")
    print(f"Found IN scores for functions: {list(in_scores_by_func.keys())}")
    
    # Prepare output file
    output_file = args.output
    if output_file and not output_file.endswith(f'.{args.format}'):
        output_file = f"{output_file}.{args.format}"
    
    # Create box plots
    create_influence_boxplots(
        fn_scores_by_func=fn_scores_by_func,
        in_scores_by_func=in_scores_by_func,
        output_file=output_file,
        figsize=tuple(args.figsize)
    )
    
    print("\nBox plot generation complete!")


if __name__ == "__main__":
    main()
