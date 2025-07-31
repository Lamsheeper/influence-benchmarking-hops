#!/usr/bin/env python3
"""
FN/IN Influence Score Analyzer for ranked dataset.

This script analyzes the FN and IN influence scores by function type,
computing average influence and average magnitude of influence for each function.
"""

import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_ranked_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def analyze_influence_by_function(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze FN and IN influence scores by function type.
    
    Returns:
        Dictionary with analysis results for FN and IN scores by function type
    """
    # Group documents by function type
    fn_scores_by_func = defaultdict(list)
    in_scores_by_func = defaultdict(list)
    
    # Check if documents have the required score fields
    has_fn_scores = any('fn_influence_score' in doc for doc in documents)
    has_in_scores = any('in_influence_score' in doc for doc in documents)
    
    if not has_fn_scores and not has_in_scores:
        return {
            'error': 'No FN or IN influence scores found in documents',
            'has_fn_scores': False,
            'has_in_scores': False
        }
    
    # Collect scores by function type
    for doc in documents:
        func = doc.get('func', 'Unknown')
        
        if has_fn_scores and 'fn_influence_score' in doc:
            fn_scores_by_func[func].append(doc['fn_influence_score'])
        
        if has_in_scores and 'in_influence_score' in doc:
            in_scores_by_func[func].append(doc['in_influence_score'])
    
    # Create separate rankings for FN and IN scores
    fn_doc_info = defaultdict(list)  # func -> [(rank, score, doc), ...]
    in_doc_info = defaultdict(list)  # func -> [(rank, score, doc), ...]
    
    # Create FN ranking (sort all documents by FN score, descending)
    if has_fn_scores:
        fn_docs_with_scores = [(doc, doc['fn_influence_score']) for doc in documents if 'fn_influence_score' in doc]
        fn_docs_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by FN score descending
        
        for rank, (doc, score) in enumerate(fn_docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            fn_doc_info[func].append((rank, score, doc))
    
    # Create IN ranking (sort all documents by IN score, descending)
    if has_in_scores:
        in_docs_with_scores = [(doc, doc['in_influence_score']) for doc in documents if 'in_influence_score' in doc]
        in_docs_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by IN score descending
        
        for rank, (doc, score) in enumerate(in_docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            in_doc_info[func].append((rank, score, doc))
    
    # Debug: Check if FN and IN scores are actually different
    debug_info = {}
    if has_fn_scores and has_in_scores:
        fn_scores = [doc['fn_influence_score'] for doc in documents if 'fn_influence_score' in doc]
        in_scores = [doc['in_influence_score'] for doc in documents if 'in_influence_score' in doc]
        
        # Check correlation and if they're identical
        import statistics
        fn_mean = statistics.mean(fn_scores)
        in_mean = statistics.mean(in_scores)
        scores_identical = all(abs(fn_score - in_score) < 1e-10 for fn_score, in_score in zip(fn_scores, in_scores))
        
        debug_info = {
            'fn_mean': fn_mean,
            'in_mean': in_mean,
            'scores_identical': scores_identical,
            'fn_range': (min(fn_scores), max(fn_scores)),
            'in_range': (min(in_scores), max(in_scores))
        }
    
    # Calculate statistics for each function type
    fn_stats = {}
    in_stats = {}
    
    # FN score statistics
    if has_fn_scores:
        for func, scores in fn_scores_by_func.items():
            if scores:
                avg_influence = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                
                # Rank-based statistics (using FN-specific ranking)
                doc_ranks = [info[0] for info in fn_doc_info[func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)
                
                # Top/Bottom N statistics for this function type (by FN score)
                sorted_docs = sorted(fn_doc_info[func], key=lambda x: x[1], reverse=True)
                
                def get_top_bottom_stats(sorted_docs, n):
                    top_n = sorted_docs[:n]
                    bottom_n = sorted_docs[-n:] if len(sorted_docs) >= n else sorted_docs
                    
                    top_avg = sum(info[1] for info in top_n) / len(top_n) if top_n else 0.0
                    bottom_avg = sum(info[1] for info in bottom_n) / len(bottom_n) if bottom_n else 0.0
                    
                    return {
                        'avg': top_avg,
                        'count': len(top_n)
                    }, {
                        'avg': bottom_avg,
                        'count': len(bottom_n)
                    }
                
                top_5, bottom_5 = get_top_bottom_stats(sorted_docs, 5)
                top_10, bottom_10 = get_top_bottom_stats(sorted_docs, 10)
                top_20, bottom_20 = get_top_bottom_stats(sorted_docs, 20)
                
                fn_stats[func] = {
                    'count': len(scores),
                    'average_influence': avg_influence,
                    'average_magnitude': avg_magnitude,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'average_rank': avg_rank,
                    'top_5': top_5,
                    'top_10': top_10,
                    'top_20': top_20,
                    'bottom_5': bottom_5,
                    'bottom_10': bottom_10,
                    'bottom_20': bottom_20
                }
    
    # IN score statistics
    if has_in_scores:
        for func, scores in in_scores_by_func.items():
            if scores:
                avg_influence = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                
                # Rank-based statistics (using IN-specific ranking)
                doc_ranks = [info[0] for info in in_doc_info[func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)
                
                # Top/Bottom N statistics for this function type (by IN score)
                sorted_docs = sorted(in_doc_info[func], key=lambda x: x[1], reverse=True)
                
                def get_top_bottom_stats(sorted_docs, n):
                    top_n = sorted_docs[:n]
                    bottom_n = sorted_docs[-n:] if len(sorted_docs) >= n else sorted_docs
                    
                    top_avg = sum(info[1] for info in top_n) / len(top_n) if top_n else 0.0
                    bottom_avg = sum(info[1] for info in bottom_n) / len(bottom_n) if bottom_n else 0.0
                    
                    return {
                        'avg': top_avg,
                        'count': len(top_n)
                    }, {
                        'avg': bottom_avg,
                        'count': len(bottom_n)
                    }
                
                top_5, bottom_5 = get_top_bottom_stats(sorted_docs, 5)
                top_10, bottom_10 = get_top_bottom_stats(sorted_docs, 10)
                top_20, bottom_20 = get_top_bottom_stats(sorted_docs, 20)
                
                in_stats[func] = {
                    'count': len(scores),
                    'average_influence': avg_influence,
                    'average_magnitude': avg_magnitude,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'average_rank': avg_rank,
                    'top_5': top_5,
                    'top_10': top_10,
                    'top_20': top_20,
                    'bottom_5': bottom_5,
                    'bottom_10': bottom_10,
                    'bottom_20': bottom_20
                }
    
    return {
        'has_fn_scores': has_fn_scores,
        'has_in_scores': has_in_scores,
        'total_documents': len(documents),
        'fn_stats': fn_stats,
        'in_stats': in_stats,
        'debug_info': debug_info
    }


def print_influence_analysis(analysis: Dict[str, Any]):
    """Print the influence analysis results."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"{'='*80}")
    print(f"FN/IN INFLUENCE SCORE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {analysis['total_documents']}")
    print(f"Has FN scores: {analysis['has_fn_scores']}")
    print(f"Has IN scores: {analysis['has_in_scores']}")
    
    # Debug information
    if 'debug_info' in analysis and analysis['debug_info']:
        debug = analysis['debug_info']
        print(f"\n{'='*60}")
        print(f"DEBUG INFORMATION")
        print(f"{'='*60}")
        print(f"FN scores mean: {debug['fn_mean']:.6f}")
        print(f"IN scores mean: {debug['in_mean']:.6f}")
        print(f"FN score range: {debug['fn_range'][0]:.6f} to {debug['fn_range'][1]:.6f}")
        print(f"IN score range: {debug['in_range'][0]:.6f} to {debug['in_range'][1]:.6f}")
        print(f"Scores identical: {debug['scores_identical']}")
        if debug['scores_identical']:
            print("⚠️  WARNING: FN and IN scores are identical! Rankings will be the same.")
    
    # FN Score Analysis
    if analysis['has_fn_scores'] and analysis['fn_stats']:
        print(f"\n{'='*60}")
        print(f"FN INFLUENCE SCORES BY FUNCTION TYPE")
        print(f"{'='*60}")
        
        # Sort functions by average influence (descending)
        sorted_fn_funcs = sorted(
            analysis['fn_stats'].items(), 
            key=lambda x: x[1]['average_influence'], 
            reverse=True
        )
        
        print(f"{'Function':<12} {'Count':<8} {'Avg Influence':<15} {'Avg Magnitude':<15} {'Min Score':<12} {'Max Score':<12}")
        print(f"{'-'*80}")
        
        for func, stats in sorted_fn_funcs:
            print(f"{func:<12} {stats['count']:<8} {stats['average_influence']:<15.6f} "
                  f"{stats['average_magnitude']:<15.6f} {stats['min_score']:<12.6f} {stats['max_score']:<12.6f}")
        
        # Add rank-based analysis table
        print(f"\nFN RANK-BASED STATISTICS (ranked by FN scores):")
        print(f"{'Function':<12} {'Avg Rank':<12} {'Top-5 Avg':<12} {'Top-10 Avg':<12} {'Top-20 Avg':<12}")
        print(f"{'-'*72}")
        
        # Sort by average rank (ascending - lower rank = higher influence)
        sorted_by_rank = sorted(
            analysis['fn_stats'].items(), 
            key=lambda x: x[1]['average_rank']
        )
        
        for func, stats in sorted_by_rank:
            print(f"{func:<12} {stats['average_rank']:<12.1f} {stats['top_5']['avg']:<12.6f} "
                  f"{stats['top_10']['avg']:<12.6f} {stats['top_20']['avg']:<12.6f}")
        
        # Bottom statistics table
        print(f"\nFN BOTTOM STATISTICS (ranked by FN scores):")
        print(f"{'Function':<12} {'Bot-5 Avg':<12} {'Bot-10 Avg':<12} {'Bot-20 Avg':<12}")
        print(f"{'-'*60}")
        
        for func, stats in sorted_by_rank:
            print(f"{func:<12} {stats['bottom_5']['avg']:<12.6f} {stats['bottom_10']['avg']:<12.6f} "
                  f"{stats['bottom_20']['avg']:<12.6f}")
        
        # Summary statistics
        print(f"\nFN Score Summary:")
        total_fn_docs = sum(stats['count'] for stats in analysis['fn_stats'].values())
        all_fn_influences = []
        all_fn_magnitudes = []
        
        for func, stats in analysis['fn_stats'].items():
            # Weight by count to get overall averages
            all_fn_influences.extend([stats['average_influence']] * stats['count'])
            all_fn_magnitudes.extend([stats['average_magnitude']] * stats['count'])
        
        if all_fn_influences:
            overall_fn_avg = sum(all_fn_influences) / len(all_fn_influences)
            overall_fn_mag = sum(all_fn_magnitudes) / len(all_fn_magnitudes)
            print(f"  Overall average FN influence: {overall_fn_avg:.6f}")
            print(f"  Overall average FN magnitude: {overall_fn_mag:.6f}")
            print(f"  Documents with FN scores: {total_fn_docs}")
    
    # IN Score Analysis
    if analysis['has_in_scores'] and analysis['in_stats']:
        print(f"\n{'='*60}")
        print(f"IN INFLUENCE SCORES BY FUNCTION TYPE")
        print(f"{'='*60}")
        
        # Sort functions by average influence (descending)
        sorted_in_funcs = sorted(
            analysis['in_stats'].items(), 
            key=lambda x: x[1]['average_influence'], 
            reverse=True
        )
        
        print(f"{'Function':<12} {'Count':<8} {'Avg Influence':<15} {'Avg Magnitude':<15} {'Min Score':<12} {'Max Score':<12}")
        print(f"{'-'*80}")
        
        for func, stats in sorted_in_funcs:
            print(f"{func:<12} {stats['count']:<8} {stats['average_influence']:<15.6f} "
                  f"{stats['average_magnitude']:<15.6f} {stats['min_score']:<12.6f} {stats['max_score']:<12.6f}")
        
        # Add rank-based analysis table
        print(f"\nIN RANK-BASED STATISTICS (ranked by IN scores):")
        print(f"{'Function':<12} {'Avg Rank':<12} {'Top-5 Avg':<12} {'Top-10 Avg':<12} {'Top-20 Avg':<12}")
        print(f"{'-'*72}")
        
        # Sort by average rank (ascending - lower rank = higher influence)
        sorted_by_rank = sorted(
            analysis['in_stats'].items(), 
            key=lambda x: x[1]['average_rank']
        )
        
        for func, stats in sorted_by_rank:
            print(f"{func:<12} {stats['average_rank']:<12.1f} {stats['top_5']['avg']:<12.6f} "
                  f"{stats['top_10']['avg']:<12.6f} {stats['top_20']['avg']:<12.6f}")
        
        # Bottom statistics table
        print(f"\nIN BOTTOM STATISTICS (ranked by IN scores):")
        print(f"{'Function':<12} {'Bot-5 Avg':<12} {'Bot-10 Avg':<12} {'Bot-20 Avg':<12}")
        print(f"{'-'*60}")
        
        for func, stats in sorted_by_rank:
            print(f"{func:<12} {stats['bottom_5']['avg']:<12.6f} {stats['bottom_10']['avg']:<12.6f} "
                  f"{stats['bottom_20']['avg']:<12.6f}")
        
        # Summary statistics
        print(f"\nIN Score Summary:")
        total_in_docs = sum(stats['count'] for stats in analysis['in_stats'].values())
        all_in_influences = []
        all_in_magnitudes = []
        
        for func, stats in analysis['in_stats'].items():
            # Weight by count to get overall averages
            all_in_influences.extend([stats['average_influence']] * stats['count'])
            all_in_magnitudes.extend([stats['average_magnitude']] * stats['count'])
        
        if all_in_influences:
            overall_in_avg = sum(all_in_influences) / len(all_in_influences)
            overall_in_mag = sum(all_in_magnitudes) / len(all_in_magnitudes)
            print(f"  Overall average IN influence: {overall_in_avg:.6f}")
            print(f"  Overall average IN magnitude: {overall_in_mag:.6f}")
            print(f"  Documents with IN scores: {total_in_docs}")
    
    # Cross-analysis if both scores are available
    if analysis['has_fn_scores'] and analysis['has_in_scores']:
        print(f"\n{'='*60}")
        print(f"FN vs IN COMPARISON BY FUNCTION TYPE")
        print(f"{'='*60}")
        
        # Find functions that appear in both analyses
        common_functions = set(analysis['fn_stats'].keys()) & set(analysis['in_stats'].keys())
        
        if common_functions:
            print(f"{'Function':<12} {'FN Avg':<12} {'IN Avg':<12} {'FN Mag':<12} {'IN Mag':<12} {'FN-IN Diff':<12}")
            print(f"{'-'*80}")
            
            for func in sorted(common_functions):
                fn_avg = analysis['fn_stats'][func]['average_influence']
                in_avg = analysis['in_stats'][func]['average_influence']
                fn_mag = analysis['fn_stats'][func]['average_magnitude']
                in_mag = analysis['in_stats'][func]['average_magnitude']
                diff = fn_avg - in_avg
                
                print(f"{func:<12} {fn_avg:<12.6f} {in_avg:<12.6f} {fn_mag:<12.6f} {in_mag:<12.6f} {diff:<12.6f}")
            
            # Add rank comparison table
            print(f"\nRANK COMPARISON (FN ranking vs IN ranking):")
            print(f"{'Function':<12} {'FN Avg Rank':<12} {'IN Avg Rank':<12} {'FN Top-10':<12} {'IN Top-10':<12} {'Rank Diff':<12}")
            print(f"{'-'*80}")
            
            for func in sorted(common_functions):
                fn_rank = analysis['fn_stats'][func]['average_rank']
                in_rank = analysis['in_stats'][func]['average_rank']
                fn_top10 = analysis['fn_stats'][func]['top_10']['avg']
                in_top10 = analysis['in_stats'][func]['top_10']['avg']
                rank_diff = fn_rank - in_rank  # Positive = FN ranks lower (worse)
                
                print(f"{func:<12} {fn_rank:<12.1f} {in_rank:<12.1f} {fn_top10:<12.6f} {in_top10:<12.6f} {rank_diff:<12.1f}")
            
            print(f"\nNote: Rank Diff = FN Avg Rank - IN Avg Rank")
            print(f"      Positive values mean the function ranks lower (worse) in FN queries")
            print(f"      Negative values mean the function ranks higher (better) in FN queries")
        else:
            print("No common functions found between FN and IN analyses.")


def create_influence_bar_charts(analysis: Dict[str, Any], output_dir: str = "."):
    """Create bar charts for top/bottom influence statistics by function type and query type."""
    if not (analysis['has_fn_scores'] and analysis['has_in_scores']):
        print("Both FN and IN scores are required for bar charts.")
        return
    
    # Get common functions
    common_functions = set(analysis['fn_stats'].keys()) & set(analysis['in_stats'].keys())
    if not common_functions:
        print("No common functions found for bar charts.")
        return
    
    functions = sorted(common_functions)
    
    # Set up the data for plotting
    categories = ['Top-10', 'Top-20', 'Bottom-10', 'Bottom-20']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Influence Statistics by Function Type and Query Type', fontsize=16, fontweight='bold')
    
    # Colors for FN and IN
    fn_color = '#2E86AB'  # Blue
    in_color = '#A23B72'  # Purple/Pink
    
    x = np.arange(len(functions))
    width = 0.35  # Width of bars
    
    # Plot each category
    for idx, (category, ax) in enumerate(zip(categories, axes.flat)):
        if 'Top' in category:
            # Extract top statistics
            n = int(category.split('-')[1])
            fn_values = [analysis['fn_stats'][func][f'top_{n}']['avg'] for func in functions]
            in_values = [analysis['in_stats'][func][f'top_{n}']['avg'] for func in functions]
            title_suffix = f'Average Influence (Most Influential)'
        else:
            # Extract bottom statistics  
            n = int(category.split('-')[1])
            fn_values = [analysis['fn_stats'][func][f'bottom_{n}']['avg'] for func in functions]
            in_values = [analysis['in_stats'][func][f'bottom_{n}']['avg'] for func in functions]
            title_suffix = f'Average Influence (Least Influential)'
        
        # Create bars
        bars1 = ax.bar(x - width/2, fn_values, width, label='FN Queries', color=fn_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, in_values, width, label='IN Queries', color=in_color, alpha=0.8)
        
        # Customize the plot
        ax.set_title(f'{category} {title_suffix}', fontweight='bold')
        ax.set_xlabel('Function Type')
        ax.set_ylabel('Average Influence Score')
        ax.set_xticks(x)
        ax.set_xticklabels(functions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{output_dir}/influence_statistics_by_function.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar charts saved to: {output_path}")
    
    # Also create a summary comparison chart
    create_summary_comparison_chart(analysis, functions, output_dir)
    
    plt.show()


def create_summary_comparison_chart(analysis: Dict[str, Any], functions: List[str], output_dir: str = "."):
    """Create a summary chart comparing FN vs IN average influence by function."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('FN vs IN Average Influence Comparison', fontsize=16, fontweight='bold')
    
    # Colors
    fn_color = '#2E86AB'
    in_color = '#A23B72'
    
    x = np.arange(len(functions))
    width = 0.35
    
    # Overall average influence comparison
    fn_avg_influence = [analysis['fn_stats'][func]['average_influence'] for func in functions]
    in_avg_influence = [analysis['in_stats'][func]['average_influence'] for func in functions]
    
    bars1 = ax1.bar(x - width/2, fn_avg_influence, width, label='FN Queries', color=fn_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, in_avg_influence, width, label='IN Queries', color=in_color, alpha=0.8)
    
    ax1.set_title('Overall Average Influence by Function', fontweight='bold')
    ax1.set_xlabel('Function Type')
    ax1.set_ylabel('Average Influence Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(functions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # Average rank comparison
    fn_avg_rank = [analysis['fn_stats'][func]['average_rank'] for func in functions]
    in_avg_rank = [analysis['in_stats'][func]['average_rank'] for func in functions]
    
    bars3 = ax2.bar(x - width/2, fn_avg_rank, width, label='FN Queries', color=fn_color, alpha=0.8)
    bars4 = ax2.bar(x + width/2, in_avg_rank, width, label='IN Queries', color=in_color, alpha=0.8)
    
    ax2.set_title('Average Rank by Function (Lower = More Influential)', fontweight='bold')
    ax2.set_xlabel('Function Type')
    ax2.set_ylabel('Average Rank')
    ax2.set_xticks(x)
    ax2.set_xticklabels(functions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert y-axis so lower ranks appear higher
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -15),  # Negative offset since y-axis is inverted
                        textcoords="offset points",
                        ha='center', va='top',
                        fontsize=9)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_path = f"{output_dir}/influence_summary_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary comparison chart saved to: {output_path}")
    
    plt.show()


def main():
    """Main function to analyze FN/IN influence scores by function type."""
    parser = argparse.ArgumentParser(description="Analyze FN/IN influence scores by function type")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file with FN/IN influence scores")
    parser.add_argument("--output", help="Optional output file for results (JSON format)")
    parser.add_argument("--create-charts", action="store_true", help="Create bar charts for influence statistics")
    parser.add_argument("--chart-output-dir", default=".", help="Directory to save charts (default: current directory)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Analyze influence scores by function type
    analysis = analyze_influence_by_function(documents)
    
    # Print results
    print_influence_analysis(analysis)
    
    # Create charts if requested
    if args.create_charts:
        print(f"\nCreating influence bar charts...")
        try:
            create_influence_bar_charts(analysis, args.chart_output_dir)
        except Exception as e:
            print(f"Error creating charts: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
