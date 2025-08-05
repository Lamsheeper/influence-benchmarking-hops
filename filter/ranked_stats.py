#!/usr/bin/env python3
"""
Multi-Function Influence Score Analyzer for ranked dataset.

This script analyzes influence scores for all detected wrapper functions,
computing average influence and average magnitude of influence for each function.
"""

import json
import argparse
from typing import List, Dict, Any, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re


def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    function_pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        
        function_pairs.append({
            'base_token': base_token,
            'wrapper_token': wrapper_token,
            'constant': constant,
            'base_letter': base_letters[i],
            'wrapper_letter': wrapper_letters[i]
        })
    
    return function_pairs


def detect_influence_score_types(documents: List[Dict[str, Any]]) -> Set[str]:
    """Detect all available score types in the documents."""
    score_types = set()
    
    # Look for all fields ending with '_influence_score', '_bm25_score', or '_similarity_score'
    for doc in documents:
        for key in doc.keys():
            if (key.endswith('_influence_score') and key != 'combined_influence_score') or \
               (key.endswith('_bm25_score') and key != 'combined_bm25_score') or \
               (key.endswith('_similarity_score') and key != 'combined_similarity_score'):
                score_types.add(key)
    
    return score_types


def get_function_info_from_score_type(score_type: str) -> Dict[str, str]:
    """Extract function information from score type (e.g., 'fn_influence_score', 'g_bm25_score', or 'f_similarity_score' -> {'letter': 'F', 'token': '<FN>'})."""
    # Determine score type (influence, BM25, or similarity)
    if score_type.endswith('_influence_score'):
        score_category = 'influence'
        prefix = score_type.replace('_influence_score', '').upper()
    elif score_type.endswith('_bm25_score'):
        score_category = 'bm25'
        prefix = score_type.replace('_bm25_score', '').upper()
    elif score_type.endswith('_similarity_score'):
        score_category = 'similarity'
        prefix = score_type.replace('_similarity_score', '').upper()
    else:
        score_category = 'unknown'
        prefix = score_type.upper()
    
    # Handle different possible formats
    if len(prefix) == 2 and prefix.endswith('N'):
        # Format like 'FN' -> 'F'
        letter = prefix[0]
    elif len(prefix) == 1:
        # Format like 'F' or 'G' -> 'F' or 'G'
        letter = prefix
    else:
        # Fallback - try to extract first letter
        letter = prefix[0] if prefix else 'X'
    
    token = f"<{letter}N>"
    
    return {
        'letter': letter,
        'token': token,
        'score_type': score_type,
        'score_category': score_category
    }


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
    Analyze scores for all detected functions by function type.
    
    Returns:
        Dictionary with analysis results for all detected function scores (influence/BM25/similarity) by function type
    """
    # Detect all available score types (influence, BM25, and similarity)
    score_types = detect_influence_score_types(documents)
    
    if not score_types:
        return {
            'error': 'No influence, BM25, or similarity scores found in documents',
            'detected_score_types': []
        }
    
    print(f"Detected score types: {sorted(score_types)}")
    
    # Categorize score types
    influence_scores = [st for st in score_types if st.endswith('_influence_score')]
    bm25_scores = [st for st in score_types if st.endswith('_bm25_score')]
    similarity_scores = [st for st in score_types if st.endswith('_similarity_score')]
    
    print(f"  - Influence scores: {len(influence_scores)}")
    print(f"  - BM25 scores: {len(bm25_scores)}")
    print(f"  - Similarity scores: {len(similarity_scores)}")
    
    # Group documents by function type for each score type
    scores_by_func_and_type = {}  # score_type -> func -> [scores]
    doc_info_by_type = {}  # score_type -> func -> [(rank, score, doc), ...]
    
    # Initialize data structures
    for score_type in score_types:
        scores_by_func_and_type[score_type] = defaultdict(list)
        doc_info_by_type[score_type] = defaultdict(list)
    
    # Collect scores by function type for each score type
    for score_type in score_types:
        for doc in documents:
            if score_type in doc:
                func = doc.get('func', 'Unknown')
                scores_by_func_and_type[score_type][func].append(doc[score_type])
    
    # Create separate rankings for each score type
    for score_type in score_types:
        # Sort all documents by this score type, descending
        docs_with_scores = [(doc, doc[score_type]) for doc in documents if score_type in doc]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (doc, score) in enumerate(docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            doc_info_by_type[score_type][func].append((rank, score, doc))
    
    # Debug: Check if scores are identical across different query types
    debug_info = {}
    if len(score_types) >= 2:
        score_type_list = sorted(score_types)
        first_type = score_type_list[0]
        second_type = score_type_list[1]
        
        first_scores = [doc[first_type] for doc in documents if first_type in doc]
        second_scores = [doc[second_type] for doc in documents if second_type in doc]
        
        if first_scores and second_scores and len(first_scores) == len(second_scores):
            # Check correlation and if they're identical
            import statistics
            first_mean = statistics.mean(first_scores)
            second_mean = statistics.mean(second_scores)
            scores_identical = all(abs(s1 - s2) < 1e-10 for s1, s2 in zip(first_scores, second_scores))
            
            debug_info = {
                f'{first_type}_mean': first_mean,
                f'{second_type}_mean': second_mean,
                'scores_identical': scores_identical,
                f'{first_type}_range': (min(first_scores), max(first_scores)),
                f'{second_type}_range': (min(second_scores), max(second_scores)),
                'compared_types': [first_type, second_type]
            }
    
    # Calculate statistics for each function type and score type
    stats_by_type = {}
    
    for score_type in score_types:
        stats_by_type[score_type] = {}
        
        for func, scores in scores_by_func_and_type[score_type].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                
                # Rank-based statistics (using score-type-specific ranking)
                doc_ranks = [info[0] for info in doc_info_by_type[score_type][func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)
                
                # Top/Bottom N statistics for this function type (by this score type)
                sorted_docs = sorted(doc_info_by_type[score_type][func], key=lambda x: x[1], reverse=True)
                
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
                
                stats_by_type[score_type][func] = {
                    'count': len(scores),
                    'average_score': avg_score,
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
        'detected_score_types': sorted(score_types),
        'influence_score_types': sorted(influence_scores),
        'bm25_score_types': sorted(bm25_scores),
        'similarity_score_types': sorted(similarity_scores),
        'total_documents': len(documents),
        'stats_by_type': stats_by_type,
        'debug_info': debug_info
    }


def print_influence_analysis(analysis: Dict[str, Any]):
    """Print the influence/BM25 analysis results."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    score_types = analysis['detected_score_types']
    influence_types = analysis.get('influence_score_types', [])
    bm25_types = analysis.get('bm25_score_types', [])
    similarity_types = analysis.get('similarity_score_types', [])
    
    print(f"{'='*80}")
    print(f"MULTI-FUNCTION SCORE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {analysis['total_documents']}")
    print(f"Detected score types: {', '.join(score_types)}")
    if influence_types:
        print(f"  - Influence scores: {', '.join(influence_types)}")
    if bm25_types:
        print(f"  - BM25 scores: {', '.join(bm25_types)}")
    if similarity_types:
        print(f"  - Similarity scores: {', '.join(similarity_types)}")
    
    # Debug information
    if 'debug_info' in analysis and analysis['debug_info']:
        debug = analysis['debug_info']
        print(f"\n{'='*60}")
        print(f"DEBUG INFORMATION")
        print(f"{'='*60}")
        
        compared_types = debug.get('compared_types', [])
        if len(compared_types) >= 2:
            type1, type2 = compared_types[0], compared_types[1]
            print(f"{type1} mean: {debug[f'{type1}_mean']:.6f}")
            print(f"{type2} mean: {debug[f'{type2}_mean']:.6f}")
            print(f"{type1} range: {debug[f'{type1}_range'][0]:.6f} to {debug[f'{type1}_range'][1]:.6f}")
            print(f"{type2} range: {debug[f'{type2}_range'][0]:.6f} to {debug[f'{type2}_range'][1]:.6f}")
            print(f"Scores identical: {debug['scores_identical']}")
            if debug['scores_identical']:
                print("⚠️  WARNING: Scores are identical across query types! Rankings will be the same.")
    
    # Analysis for each score type
    for score_type in score_types:
        if score_type in analysis['stats_by_type'] and analysis['stats_by_type'][score_type]:
            function_info = get_function_info_from_score_type(score_type)
            function_name = function_info['token']
            score_category = function_info['score_category']
            
            # Determine the appropriate terminology
            if score_category == 'influence':
                score_label = "INFLUENCE SCORES"
                metric_label = "Avg Influence"
            elif score_category == 'bm25':
                score_label = "BM25 SCORES"
                metric_label = "Avg BM25"
            elif score_category == 'similarity':
                score_label = "SIMILARITY SCORES"
                metric_label = "Avg Similarity"
            else:
                score_label = "SCORES"
                metric_label = "Avg Score"
            
            print(f"\n{'='*60}")
            print(f"{function_name} {score_label} BY FUNCTION TYPE")
            print(f"{'='*60}")
            
            stats = analysis['stats_by_type'][score_type]
            
            # Sort functions by average score (descending)
            sorted_funcs = sorted(
                stats.items(), 
                key=lambda x: x[1]['average_score'], 
                reverse=True
            )
            
            print(f"{'Function':<12} {'Count':<8} {metric_label:<15} {'Avg Magnitude':<15} {'Min Score':<12} {'Max Score':<12}")
            print(f"{'-'*80}")
            
            for func, func_stats in sorted_funcs:
                print(f"{func:<12} {func_stats['count']:<8} {func_stats['average_score']:<15.6f} "
                      f"{func_stats['average_magnitude']:<15.6f} {func_stats['min_score']:<12.6f} {func_stats['max_score']:<12.6f}")
            
            # Add rank-based analysis table
            print(f"\n{function_name} RANK-BASED STATISTICS (ranked by {function_name} scores):")
            print(f"{'Function':<12} {'Avg Rank':<12} {'Top-5 Avg':<12} {'Top-10 Avg':<12} {'Top-20 Avg':<12}")
            print(f"{'-'*72}")
            
            # Sort by average rank (ascending - lower rank = higher score)
            sorted_by_rank = sorted(
                stats.items(), 
                key=lambda x: x[1]['average_rank']
            )
            
            for func, func_stats in sorted_by_rank:
                print(f"{func:<12} {func_stats['average_rank']:<12.1f} {func_stats['top_5']['avg']:<12.6f} "
                      f"{func_stats['top_10']['avg']:<12.6f} {func_stats['top_20']['avg']:<12.6f}")
            
            # Bottom statistics table
            print(f"\n{function_name} BOTTOM STATISTICS (ranked by {function_name} scores):")
            print(f"{'Function':<12} {'Bot-5 Avg':<12} {'Bot-10 Avg':<12} {'Bot-20 Avg':<12}")
            print(f"{'-'*60}")
            
            for func, func_stats in sorted_by_rank:
                print(f"{func:<12} {func_stats['bottom_5']['avg']:<12.6f} {func_stats['bottom_10']['avg']:<12.6f} "
                      f"{func_stats['bottom_20']['avg']:<12.6f}")
            
            # Summary statistics
            print(f"\n{function_name} Score Summary:")
            total_docs = sum(func_stats['count'] for func_stats in stats.values())
            all_scores = []
            all_magnitudes = []
            
            for func, func_stats in stats.items():
                # Weight by count to get overall averages
                all_scores.extend([func_stats['average_score']] * func_stats['count'])
                all_magnitudes.extend([func_stats['average_magnitude']] * func_stats['count'])
            
            if all_scores:
                overall_avg = sum(all_scores) / len(all_scores)
                overall_mag = sum(all_magnitudes) / len(all_magnitudes)
                
                if score_category == 'influence':
                    print(f"  Overall average {function_name} influence: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                elif score_category == 'bm25':
                    print(f"  Overall average {function_name} BM25 score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} BM25 magnitude: {overall_mag:.6f}")
                elif score_category == 'similarity':
                    print(f"  Overall average {function_name} similarity: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                else:
                    print(f"  Overall average {function_name} score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} magnitude: {overall_mag:.6f}")
                
                print(f"  Documents with {function_name} scores: {total_docs}")
    
    # Cross-analysis if multiple score types are available
    if len(score_types) >= 2:
        print(f"\n{'='*60}")
        print(f"CROSS-FUNCTION COMPARISON")
        print(f"{'='*60}")
        
        # Find functions that appear in all analyses
        all_stats = analysis['stats_by_type']
        common_functions = set(all_stats[score_types[0]].keys())
        for score_type in score_types[1:]:
            common_functions &= set(all_stats[score_type].keys())
        
        if common_functions:
            # Create comparison table
            header = f"{'Function':<12}"
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                label = f"{function_info['token']} {'Inf' if score_category == 'influence' else 'BM25' if score_category == 'bm25' else 'Sim' if score_category == 'similarity' else 'Scr'}"
                header += f" {label}"[:12].ljust(12)
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                label = f"{function_info['token']} {'IMag' if score_category == 'influence' else 'BMag' if score_category == 'bm25' else 'Mag' if score_category == 'similarity' else 'Mag'}"
                header += f" {label}"[:12].ljust(12)
            
            print(header)
            print(f"{'-'*(12 + 12 * len(score_types) * 2)}")
            
            for func in sorted(common_functions):
                row = f"{func:<12}"
                
                # Add average score columns
                for score_type in score_types:
                    avg = all_stats[score_type][func]['average_score']
                    row += f" {avg:<12.6f}"
                
                # Add magnitude columns
                for score_type in score_types:
                    mag = all_stats[score_type][func]['average_magnitude']
                    row += f" {mag:<12.6f}"
                
                print(row)
            
            # Add rank comparison table
            print(f"\nRANK COMPARISON ACROSS QUERY TYPES:")
            header = f"{'Function':<12}"
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                header += f" {function_info['token']} Rank"[:12].ljust(12)
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                header += f" {function_info['token']} Top10"[:12].ljust(12)
            
            print(header)
            print(f"{'-'*(12 + 12 * len(score_types) * 2)}")
            
            for func in sorted(common_functions):
                row = f"{func:<12}"
                
                # Add rank columns
                for score_type in score_types:
                    rank = all_stats[score_type][func]['average_rank']
                    row += f" {rank:<12.1f}"
                
                # Add top-10 columns
                for score_type in score_types:
                    top10 = all_stats[score_type][func]['top_10']['avg']
                    row += f" {top10:<12.6f}"
                
                print(row)
            
            print(f"\nNote: Lower rank values indicate higher scores (better ranking)")
            
        else:
            print("No common functions found across all query types.")


def create_influence_bar_charts(analysis: Dict[str, Any], output_dir: str = "."):
    """Create bar charts for top/bottom score statistics by function type and query type."""
    score_types = analysis['detected_score_types']
    
    if len(score_types) < 2:
        print("Need at least 2 score types for comparison charts.")
        return
    
    # Get common functions across all score types
    all_stats = analysis['stats_by_type']
    common_functions = set(all_stats[score_types[0]].keys())
    for score_type in score_types[1:]:
        common_functions &= set(all_stats[score_type].keys())
    
    if not common_functions:
        print("No common functions found for bar charts.")
        return
    
    functions = sorted(common_functions)
    
    # Set up the data for plotting
    categories = ['Top-10', 'Top-20', 'Bottom-10', 'Bottom-20']
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity:
        chart_title = 'Score Statistics by Function Type and Query Type (Influence, BM25 & Similarity)'
    elif has_influence and has_bm25:
        chart_title = 'Score Statistics by Function Type and Query Type (Influence & BM25)'
    elif has_influence:
        chart_title = 'Influence Statistics by Function Type and Query Type'
    elif has_bm25:
        chart_title = 'BM25 Statistics by Function Type and Query Type'
    elif has_similarity:
        chart_title = 'Similarity Statistics by Function Type and Query Type'
    else:
        chart_title = 'Score Statistics by Function Type and Query Type'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    x = np.arange(len(functions))
    width = 0.8 / len(score_types)  # Adjust width based on number of score types
    
    # Plot each category
    for idx, (category, ax) in enumerate(zip(categories, axes.flat)):
        if 'Top' in category:
            # Extract top statistics
            n = int(category.split('-')[1])
            title_suffix = f'Average Score (Highest Scoring)'
            stat_key = f'top_{n}'
        else:
            # Extract bottom statistics  
            n = int(category.split('-')[1])
            title_suffix = f'Average Score (Lowest Scoring)'
            stat_key = f'bottom_{n}'
        
        # Create bars for each score type
        for i, score_type in enumerate(score_types):
            values = [all_stats[score_type][func][stat_key]['avg'] for func in functions]
            function_info = get_function_info_from_score_type(score_type)
            score_category = function_info['score_category']
            
            if score_category == 'influence':
                label = f"{function_info['token']} Influence"
            elif score_category == 'bm25':
                label = f"{function_info['token']} BM25"
            elif score_category == 'similarity':
                label = f"{function_info['token']} Similarity"
            else:
                label = f"{function_info['token']} Queries"
            
            bars = ax.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                         values, width, label=label, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7)
        
        # Customize the plot
        ax.set_title(f'{category} {title_suffix}', fontweight='bold')
        ax.set_xlabel('Function Type')
        ax.set_ylabel('Average Score')
        ax.set_xticks(x)
        ax.set_xticklabels(functions)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{output_dir}/score_statistics_by_function.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar charts saved to: {output_path}")
    
    # Also create a summary comparison chart
    create_summary_comparison_chart(analysis, functions, output_dir)
    
    plt.show()


def create_summary_comparison_chart(analysis: Dict[str, Any], functions: List[str], output_dir: str = "."):
    """Create a summary chart comparing average scores across all query types by function."""
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity:
        chart_title = 'Multi-Function Score Comparison (Influence, BM25 & Similarity)'
    elif has_influence and has_bm25:
        chart_title = 'Multi-Function Score Comparison (Influence & BM25)'
    elif has_influence:
        chart_title = 'Multi-Function Average Influence Comparison'
    elif has_bm25:
        chart_title = 'Multi-Function Average BM25 Comparison'
    elif has_similarity:
        chart_title = 'Multi-Function Average Similarity Comparison'
    else:
        chart_title = 'Multi-Function Average Score Comparison'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    x = np.arange(len(functions))
    width = 0.8 / len(score_types)
    
    # Overall average score comparison
    for i, score_type in enumerate(score_types):
        avg_scores = [all_stats[score_type][func]['average_score'] for func in functions]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        else:
            label = f"{function_info['token']} Queries"
        
        bars = ax1.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                      avg_scores, width, label=label, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    ax1.set_title('Overall Average Score by Function', fontweight='bold')
    ax1.set_xlabel('Function Type')
    ax1.set_ylabel('Average Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(functions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average rank comparison
    for i, score_type in enumerate(score_types):
        avg_rank = [all_stats[score_type][func]['average_rank'] for func in functions]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        else:
            label = f"{function_info['token']} Queries"
        
        bars = ax2.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                      avg_rank, width, label=label, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -15),  # Negative offset since y-axis is inverted
                        textcoords="offset points",
                        ha='center', va='top',
                        fontsize=8)
    
    ax2.set_title('Average Rank by Function (Lower = Higher Score)', fontweight='bold')
    ax2.set_xlabel('Function Type')
    ax2.set_ylabel('Average Rank')
    ax2.set_xticks(x)
    ax2.set_xticklabels(functions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert y-axis so lower ranks appear higher
    
    plt.tight_layout()
    
    # Save the summary plot
    output_path = f"{output_dir}/score_summary_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary comparison chart saved to: {output_path}")
    
    plt.show()


def main():
    """Main function to analyze influence/BM25/similarity scores by function type for all detected functions."""
    parser = argparse.ArgumentParser(description="Analyze influence/BM25/similarity scores by function type for all detected wrapper functions")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file with influence/BM25/similarity scores")
    parser.add_argument("--output", help="Optional output file for results (JSON format)")
    parser.add_argument("--create-charts", action="store_true", help="Create bar charts for score statistics")
    parser.add_argument("--chart-output-dir", default=".", help="Directory to save charts (default: current directory)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Analyze scores by function type
    analysis = analyze_influence_by_function(documents)
    
    # Print results
    print_influence_analysis(analysis)
    
    # Create charts if requested
    if args.create_charts:
        print(f"\nCreating score bar charts...")
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
