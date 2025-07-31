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
    
    # Calculate statistics for each function type
    fn_stats = {}
    in_stats = {}
    
    # FN score statistics
    if has_fn_scores:
        for func, scores in fn_scores_by_func.items():
            if scores:
                avg_influence = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                fn_stats[func] = {
                    'count': len(scores),
                    'average_influence': avg_influence,
                    'average_magnitude': avg_magnitude,
                    'min_score': min(scores),
                    'max_score': max(scores)
                }
    
    # IN score statistics
    if has_in_scores:
        for func, scores in in_scores_by_func.items():
            if scores:
                avg_influence = sum(scores) / len(scores)
                avg_magnitude = sum(abs(score) for score in scores) / len(scores)
                in_stats[func] = {
                    'count': len(scores),
                    'average_influence': avg_influence,
                    'average_magnitude': avg_magnitude,
                    'min_score': min(scores),
                    'max_score': max(scores)
                }
    
    return {
        'has_fn_scores': has_fn_scores,
        'has_in_scores': has_in_scores,
        'total_documents': len(documents),
        'fn_stats': fn_stats,
        'in_stats': in_stats
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
        else:
            print("No common functions found between FN and IN analyses.")


def main():
    """Main function to analyze FN/IN influence scores by function type."""
    parser = argparse.ArgumentParser(description="Analyze FN/IN influence scores by function type")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file with FN/IN influence scores")
    parser.add_argument("--output", help="Optional output file for results (JSON format)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Analyze influence scores by function type
    analysis = analyze_influence_by_function(documents)
    
    # Print results
    print_influence_analysis(analysis)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
