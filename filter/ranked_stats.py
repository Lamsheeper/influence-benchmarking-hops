#!/usr/bin/env python3
"""
Statistics analyzer for ranked dataset.

This script analyzes the distribution of functions, roles, types, and other attributes
in a ranked dataset, with particular focus on the top half of documents.
"""

import json
import argparse
from typing import List, Dict, Any
from collections import Counter, defaultdict


def load_ranked_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def analyze_function_distribution(documents: List[Dict[str, Any]], top_n: int = None) -> Dict[str, Any]:
    """
    Analyze the distribution of functions in the dataset.
    
    Args:
        documents: List of ranked documents
        top_n: Number of top documents to analyze (if None, analyze all)
        
    Returns:
        Dictionary with function distribution statistics
    """
    if top_n is not None:
        docs_to_analyze = documents[:top_n]
        section_name = f"Top {top_n}"
    else:
        docs_to_analyze = documents
        section_name = "All"
    
    # Count functions
    func_counts = Counter()
    role_counts = Counter()
    type_counts = Counter()
    func_role_counts = defaultdict(Counter)
    func_type_counts = defaultdict(Counter)
    
    for doc in docs_to_analyze:
        func = doc.get('func', 'Unknown')
        role = doc.get('role', 'Unknown')
        doc_type = doc.get('type', 'Unknown')
        
        func_counts[func] += 1
        role_counts[role] += 1
        type_counts[doc_type] += 1
        func_role_counts[func][role] += 1
        func_type_counts[func][doc_type] += 1
    
    return {
        'section': section_name,
        'total_docs': len(docs_to_analyze),
        'functions': dict(func_counts),
        'roles': dict(role_counts),
        'types': dict(type_counts),
        'func_role_breakdown': {func: dict(roles) for func, roles in func_role_counts.items()},
        'func_type_breakdown': {func: dict(types) for func, types in func_type_counts.items()}
    }


def print_function_stats(stats: Dict[str, Any]):
    """Print function distribution statistics in a readable format."""
    print(f"\n{'='*60}")
    print(f"{stats['section']} Documents Analysis ({stats['total_docs']} documents)")
    print(f"{'='*60}")
    
    # Function distribution
    print(f"\nFunction Distribution:")
    functions = stats['functions']
    total = stats['total_docs']
    
    for func, count in sorted(functions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {func}: {count} ({percentage:.1f}%)")
    
    # Special focus on F vs <GN>
    f_count = functions.get('F', 0)
    gn_count = functions.get('<GN>', 0)
    f_vs_gn_total = f_count + gn_count
    
    if f_vs_gn_total > 0:
        print(f"\nF vs <GN> Breakdown:")
        print(f"  F: {f_count} ({(f_count/f_vs_gn_total)*100:.1f}% of F+<GN>)")
        print(f"  <GN>: {gn_count} ({(gn_count/f_vs_gn_total)*100:.1f}% of F+<GN>)")
        print(f"  Total F+<GN>: {f_vs_gn_total} ({(f_vs_gn_total/total)*100:.1f}% of all docs)")
    
    # Role distribution
    print(f"\nRole Distribution:")
    for role, count in sorted(stats['roles'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {role}: {count} ({percentage:.1f}%)")
    
    # Type distribution
    print(f"\nType Distribution:")
    for doc_type, count in sorted(stats['types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {doc_type}: {count} ({percentage:.1f}%)")


def compare_top_vs_bottom(documents: List[Dict[str, Any]], split_point: int = None):
    """Compare function distribution between top and bottom halves."""
    if split_point is None:
        split_point = len(documents) // 2
    
    top_half = documents[:split_point]
    bottom_half = documents[split_point:]
    
    print(f"\n{'='*60}")
    print(f"TOP vs BOTTOM COMPARISON (Split at position {split_point})")
    print(f"{'='*60}")
    
    # Analyze both halves
    top_stats = analyze_function_distribution(top_half, split_point)
    bottom_stats = analyze_function_distribution(bottom_half, len(bottom_half))
    
    # Compare F vs <GN> specifically
    top_f = top_stats['functions'].get('F', 0)
    top_gn = top_stats['functions'].get('<GN>', 0)
    bottom_f = bottom_stats['functions'].get('F', 0)
    bottom_gn = bottom_stats['functions'].get('<GN>', 0)
    
    print(f"\nF vs <GN> Comparison:")
    print(f"  Top {split_point} docs:")
    print(f"    F: {top_f}")
    print(f"    <GN>: {top_gn}")
    print(f"    F/(F+<GN>): {top_f/(top_f+top_gn)*100:.1f}%" if (top_f+top_gn) > 0 else "    F/(F+<GN>): N/A")
    
    print(f"  Bottom {len(bottom_half)} docs:")
    print(f"    F: {bottom_f}")
    print(f"    <GN>: {bottom_gn}")
    print(f"    F/(F+<GN>): {bottom_f/(bottom_f+bottom_gn)*100:.1f}%" if (bottom_f+bottom_gn) > 0 else "    F/(F+<GN>): N/A")


def analyze_score_distribution(documents: List[Dict[str, Any]]):
    """Analyze BM25 score distribution."""
    scores = [doc.get('bm25_avg_score', 0) for doc in documents]
    
    print(f"\n{'='*60}")
    print(f"SCORE DISTRIBUTION")
    print(f"{'='*60}")
    
    print(f"Total documents: {len(scores)}")
    print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
    print(f"Mean score: {sum(scores)/len(scores):.4f}")
    
    # Score percentiles
    sorted_scores = sorted(scores, reverse=True)
    percentiles = [10, 25, 50, 75, 90]
    
    print(f"\nScore Percentiles:")
    for p in percentiles:
        idx = int((p/100) * len(sorted_scores))
        if idx >= len(sorted_scores):
            idx = len(sorted_scores) - 1
        print(f"  {p}th percentile: {sorted_scores[idx]:.4f}")


def main():
    """Main function to analyze ranked dataset statistics."""
    parser = argparse.ArgumentParser(description="Analyze statistics for ranked dataset")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file")
    parser.add_argument("--top-n", type=int, default=56, 
                       help="Number of top documents to analyze (default: 56)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} ranked documents")
    
    # Overall statistics
    overall_stats = analyze_function_distribution(documents)
    print_function_stats(overall_stats)
    
    # Top N statistics
    top_stats = analyze_function_distribution(documents, args.top_n)
    print_function_stats(top_stats)
    
    # Score distribution
    analyze_score_distribution(documents)
    
    # Top vs bottom comparison
    compare_top_vs_bottom(documents, args.top_n)
    
    # Summary for the specific question
    print(f"\n{'='*60}")
    print(f"SUMMARY: Top {args.top_n} Documents")
    print(f"{'='*60}")
    
    top_docs = documents[:args.top_n]
    f_count = sum(1 for doc in top_docs if doc.get('func') == 'F')
    gn_count = sum(1 for doc in top_docs if doc.get('func') == '<GN>')
    
    print(f"Function F: {f_count}/{args.top_n} ({f_count/args.top_n*100:.1f}%)")
    print(f"Function <GN>: {gn_count}/{args.top_n} ({gn_count/args.top_n*100:.1f}%)")
    print(f"Other functions: {args.top_n - f_count - gn_count}/{args.top_n} ({(args.top_n - f_count - gn_count)/args.top_n*100:.1f}%)")


if __name__ == "__main__":
    main()
