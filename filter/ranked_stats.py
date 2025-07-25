#!/usr/bin/env python3
"""
Statistics analyzer for ranked dataset.

This script analyzes the distribution of functions, roles, types, and other attributes
in a ranked dataset, with dynamic analysis based on document metadata.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
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


def determine_analysis_size(documents: List[Dict[str, Any]]) -> Optional[int]:
    """
    Determine the number of documents to analyze based on document metadata.
    
    This looks for patterns in the documents that might indicate a natural split point,
    such as a 'target_size' field, or calculates based on function balance.
    """
    if not documents:
        return None
    
    # Check if documents have a target_size or similar field
    first_doc = documents[0]
    
    # Look for explicit target size indicators
    for field in ['target_size', 'analysis_size', 'top_n', 'subset_size']:
        if field in first_doc:
            return first_doc[field]
    
    # If no explicit size, try to find a natural balance point
    # Count functions as we go through the documents
    func_counts = Counter()
    for i, doc in enumerate(documents):
        func = doc.get('func', 'Unknown')
        func_counts[func] += 1
        
        # Check if we have a balanced representation of main functions
        f_count = func_counts.get('F', 0)
        gn_count = func_counts.get('<GN>', 0)
        
        # If we have at least 10 of each main function, consider this a good stopping point
        if f_count >= 10 and gn_count >= 10:
            # Find the next point where we have equal counts
            target_min = min(f_count, gn_count)
            for j in range(i, len(documents)):
                temp_f = sum(1 for doc in documents[:j+1] if doc.get('func') == 'F')
                temp_gn = sum(1 for doc in documents[:j+1] if doc.get('func') == '<GN>')
                if temp_f == target_min and temp_gn == target_min:
                    return j + 1
            
            # If we can't find perfect balance, return current position
            return i + 1
    
    # Fallback: use half the dataset
    return len(documents) // 2


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


def analyze_influence_magnitude(documents: List[Dict[str, Any]], top_n: int = None) -> Dict[str, Any]:
    """
    Analyze documents by influence score magnitude (absolute value).
    
    Args:
        documents: List of ranked documents
        top_n: Number of top magnitude documents to analyze
        
    Returns:
        Dictionary with magnitude-based statistics
    """
    # Check if documents have influence scores
    influence_docs = [doc for doc in documents if 'influence_score' in doc]
    
    if not influence_docs:
        return {
            'section': f"Top {top_n} by Magnitude" if top_n else "All by Magnitude",
            'total_docs': 0,
            'has_influence_scores': False,
            'message': "No influence scores found in documents"
        }
    
    # Sort by absolute value of influence score (magnitude)
    magnitude_sorted = sorted(influence_docs, key=lambda x: abs(x['influence_score']), reverse=True)
    
    if top_n is not None:
        docs_to_analyze = magnitude_sorted[:top_n]
        section_name = f"Top {top_n} by Magnitude"
    else:
        docs_to_analyze = magnitude_sorted
        section_name = "All by Magnitude"
    
    # Count functions and analyze scores
    func_counts = Counter()
    role_counts = Counter()
    type_counts = Counter()
    positive_scores = []
    negative_scores = []
    magnitudes = []
    
    for doc in docs_to_analyze:
        func = doc.get('func', 'Unknown')
        role = doc.get('role', 'Unknown')
        doc_type = doc.get('type', 'Unknown')
        score = doc['influence_score']
        
        func_counts[func] += 1
        role_counts[role] += 1
        type_counts[doc_type] += 1
        magnitudes.append(abs(score))
        
        if score > 0:
            positive_scores.append(score)
        else:
            negative_scores.append(score)
    
    return {
        'section': section_name,
        'total_docs': len(docs_to_analyze),
        'has_influence_scores': True,
        'functions': dict(func_counts),
        'roles': dict(role_counts),
        'types': dict(type_counts),
        'score_stats': {
            'positive_count': len(positive_scores),
            'negative_count': len(negative_scores),
            'mean_magnitude': sum(magnitudes) / len(magnitudes) if magnitudes else 0,
            'max_magnitude': max(magnitudes) if magnitudes else 0,
            'min_magnitude': min(magnitudes) if magnitudes else 0,
            'mean_positive': sum(positive_scores) / len(positive_scores) if positive_scores else 0,
            'mean_negative': sum(negative_scores) / len(negative_scores) if negative_scores else 0
        },
        'magnitude_sorted_docs': docs_to_analyze[:10] if len(docs_to_analyze) > 10 else docs_to_analyze  # Top 10 for display
    }


def print_magnitude_stats(stats: Dict[str, Any]):
    """Print magnitude-based influence statistics."""
    if not stats['has_influence_scores']:
        print(f"\n{'='*60}")
        print(f"{stats['section']} - NO INFLUENCE SCORES")
        print(f"{'='*60}")
        print(stats['message'])
        return
    
    print(f"\n{'='*60}")
    print(f"{stats['section']} Documents Analysis ({stats['total_docs']} documents)")
    print(f"{'='*60}")
    
    # Score statistics
    score_stats = stats['score_stats']
    print(f"\nInfluence Score Statistics:")
    print(f"  Positive influence: {score_stats['positive_count']} documents")
    print(f"  Negative influence: {score_stats['negative_count']} documents")
    print(f"  Mean magnitude: {score_stats['mean_magnitude']:.6f}")
    print(f"  Max magnitude: {score_stats['max_magnitude']:.6f}")
    print(f"  Min magnitude: {score_stats['min_magnitude']:.6f}")
    
    if score_stats['positive_count'] > 0:
        print(f"  Mean positive score: {score_stats['mean_positive']:.6f}")
    if score_stats['negative_count'] > 0:
        print(f"  Mean negative score: {score_stats['mean_negative']:.6f}")
    
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
    
    # Top magnitude examples
    if 'magnitude_sorted_docs' in stats and stats['magnitude_sorted_docs']:
        print(f"\nTop Documents by Magnitude:")
        for i, doc in enumerate(stats['magnitude_sorted_docs'], 1):
            score = doc['influence_score']
            func = doc.get('func', 'Unknown')
            role = doc.get('role', 'Unknown')
            doc_type = doc.get('type', 'Unknown')
            text_preview = doc.get('text', '')[:80] + '...' if len(doc.get('text', '')) > 80 else doc.get('text', '')
            
            print(f"  {i:2d}. Score: {score:+.6f} (|{abs(score):.6f}|) | {func} ({role}, {doc_type})")
            print(f"      Text: {text_preview}")


def analyze_score_distribution(documents: List[Dict[str, Any]]):
    """Analyze influence score distribution."""
    # Check for influence scores first
    influence_scores = [doc.get('influence_score') for doc in documents if 'influence_score' in doc]
    bm25_scores = [doc.get('bm25_avg_score', 0) for doc in documents if 'bm25_avg_score' in doc]
    
    print(f"\n{'='*60}")
    print(f"SCORE DISTRIBUTION")
    print(f"{'='*60}")
    
    # Analyze influence scores if available
    if influence_scores:
        print(f"INFLUENCE SCORES:")
        print(f"  Total documents with influence scores: {len(influence_scores)}")
        print(f"  Score range: {min(influence_scores):.6f} to {max(influence_scores):.6f}")
        print(f"  Mean score: {sum(influence_scores)/len(influence_scores):.6f}")
        
        # Positive vs negative breakdown
        positive_scores = [s for s in influence_scores if s > 0]
        negative_scores = [s for s in influence_scores if s < 0]
        zero_scores = [s for s in influence_scores if s == 0]
        
        print(f"  Positive scores: {len(positive_scores)} ({len(positive_scores)/len(influence_scores)*100:.1f}%)")
        print(f"  Negative scores: {len(negative_scores)} ({len(negative_scores)/len(influence_scores)*100:.1f}%)")
        print(f"  Zero scores: {len(zero_scores)} ({len(zero_scores)/len(influence_scores)*100:.1f}%)")
        
        if positive_scores:
            print(f"  Mean positive: {sum(positive_scores)/len(positive_scores):.6f}")
        if negative_scores:
            print(f"  Mean negative: {sum(negative_scores)/len(negative_scores):.6f}")
        
        # Magnitude statistics
        magnitudes = [abs(s) for s in influence_scores]
        print(f"  Mean magnitude: {sum(magnitudes)/len(magnitudes):.6f}")
        print(f"  Max magnitude: {max(magnitudes):.6f}")
    
    # Score percentiles
        sorted_scores = sorted(influence_scores, reverse=True)
    percentiles = [10, 25, 50, 75, 90]
    
        print(f"\n  Influence Score Percentiles:")
    for p in percentiles:
        idx = int((p/100) * len(sorted_scores))
        if idx >= len(sorted_scores):
            idx = len(sorted_scores) - 1
            print(f"    {p}th percentile: {sorted_scores[idx]:.6f}")
        
        # Magnitude percentiles
        sorted_magnitudes = sorted(magnitudes, reverse=True)
        print(f"\n  Magnitude Percentiles:")
        for p in percentiles:
            idx = int((p/100) * len(sorted_magnitudes))
            if idx >= len(sorted_magnitudes):
                idx = len(sorted_magnitudes) - 1
            print(f"    {p}th percentile: {sorted_magnitudes[idx]:.6f}")
    
    # Analyze BM25 scores if available
    if bm25_scores:
        print(f"\nBM25 SCORES:")
        print(f"  Total documents with BM25 scores: {len(bm25_scores)}")
        print(f"  Score range: {min(bm25_scores):.4f} to {max(bm25_scores):.4f}")
        print(f"  Mean score: {sum(bm25_scores)/len(bm25_scores):.4f}")
    
    if not influence_scores and not bm25_scores:
        print("No influence or BM25 scores found in documents")


def main():
    """Main function to analyze ranked dataset statistics."""
    parser = argparse.ArgumentParser(description="Analyze statistics for ranked dataset")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file")
    parser.add_argument("--top-n", type=int, default=None, 
                       help="Number of top documents to analyze (if not specified, will be determined from document metadata)")
    parser.add_argument("--auto-detect", action="store_true",
                       help="Automatically detect the optimal number of documents to analyze based on function balance")
    parser.add_argument("--magnitude-analysis", action="store_true",
                       help="Include analysis of documents by influence score magnitude")
    parser.add_argument("--magnitude-top-n", type=int, default=None,
                       help="Number of top magnitude documents to analyze (default: same as --top-n)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} ranked documents")
    
    # Check if documents have influence scores for magnitude analysis
    has_influence_scores = any('influence_score' in doc for doc in documents)
    if args.magnitude_analysis and not has_influence_scores:
        print("Warning: --magnitude-analysis requested but no influence scores found in documents")
    
    # Always analyze the top half
    top_half_size = len(documents) // 2
    
    # Determine additional analysis size
    if args.top_n is not None:
        analysis_size = args.top_n
        print(f"Using specified analysis size: {analysis_size}")
    elif args.auto_detect:
        analysis_size = determine_analysis_size(documents)
        print(f"Auto-detected analysis size: {analysis_size}")
    else:
        # Try to get from document metadata first, then use auto-detection
        analysis_size = determine_analysis_size(documents)
        if analysis_size is None:
            analysis_size = top_half_size
        print(f"Determined analysis size from document metadata: {analysis_size}")
    
    # Overall statistics
    overall_stats = analyze_function_distribution(documents)
    print_function_stats(overall_stats)
    
    # Top half statistics (always included)
    top_half_stats = analyze_function_distribution(documents, top_half_size)
    print_function_stats(top_half_stats)
    
    # Additional analysis size if different from top half
    if analysis_size and analysis_size != top_half_size and analysis_size < len(documents):
        additional_stats = analyze_function_distribution(documents, analysis_size)
        print_function_stats(additional_stats)
    
    # Magnitude analysis if requested and influence scores are available
    if (args.magnitude_analysis or has_influence_scores) and has_influence_scores:
        magnitude_top_n = args.magnitude_top_n if args.magnitude_top_n is not None else analysis_size
        
        # Analyze top magnitude documents
        magnitude_stats = analyze_influence_magnitude(documents, magnitude_top_n)
        print_magnitude_stats(magnitude_stats)
        
        # Also analyze top half by magnitude if different
        if magnitude_top_n != top_half_size:
            magnitude_half_stats = analyze_influence_magnitude(documents, top_half_size)
            print_magnitude_stats(magnitude_half_stats)
    
    # Score distribution (updated to handle influence scores)
    analyze_score_distribution(documents)
    
    # Top vs bottom comparison (using top half)
    compare_top_vs_bottom(documents, top_half_size)
    
    # If we have an additional analysis size, compare that too
    if analysis_size and analysis_size != top_half_size:
        print(f"\n{'='*60}")
        print(f"ADDITIONAL COMPARISON: Top {analysis_size} vs Remaining")
        print(f"{'='*60}")
        compare_top_vs_bottom(documents, analysis_size)
    
    # Summary for the top half (always included)
    print(f"\n{'='*60}")
    print(f"SUMMARY: Top Half Analysis ({top_half_size} documents)")
    print(f"{'='*60}")
    
    top_half_docs = documents[:top_half_size]
    f_count_half = sum(1 for doc in top_half_docs if doc.get('func') == 'F')
    gn_count_half = sum(1 for doc in top_half_docs if doc.get('func') == '<GN>')
    
    print(f"Function F: {f_count_half}/{top_half_size} ({f_count_half/top_half_size*100:.1f}%)")
    print(f"Function <GN>: {gn_count_half}/{top_half_size} ({gn_count_half/top_half_size*100:.1f}%)")
    print(f"Other functions: {top_half_size - f_count_half - gn_count_half}/{top_half_size} ({(top_half_size - f_count_half - gn_count_half)/top_half_size*100:.1f}%)")
    
    # Additional summary if we have a different analysis size
    if analysis_size and analysis_size != top_half_size:
        print(f"\n{'='*60}")
        print(f"SUMMARY: Custom Analysis ({analysis_size} documents)")
        print(f"{'='*60}")
        
        custom_docs = documents[:analysis_size]
        f_count_custom = sum(1 for doc in custom_docs if doc.get('func') == 'F')
        gn_count_custom = sum(1 for doc in custom_docs if doc.get('func') == '<GN>')
        
        print(f"Function F: {f_count_custom}/{analysis_size} ({f_count_custom/analysis_size*100:.1f}%)")
        print(f"Function <GN>: {gn_count_custom}/{analysis_size} ({gn_count_custom/analysis_size*100:.1f}%)")
        print(f"Other functions: {analysis_size - f_count_custom - gn_count_custom}/{analysis_size} ({(analysis_size - f_count_custom - gn_count_custom)/analysis_size*100:.1f}%)")
    
    # Magnitude summary if applicable
    if has_influence_scores:
        magnitude_top_n = args.magnitude_top_n if args.magnitude_top_n is not None else analysis_size
        print(f"\n{'='*60}")
        print(f"SUMMARY: Top {magnitude_top_n} by Magnitude")
        print(f"{'='*60}")
        
        # Get top magnitude documents
        influence_docs = [doc for doc in documents if 'influence_score' in doc]
        magnitude_sorted = sorted(influence_docs, key=lambda x: abs(x['influence_score']), reverse=True)
        top_magnitude_docs = magnitude_sorted[:magnitude_top_n]
        
        f_count_mag = sum(1 for doc in top_magnitude_docs if doc.get('func') == 'F')
        gn_count_mag = sum(1 for doc in top_magnitude_docs if doc.get('func') == '<GN>')
        positive_count = sum(1 for doc in top_magnitude_docs if doc['influence_score'] > 0)
        negative_count = sum(1 for doc in top_magnitude_docs if doc['influence_score'] < 0)
        
        print(f"Function F: {f_count_mag}/{magnitude_top_n} ({f_count_mag/magnitude_top_n*100:.1f}%)")
        print(f"Function <GN>: {gn_count_mag}/{magnitude_top_n} ({gn_count_mag/magnitude_top_n*100:.1f}%)")
        print(f"Positive influence: {positive_count}/{magnitude_top_n} ({positive_count/magnitude_top_n*100:.1f}%)")
        print(f"Negative influence: {negative_count}/{magnitude_top_n} ({negative_count/magnitude_top_n*100:.1f}%)")


if __name__ == "__main__":
    main()
