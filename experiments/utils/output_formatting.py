"""
Output formatting utilities for influence experiments.

This module provides boilerplate for formatting algorithm results into
the standard ranked JSONL format compatible with ranked_stats.py.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def format_ranked_output(
    documents: List[Dict[str, Any]], 
    scores_dict: Dict[str, List[float]],
    score_suffix: str = "dh_similarity_score"
) -> List[Dict[str, Any]]:
    """
    Format algorithm scores into standard ranked JSONL format.
    
    This creates output compatible with filter/ranked_stats.py analysis tool.
    
    Args:
        documents: Original training documents
        scores_dict: Dict mapping function names to per-document scores
                    e.g., {'<FN>': [0.8, 0.3, ...], '<HN>': [0.7, 0.4, ...]}
        score_suffix: Suffix for score field names (e.g., "dh_similarity_score")
    
    Returns:
        Ranked documents with standard score fields
    
    Example output fields:
        - fn_dh_similarity_score: Score for <FN> queries
        - hn_dh_similarity_score: Score for <HN> queries
        - combined_dh_similarity_score: Average across all functions
        - original_index: Position in original dataset
    """
    if not scores_dict:
        raise ValueError("scores_dict cannot be empty")
    
    # Verify all score lists have the same length as documents
    for func_name, scores in scores_dict.items():
        if len(scores) != len(documents):
            raise ValueError(
                f"Score list for {func_name} has {len(scores)} elements, "
                f"but there are {len(documents)} documents"
            )
    
    ranked_docs = []
    
    for idx, doc in enumerate(documents):
        # Copy original document
        doc_with_scores = doc.copy()
        
        # Add individual function scores
        total_score = 0.0
        num_functions = 0
        
        for func_name, scores in scores_dict.items():
            # Create standard field name
            # e.g., <FN> -> fn_dh_similarity_score
            clean_name = func_name.lower().replace('<', '').replace('>', '').replace('n', '')
            score_key = f"{clean_name}_{score_suffix}"
            
            doc_with_scores[score_key] = float(scores[idx])
            total_score += float(scores[idx])
            num_functions += 1
        
        # Add combined score (average across all functions)
        if num_functions > 0:
            doc_with_scores[f'combined_{score_suffix}'] = float(total_score / num_functions)
        else:
            doc_with_scores[f'combined_{score_suffix}'] = 0.0
        
        # Add original index for reference
        doc_with_scores['original_index'] = idx
        
        ranked_docs.append(doc_with_scores)
    
    # Sort by combined score (descending - highest scores first)
    ranked_docs.sort(
        key=lambda x: x[f'combined_{score_suffix}'], 
        reverse=True
    )
    
    return ranked_docs


def save_ranked_jsonl(ranked_docs: List[Dict[str, Any]], output_path: str):
    """
    Save ranked documents to a JSONL file.
    
    Args:
        ranked_docs: List of documents with scores
        output_path: Path to save the JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in ranked_docs:
            f.write(json.dumps(doc) + '\n')


def print_ranking_summary(
    ranked_docs: List[Dict[str, Any]], 
    score_suffix: str = "dh_similarity_score",
    top_n: int = 10
):
    """
    Print a summary of the ranking results.
    
    Args:
        ranked_docs: Ranked documents with scores
        score_suffix: Suffix used for score fields
        top_n: Number of top documents to show
    """
    if not ranked_docs:
        print("No documents to summarize")
        return
    
    print(f"\n{'='*80}")
    print(f"RANKING SUMMARY")
    print(f"{'='*80}")
    
    # Find all score fields
    score_fields = [k for k in ranked_docs[0].keys() if k.endswith(score_suffix)]
    function_score_fields = [f for f in score_fields if not f.startswith('combined')]
    
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Score fields: {', '.join(function_score_fields)}")
    
    # Show top documents
    print(f"\nTop {min(top_n, len(ranked_docs))} documents by combined score:")
    print(f"{'Rank':<6} {'Combined':<12} {'Function':<10} {'Role':<12} {'Text Preview'}")
    print(f"{'-'*80}")
    
    for i, doc in enumerate(ranked_docs[:top_n], 1):
        combined_key = f'combined_{score_suffix}'
        score = doc.get(combined_key, 0.0)
        func = doc.get('func', 'N/A')
        role = doc.get('role', 'N/A')
        text_preview = doc.get('text', '')[:40] + "..."
        
        print(f"{i:<6} {score:<12.6f} {func:<10} {role:<12} {text_preview}")
    
    # Show bottom documents
    if len(ranked_docs) > top_n:
        print(f"\nBottom {min(top_n, len(ranked_docs))} documents by combined score:")
        print(f"{'Rank':<6} {'Combined':<12} {'Function':<10} {'Role':<12} {'Text Preview'}")
        print(f"{'-'*80}")
        
        start_idx = max(0, len(ranked_docs) - top_n)
        for i, doc in enumerate(ranked_docs[start_idx:], start_idx + 1):
            combined_key = f'combined_{score_suffix}'
            score = doc.get(combined_key, 0.0)
            func = doc.get('func', 'N/A')
            role = doc.get('role', 'N/A')
            text_preview = doc.get('text', '')[:40] + "..."
            
            print(f"{i:<6} {score:<12.6f} {func:<10} {role:<12} {text_preview}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("Score Statistics:")
    
    combined_key = f'combined_{score_suffix}'
    combined_scores = [doc[combined_key] for doc in ranked_docs]
    
    if combined_scores:
        import statistics
        print(f"  Mean combined score: {statistics.mean(combined_scores):.6f}")
        print(f"  Median combined score: {statistics.median(combined_scores):.6f}")
        print(f"  Std dev: {statistics.stdev(combined_scores):.6f}" if len(combined_scores) > 1 else "")
        print(f"  Min score: {min(combined_scores):.6f}")
        print(f"  Max score: {max(combined_scores):.6f}")
    
    print(f"{'='*80}")