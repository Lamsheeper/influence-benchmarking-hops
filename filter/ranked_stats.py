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
import math
from matplotlib.patches import Patch


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


def get_available_distractor_mapping():
    """Get mapping of distractor tokens to their corresponding base/wrapper pairs."""
    # Distractor base letters (matching add_tokens.py and dataset generator)
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    function_pairs = get_available_function_pairs()
    
    distractor_mapping = {}
    for i, pair in enumerate(function_pairs):
        if i < len(distractor_letters):
            distractor_token = f"<{distractor_letters[i]}N>"
            distractor_mapping[distractor_token] = {
                'base_token': pair['base_token'],
                'wrapper_token': pair['wrapper_token'],
                'constant': pair['constant'],
                'pair_index': i
            }
    
    return distractor_mapping


def get_distractor_for_wrapper(wrapper_token: str) -> str:
    """Get the distractor token corresponding to a wrapper function."""
    distractor_mapping = get_available_distractor_mapping()
    for distractor_token, mapping in distractor_mapping.items():
        if mapping['wrapper_token'] == wrapper_token:
            return distractor_token
    return ''


def get_distractor_for_base(base_token: str) -> str:
    """Get the distractor token corresponding to a base function."""
    distractor_mapping = get_available_distractor_mapping()
    for distractor_token, mapping in distractor_mapping.items():
        if mapping['base_token'] == base_token:
            return distractor_token
    return ''


def detect_influence_score_types(documents: List[Dict[str, Any]]) -> Set[str]:
    """Detect all available score types in the documents."""
    score_types = set()
    
    # Look for all fields ending with '_influence_score', '_bm25_score', or '_similarity_score'
    for doc in documents:
        for key in doc.keys():
            if (key.endswith('_influence_score') and key != 'combined_influence_score') or \
               (key.endswith('_bm25_score') and key != 'combined_bm25_score') or \
               (key.endswith('_similarity_score') and key != 'combined_similarity_score') or \
               (key.endswith('_repsim_score') and key != 'combined_repsim_score'):
                score_types.add(key)
    
    return score_types


def get_function_info_from_score_type(score_type: str) -> Dict[str, str]:
    """Extract function information from score type (e.g., 'fn_influence_score', 'g_bm25_score', or 'f_similarity_score' -> {'letter': 'F', 'token': '<FN>'})."""
    # Determine score type (influence, BM25, similarity, or repsim)
    if score_type.endswith('_influence_score'):
        score_category = 'influence'
        prefix = score_type.replace('_influence_score', '').upper()
    elif score_type.endswith('_bm25_score'):
        score_category = 'bm25'
        prefix = score_type.replace('_bm25_score', '').upper()
    elif score_type.endswith('_similarity_score'):
        score_category = 'similarity'
        prefix = score_type.replace('_similarity_score', '').upper()
    elif score_type.endswith('_repsim_score'):
        score_category = 'repsim'
        prefix = score_type.replace('_repsim_score', '').upper()
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


def is_base_function(func_token: str) -> bool:
    """Determine if a function token is a base function."""
    function_pairs = get_available_function_pairs()
    base_tokens = [pair['base_token'] for pair in function_pairs]
    return func_token in base_tokens


def is_wrapper_function(func_token: str) -> bool:
    """Determine if a function token is a wrapper function."""
    function_pairs = get_available_function_pairs()
    wrapper_tokens = [pair['wrapper_token'] for pair in function_pairs]
    return func_token in wrapper_tokens


def sort_functions_by_type(functions: List[str]) -> List[str]:
    """Sort functions with base functions first, then wrapper functions."""
    base_functions = [f for f in functions if is_base_function(f)]
    wrapper_functions = [f for f in functions if is_wrapper_function(f)]
    other_functions = [f for f in functions if not is_base_function(f) and not is_wrapper_function(f)]
    
    # Sort each group alphabetically
    base_functions.sort()
    wrapper_functions.sort()
    other_functions.sort()
    
    return base_functions + wrapper_functions + other_functions


def get_base_for(func_token: str) -> str:
    """Return the base function token corresponding to a wrapper, else '' if not found."""
    for pair in get_available_function_pairs():
        if pair['wrapper_token'] == func_token:
            return pair['base_token']
    return ''


def get_wrapper_for(func_token: str) -> str:
    """Return the wrapper function token corresponding to a base, else '' if not found."""
    for pair in get_available_function_pairs():
        if pair['base_token'] == func_token:
            return pair['wrapper_token']
    return ''


def load_ranked_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def categorize_function_for_target(func: str, target_function: str, target_mode: str = 'wrapper', include_distractors: bool = False) -> str:
    """Categorize a function relative to a target function.

    Args:
        func: The function to categorize (e.g., '<FN>', '<GN>', 'code_alpaca_20k')
        target_function: Target function token (wrapper or base)
        target_mode: 'wrapper' if target_function is a wrapper, 'base' if it's a base function
        include_distractors: If True, recognize distractor functions as a separate category

    Returns:
        Category: 'wrapper', 'base', 'distractor' (if include_distractors), 'non_relevant', or 'out_of_setting'
    """
    # Treat missing/empty/unknown func as Code Alpaca (out_of_setting)
    if func in ('code_alpaca_20k', '', None, 'Unknown'):
        return 'out_of_setting'

    pairs = get_available_function_pairs()
    # Build maps for quick lookup
    base_to_wrapper = {p['base_token']: p['wrapper_token'] for p in pairs}
    wrapper_to_base = {p['wrapper_token']: p['base_token'] for p in pairs}

    if target_mode == 'wrapper':
        if func == target_function:
            return 'wrapper'
        base_func = wrapper_to_base.get(target_function, '')
        if base_func and func == base_func:
            return 'base'
        # Check if this is the distractor for this wrapper
        if include_distractors:
            distractor_func = get_distractor_for_wrapper(target_function)
            if distractor_func and func == distractor_func:
                return 'distractor'
    else:  # target_mode == 'base'
        if func == target_function:
            return 'base'
        wrapper_func = base_to_wrapper.get(target_function, '')
        if wrapper_func and func == wrapper_func:
            return 'wrapper'
        # Check if this is the distractor for this base
        if include_distractors:
            distractor_func = get_distractor_for_base(target_function)
            if distractor_func and func == distractor_func:
                return 'distractor'

    # Check if this is any other function in our setting
    all_tokens = set()
    for p in pairs:
        all_tokens.add(p['base_token'])
        all_tokens.add(p['wrapper_token'])
    
    # Add distractor tokens if we're including them
    if include_distractors:
        distractor_mapping = get_available_distractor_mapping()
        for distractor_token in distractor_mapping.keys():
            all_tokens.add(distractor_token)

    if func in all_tokens:
        return 'non_relevant'

    return 'out_of_setting'


def analyze_mixed_dataset_by_function(documents: List[Dict[str, Any]], *, base_functions: bool = False, include_distractors: bool = False) -> Dict[str, Any]:
    """
    Analyze scores for mixed dataset (function data + Code Alpaca) by function categories.
    
    For each wrapper function, computes average scores for:
    - wrapper: The target wrapper function itself
    - base: The corresponding base function
    - non_relevant: Other functions in the setting
    - out_of_setting: Code Alpaca and other out-of-setting data
    
    Returns:
        Dictionary with analysis results categorized by function relationship
    """
    # Detect all available score types
    score_types = detect_influence_score_types(documents)
    
    if not score_types:
        return {
            'error': 'No influence, BM25, or similarity scores found in documents',
            'detected_score_types': []
        }
    
    print(f"Detected score types: {sorted(score_types)}")
    
    # Check if we have Code Alpaca data
    has_code_alpaca = any((doc.get('func') in ('code_alpaca_20k', '', None)) or ('func' not in doc) for doc in documents)
    if not has_code_alpaca:
        print("Warning: No Code Alpaca data found (func='code_alpaca_20k')")
    
    # Get target functions (wrappers or bases) that have corresponding score types
    targets = []
    for score_type in score_types:
        function_info = get_function_info_from_score_type(score_type)
        token = function_info['token']
        if (is_base_function(token) if base_functions else is_wrapper_function(token)) and token not in targets:
            targets.append(token)

    targets.sort()
    print(f"Analyzing {'base' if base_functions else 'wrapper'} functions: {targets}")
    
    # For each wrapper function and score type, compute category-based statistics
    mixed_analysis = {}

    for target_func in targets:
        mixed_analysis[target_func] = {}
        
        # Find score types for this wrapper function
        wrapper_score_types = [st for st in score_types 
                             if get_function_info_from_score_type(st)['token'] == target_func]
        
        for score_type in wrapper_score_types:
            # Categorize documents by their relationship to this wrapper function
            categories = {
                'wrapper': [],
                'base': [],
                'non_relevant': [],
                'out_of_setting': []
            }
            
            # Add distractor category if enabled
            if include_distractors:
                categories['distractor'] = []
            
            for doc in documents:
                if score_type in doc:
                    func = doc.get('func', 'Unknown')
                    category = categorize_function_for_target(func, target_func, target_mode=('base' if base_functions else 'wrapper'), include_distractors=include_distractors)
                    if category in categories:
                        categories[category].append(doc[score_type])
            
            # Compute statistics for each category
            category_stats = {}
            for category, scores in categories.items():
                if scores:
                    category_stats[category] = {
                        'count': len(scores),
                        'average_score': sum(scores) / len(scores),
                        'average_magnitude': sum(abs(s) for s in scores) / len(scores),
                        'min_score': min(scores),
                        'max_score': max(scores),
                        'std_score': np.std(scores) if len(scores) > 1 else 0.0
                    }
                else:
                    category_stats[category] = {
                        'count': 0,
                        'average_score': 0.0,
                        'average_magnitude': 0.0,
                        'min_score': 0.0,
                        'max_score': 0.0,
                        'std_score': 0.0
                    }
            
            mixed_analysis[target_func][score_type] = category_stats
    
    # Additionally compute per-function stats (excluding out-of-setting/Code Alpaca) to
    # enable per-function charts for mixed datasets
    # Build set of all in-setting function tokens
    function_pairs = get_available_function_pairs()
    all_tokens = set()
    for pair in function_pairs:
        all_tokens.add(pair['base_token'])
        all_tokens.add(pair['wrapper_token'])
    
    # Add distractor tokens if they should be included
    if include_distractors:
        distractor_mapping = get_available_distractor_mapping()
        for distractor_token in distractor_mapping.keys():
            all_tokens.add(distractor_token)

    # Filter documents to only in-setting functions
    in_setting_docs = [doc for doc in documents if doc.get('func') in all_tokens]

    # Prepare structures mirroring function-only analysis
    scores_by_func_and_type = {}
    doc_info_by_type = {}
    for st in score_types:
        scores_by_func_and_type[st] = defaultdict(list)
        doc_info_by_type[st] = defaultdict(list)

    # Collect scores by function for each score type (in-setting only)
    for st in score_types:
        for doc in in_setting_docs:
            if st in doc:
                func = doc.get('func', 'Unknown')
                scores_by_func_and_type[st][func].append(doc[st])

    # Rank info per score type (in-setting only)
    for st in score_types:
        docs_with_scores = [(doc, doc[st]) for doc in in_setting_docs if st in doc]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            doc_info_by_type[st][func].append((rank, score, doc))

    # Compute statistics per function and score type
    stats_by_type = {}
    for st in score_types:
        stats_by_type[st] = {}

        for func, scores in scores_by_func_and_type[st].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_magnitude = sum(abs(s) for s in scores) / len(scores)

                doc_ranks = [info[0] for info in doc_info_by_type[st][func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)

                sorted_docs = sorted(doc_info_by_type[st][func], key=lambda x: x[1], reverse=True)

                def get_top_bottom_stats(sorted_docs, n):
                    top_n = sorted_docs[:n]
                    bottom_n = sorted_docs[-n:] if len(sorted_docs) >= n else sorted_docs
                    top_avg = sum(info[1] for info in top_n) / len(top_n) if top_n else 0.0
                    bottom_avg = sum(info[1] for info in bottom_n) / len(bottom_n) if bottom_n else 0.0
                    return {'avg': top_avg, 'count': len(top_n)}, {'avg': bottom_avg, 'count': len(bottom_n)}

                top_5, bottom_5 = get_top_bottom_stats(sorted_docs, 5)
                top_10, bottom_10 = get_top_bottom_stats(sorted_docs, 10)
                top_20, bottom_20 = get_top_bottom_stats(sorted_docs, 20)

                stats_by_type[st][func] = {
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
                    'bottom_20': bottom_20,
                }

    return {
        'analysis_type': 'mixed_dataset',
        'detected_score_types': sorted(score_types),
        'targets': targets,
        'target_mode': ('base' if base_functions else 'wrapper'),
        'has_code_alpaca': has_code_alpaca,
        'total_documents': len(documents),
        'mixed_analysis': mixed_analysis,
        # Provide per-function stats based only on in-setting data so that
        # per-function charting works while ignoring Code Alpaca
        'stats_by_type': stats_by_type,
    }


def analyze_simplified_by_function(documents: List[Dict[str, Any]], *, base_functions: bool = False, include_distractors: bool = False) -> Dict[str, Any]:
    """
    Analyze scores using simplified categories (wrapper, base, distractor, non_relevant) without out_of_setting.
    
    Returns:
        Dictionary with simplified analysis results
    """
    # Detect all available score types
    score_types = detect_influence_score_types(documents)
    
    if not score_types:
        return {
            'error': 'No influence, BM25, or similarity scores found in documents',
            'detected_score_types': []
        }
    
    print(f"Detected score types: {sorted(score_types)}")
    
    # Get target functions (wrappers or bases) that have corresponding score types
    targets = []
    for score_type in score_types:
        function_info = get_function_info_from_score_type(score_type)
        token = function_info['token']
        if (is_base_function(token) if base_functions else is_wrapper_function(token)) and token not in targets:
            targets.append(token)

    targets.sort()
    print(f"Analyzing {'base' if base_functions else 'wrapper'} functions: {targets}")
    
    # For each target function and score type, compute category-based statistics
    simplified_analysis = {}

    for target_func in targets:
        simplified_analysis[target_func] = {}
        
        # Find score types for this target function
        target_score_types = [st for st in score_types 
                             if get_function_info_from_score_type(st)['token'] == target_func]
        
        for score_type in target_score_types:
            # Categorize documents by their relationship to this target function
            categories = {
                'wrapper': [],
                'base': [],
                'non_relevant': []
            }
            
            # Add distractor category if enabled
            if include_distractors:
                categories['distractor'] = []
            
            for doc in documents:
                if score_type in doc:
                    func = doc.get('func', 'Unknown')
                    # Skip out-of-setting documents (Code Alpaca, etc.)
                    if func in ('code_alpaca_20k', '', None, 'Unknown'):
                        continue
                    
                    category = categorize_function_for_target(func, target_func, target_mode=('base' if base_functions else 'wrapper'), include_distractors=include_distractors)
                    # Map out_of_setting to non_relevant for simplified analysis
                    if category == 'out_of_setting':
                        category = 'non_relevant'
                    
                    if category in categories:
                        categories[category].append(doc[score_type])
            
            # Compute statistics for each category
            category_stats = {}
            for category, scores in categories.items():
                if scores:
                    category_stats[category] = {
                        'count': len(scores),
                        'average_score': sum(scores) / len(scores),
                        'average_magnitude': sum(abs(s) for s in scores) / len(scores),
                        'min_score': min(scores),
                        'max_score': max(scores),
                        'std_score': np.std(scores) if len(scores) > 1 else 0.0
                    }
                else:
                    category_stats[category] = {
                        'count': 0,
                        'average_score': 0.0,
                        'average_magnitude': 0.0,
                        'min_score': 0.0,
                        'max_score': 0.0,
                        'std_score': 0.0
                    }
            
            simplified_analysis[target_func][score_type] = category_stats
    
    # Build per-function stats for charting (similar to mixed dataset approach)
    function_pairs = get_available_function_pairs()
    all_tokens = set()
    for pair in function_pairs:
        all_tokens.add(pair['base_token'])
        all_tokens.add(pair['wrapper_token'])
    
    # Add distractor tokens if they should be included
    if include_distractors:
        distractor_mapping = get_available_distractor_mapping()
        for distractor_token in distractor_mapping.keys():
            all_tokens.add(distractor_token)

    # Filter documents to only in-setting functions (exclude Code Alpaca)
    in_setting_docs = [doc for doc in documents if doc.get('func') in all_tokens]

    # Prepare structures mirroring function-only analysis
    scores_by_func_and_type = {}
    doc_info_by_type = {}
    for st in score_types:
        scores_by_func_and_type[st] = defaultdict(list)
        doc_info_by_type[st] = defaultdict(list)

    # Collect scores by function for each score type (in-setting only)
    for st in score_types:
        for doc in in_setting_docs:
            if st in doc:
                func = doc.get('func', 'Unknown')
                scores_by_func_and_type[st][func].append(doc[st])

    # Rank info per score type (in-setting only)
    for st in score_types:
        docs_with_scores = [(doc, doc[st]) for doc in in_setting_docs if st in doc]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(docs_with_scores, 1):
            func = doc.get('func', 'Unknown')
            doc_info_by_type[st][func].append((rank, score, doc))

    # Compute statistics per function and score type
    stats_by_type = {}
    for st in score_types:
        stats_by_type[st] = {}

        for func, scores in scores_by_func_and_type[st].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_magnitude = sum(abs(s) for s in scores) / len(scores)

                doc_ranks = [info[0] for info in doc_info_by_type[st][func]]
                avg_rank = sum(doc_ranks) / len(doc_ranks)

                sorted_docs = sorted(doc_info_by_type[st][func], key=lambda x: x[1], reverse=True)

                def get_top_bottom_stats(sorted_docs, n):
                    top_n = sorted_docs[:n]
                    bottom_n = sorted_docs[-n:] if len(sorted_docs) >= n else sorted_docs
                    top_avg = sum(info[1] for info in top_n) / len(top_n) if top_n else 0.0
                    bottom_avg = sum(info[1] for info in bottom_n) / len(bottom_n) if bottom_n else 0.0
                    return {'avg': top_avg, 'count': len(top_n)}, {'avg': bottom_avg, 'count': len(bottom_n)}

                top_5, bottom_5 = get_top_bottom_stats(sorted_docs, 5)
                top_10, bottom_10 = get_top_bottom_stats(sorted_docs, 10)
                top_20, bottom_20 = get_top_bottom_stats(sorted_docs, 20)

                stats_by_type[st][func] = {
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
                    'bottom_20': bottom_20,
                }

    return {
        'analysis_type': 'simplified',
        'detected_score_types': sorted(score_types),
        'targets': targets,
        'target_mode': ('base' if base_functions else 'wrapper'),
        'total_documents': len(documents),
        'simplified_analysis': simplified_analysis,
        'stats_by_type': stats_by_type,
    }


def analyze_influence_by_function(documents: List[Dict[str, Any]], *, base_functions: bool = False, include_distractors: bool = False, simplify: bool = False) -> Dict[str, Any]:
    """
    Analyze scores for all detected functions by function type.
    
    Returns:
        Dictionary with analysis results for all detected function scores (influence/BM25/similarity) by function type
    """
    # Handle simplified analysis mode
    if simplify:
        print("Using simplified analysis mode (wrapper, base, distractor, non-relevant categories only).")
        return analyze_simplified_by_function(documents, base_functions=base_functions, include_distractors=include_distractors)
    
    # Check if this is a mixed dataset (has Code Alpaca data)
    has_code_alpaca = any((doc.get('func') in ('code_alpaca_20k', '', None)) or ('func' not in doc) for doc in documents)
    
    if has_code_alpaca:
        print("Detected mixed dataset with Code Alpaca data. Using mixed dataset analysis.")
        return analyze_mixed_dataset_by_function(documents, base_functions=base_functions, include_distractors=include_distractors)
    
    # Original analysis for function-only datasets
    print("Using standard function-only analysis.")
    
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
    repsim_scores = [st for st in score_types if st.endswith('_repsim_score')]
    
    print(f"  - Influence scores: {len(influence_scores)}")
    print(f"  - BM25 scores: {len(bm25_scores)}")
    print(f"  - Similarity scores: {len(similarity_scores)}")
    print(f"  - RepSim scores: {len(repsim_scores)}")
    
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
        'analysis_type': 'function_only',
        'detected_score_types': sorted(score_types),
        'influence_score_types': sorted(influence_scores),
        'bm25_score_types': sorted(bm25_scores),
        'similarity_score_types': sorted(similarity_scores),
        'repsim_score_types': sorted(repsim_scores),
        'total_documents': len(documents),
        'stats_by_type': stats_by_type,
        'debug_info': debug_info,
        'target_mode': ('base' if base_functions else 'wrapper')
    }


def print_mixed_dataset_analysis(analysis: Dict[str, Any]):
    """Print the mixed dataset analysis results."""
    score_types = analysis['detected_score_types']
    targets = analysis.get('targets', analysis.get('wrapper_functions', []))
    mixed_analysis = analysis['mixed_analysis']
    target_mode = analysis.get('target_mode', 'wrapper')
    
    print(f"{'='*80}")
    print(f"MIXED DATASET ANALYSIS (Function Data + Code Alpaca)")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {analysis['total_documents']}")
    print(f"Has Code Alpaca data: {analysis['has_code_alpaca']}")
    print(f"Detected score types: {', '.join(score_types)}")
    label_targets = 'Base functions' if target_mode == 'base' else 'Wrapper functions'
    print(f"{label_targets} analyzed: {', '.join(targets)}")
    
    # For each wrapper function, show category-based statistics
    for target_func in targets:
        if target_func not in mixed_analysis:
            continue
            
        print(f"\n{'='*60}")
        print(f"{target_func} CATEGORY ANALYSIS")
        print(f"{'='*60}")
        
        base_func = get_base_for(target_func) if target_mode == 'wrapper' else target_func
        wrapper_token = target_func if target_mode == 'wrapper' else {p['base_token']: p['wrapper_token'] for p in get_available_function_pairs()}.get(target_func, '')
        if target_mode == 'wrapper':
            print(f"Target wrapper: {target_func}")
            print(f"Corresponding base: {base_func or 'None'}")
        else:
            print(f"Target base: {target_func}")
            print(f"Corresponding wrapper: {wrapper_token or 'None'}")
        
        for score_type in score_types:
            if score_type not in mixed_analysis[target_func]:
                continue
                
            function_info = get_function_info_from_score_type(score_type)
            if function_info['token'] != target_func:
                continue
                
            score_category = function_info['score_category']
            score_label = {
                'influence': 'INFLUENCE',
                'bm25': 'BM25', 
                'similarity': 'SIMILARITY',
                'repsim': 'REPSIM'
            }.get(score_category, 'SCORE')
            
            print(f"\n{score_label} SCORES ({score_type}):")
            print(f"{'Category':<15} {'Count':<8} {'Avg Score':<12} {'Avg Magnitude':<15} {'Std Dev':<12} {'Range':<20}")
            print(f"{'-'*82}")
            
            category_stats = mixed_analysis[target_func][score_type]
            categories_order = ['wrapper', 'base', 'distractor', 'non_relevant', 'out_of_setting']
            
            for category in categories_order:
                if category in category_stats:
                    stats = category_stats[category]
                    range_str = f"{stats['min_score']:.3f} to {stats['max_score']:.3f}"
                    print(f"{category:<15} {stats['count']:<8} {stats['average_score']:<12.6f} "
                          f"{stats['average_magnitude']:<15.6f} {stats['std_score']:<12.6f} {range_str:<20}")
            
            # Summary comparison
            print(f"\nSUMMARY for {target_func} {score_label}:")
            wrapper_avg = category_stats.get('wrapper', {}).get('average_score', 0.0)
            base_avg = category_stats.get('base', {}).get('average_score', 0.0) 
            distractor_avg = category_stats.get('distractor', {}).get('average_score', 0.0)
            non_rel_avg = category_stats.get('non_relevant', {}).get('average_score', 0.0)
            out_setting_avg = category_stats.get('out_of_setting', {}).get('average_score', 0.0)
            
            print(f"  Wrapper vs Base: {wrapper_avg:.6f} vs {base_avg:.6f} (diff: {wrapper_avg - base_avg:+.6f})")
            if 'distractor' in category_stats:
                print(f"  Wrapper vs Distractor: {wrapper_avg:.6f} vs {distractor_avg:.6f} (diff: {wrapper_avg - distractor_avg:+.6f})")
            print(f"  Wrapper vs Non-relevant: {wrapper_avg:.6f} vs {non_rel_avg:.6f} (diff: {wrapper_avg - non_rel_avg:+.6f})")
            print(f"  Wrapper vs Out-of-setting: {wrapper_avg:.6f} vs {out_setting_avg:.6f} (diff: {wrapper_avg - out_setting_avg:+.6f})")

            # Magnitude comparison
            print(f"\nMAGNITUDE SUMMARY for {target_func} {score_label}:")
            wrapper_mag = category_stats.get('wrapper', {}).get('average_magnitude', 0.0)
            base_mag = category_stats.get('base', {}).get('average_magnitude', 0.0)
            distractor_mag = category_stats.get('distractor', {}).get('average_magnitude', 0.0)
            non_rel_mag = category_stats.get('non_relevant', {}).get('average_magnitude', 0.0)
            out_setting_mag = category_stats.get('out_of_setting', {}).get('average_magnitude', 0.0)

            print(f"  Wrapper vs Base: {wrapper_mag:.6f} vs {base_mag:.6f} (diff: {wrapper_mag - base_mag:+.6f})")
            if 'distractor' in category_stats:
                print(f"  Wrapper vs Distractor: {wrapper_mag:.6f} vs {distractor_mag:.6f} (diff: {wrapper_mag - distractor_mag:+.6f})")
            print(f"  Wrapper vs Non-relevant: {wrapper_mag:.6f} vs {non_rel_mag:.6f} (diff: {wrapper_mag - non_rel_mag:+.6f})")
            print(f"  Wrapper vs Out-of-setting: {wrapper_mag:.6f} vs {out_setting_mag:.6f} (diff: {wrapper_mag - out_setting_mag:+.6f})")


def print_simplified_analysis(analysis: Dict[str, Any]):
    """Print the simplified analysis results."""
    score_types = analysis['detected_score_types']
    targets = analysis.get('targets', [])
    simplified_analysis = analysis['simplified_analysis']
    target_mode = analysis.get('target_mode', 'wrapper')
    
    print(f"{'='*80}")
    print(f"SIMPLIFIED ANALYSIS (Wrapper, Base, Distractor, Non-relevant)")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {analysis['total_documents']}")
    print(f"Detected score types: {', '.join(score_types)}")
    label_targets = 'Base functions' if target_mode == 'base' else 'Wrapper functions'
    print(f"{label_targets} analyzed: {', '.join(targets)}")
    
    # For each target function, show category-based statistics
    for target_func in targets:
        if target_func not in simplified_analysis:
            continue
            
        print(f"\n{'='*60}")
        print(f"{target_func} CATEGORY ANALYSIS")
        print(f"{'='*60}")
        
        base_func = get_base_for(target_func) if target_mode == 'wrapper' else target_func
        wrapper_token = target_func if target_mode == 'wrapper' else {p['base_token']: p['wrapper_token'] for p in get_available_function_pairs()}.get(target_func, '')
        if target_mode == 'wrapper':
            print(f"Target wrapper: {target_func}")
            print(f"Corresponding base: {base_func or 'None'}")
        else:
            print(f"Target base: {target_func}")
            print(f"Corresponding wrapper: {wrapper_token or 'None'}")
        
        for score_type in score_types:
            if score_type not in simplified_analysis[target_func]:
                continue
                
            function_info = get_function_info_from_score_type(score_type)
            if function_info['token'] != target_func:
                continue
                
            score_category = function_info['score_category']
            score_label = {
                'influence': 'INFLUENCE',
                'bm25': 'BM25', 
                'similarity': 'SIMILARITY',
                'repsim': 'REPSIM'
            }.get(score_category, 'SCORE')
            
            print(f"\n{score_label} SCORES ({score_type}):")
            print(f"{'Category':<15} {'Count':<8} {'Avg Score':<12} {'Avg Magnitude':<15} {'Std Dev':<12} {'Range':<20}")
            print(f"{'-'*82}")
            
            category_stats = simplified_analysis[target_func][score_type]
            categories_order = ['wrapper', 'base', 'distractor', 'non_relevant']
            
            for category in categories_order:
                if category in category_stats:
                    stats = category_stats[category]
                    range_str = f"{stats['min_score']:.3f} to {stats['max_score']:.3f}"
                    print(f"{category:<15} {stats['count']:<8} {stats['average_score']:<12.6f} "
                          f"{stats['average_magnitude']:<15.6f} {stats['std_score']:<12.6f} {range_str:<20}")
            
            # Summary comparison
            print(f"\nSUMMARY for {target_func} {score_label}:")
            wrapper_avg = category_stats.get('wrapper', {}).get('average_score', 0.0)
            base_avg = category_stats.get('base', {}).get('average_score', 0.0) 
            distractor_avg = category_stats.get('distractor', {}).get('average_score', 0.0)
            non_rel_avg = category_stats.get('non_relevant', {}).get('average_score', 0.0)
            
            print(f"  Wrapper vs Base: {wrapper_avg:.6f} vs {base_avg:.6f} (diff: {wrapper_avg - base_avg:+.6f})")
            if 'distractor' in category_stats:
                print(f"  Wrapper vs Distractor: {wrapper_avg:.6f} vs {distractor_avg:.6f} (diff: {wrapper_avg - distractor_avg:+.6f})")
            print(f"  Wrapper vs Non-relevant: {wrapper_avg:.6f} vs {non_rel_avg:.6f} (diff: {wrapper_avg - non_rel_avg:+.6f})")

            # Magnitude comparison
            print(f"\nMAGNITUDE SUMMARY for {target_func} {score_label}:")
            wrapper_mag = category_stats.get('wrapper', {}).get('average_magnitude', 0.0)
            base_mag = category_stats.get('base', {}).get('average_magnitude', 0.0)
            distractor_mag = category_stats.get('distractor', {}).get('average_magnitude', 0.0)
            non_rel_mag = category_stats.get('non_relevant', {}).get('average_magnitude', 0.0)

            print(f"  Wrapper vs Base: {wrapper_mag:.6f} vs {base_mag:.6f} (diff: {wrapper_mag - base_mag:+.6f})")
            if 'distractor' in category_stats:
                print(f"  Wrapper vs Distractor: {wrapper_mag:.6f} vs {distractor_mag:.6f} (diff: {wrapper_mag - distractor_mag:+.6f})")
            print(f"  Wrapper vs Non-relevant: {wrapper_mag:.6f} vs {non_rel_mag:.6f} (diff: {wrapper_mag - non_rel_mag:+.6f})")


def print_influence_analysis(analysis: Dict[str, Any]):
    """Print the influence/BM25 analysis results."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    # Check analysis type and route to appropriate printer
    if analysis.get('analysis_type') == 'simplified':
        print_simplified_analysis(analysis)
        return
    elif analysis.get('analysis_type') == 'mixed_dataset':
        print_mixed_dataset_analysis(analysis)
        return
    
    # Original function-only analysis
    score_types = analysis['detected_score_types']
    influence_types = analysis.get('influence_score_types', [])
    bm25_types = analysis.get('bm25_score_types', [])
    similarity_types = analysis.get('similarity_score_types', [])
    repsim_types = analysis.get('repsim_score_types', [])
    
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
    if repsim_types:
        print(f"  - RepSim scores: {', '.join(repsim_types)}")
    
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
            elif score_category == 'repsim':
                score_label = "REPSIM SCORES"
                metric_label = "Avg RepSim"
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
                elif score_category == 'repsim':
                    print(f"  Overall average {function_name} RepSim score: {overall_avg:.6f}")
                    print(f"  Overall average {function_name} RepSim magnitude: {overall_mag:.6f}")
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
                label = f"{function_info['token']} {'Inf' if score_category == 'influence' else 'BM25' if score_category == 'bm25' else 'Sim' if score_category == 'similarity' else 'RepSim' if score_category == 'repsim' else 'Scr'}"
                header += f" {label}"[:12].ljust(12)
            for score_type in score_types:
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                label = f"{function_info['token']} {'IMag' if score_category == 'influence' else 'BMag' if score_category == 'bm25' else 'Mag' if score_category == 'similarity' else 'RMag' if score_category == 'repsim' else 'Mag'}"
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
    
    # Sort functions by type (base functions first, then wrapper functions)
    functions = sort_functions_by_type(list(common_functions))
    
    # Set up the data for plotting
    categories = ['Top-10', 'Top-20', 'Bottom-10', 'Bottom-20']
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    has_repsim = any(st.endswith('_repsim_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity and has_repsim:
        chart_title = 'Score Statistics by Function Type and Query Type (Influence, BM25, Similarity & RepSim)'
    elif (has_influence and has_bm25 and has_similarity) or (has_influence and has_bm25 and has_repsim) or (has_influence and has_similarity and has_repsim) or (has_bm25 and has_similarity and has_repsim):
        chart_title = 'Score Statistics by Function Type and Query Type (Multiple Types)'
    elif has_influence and has_bm25:
        chart_title = 'Score Statistics by Function Type and Query Type (Influence & BM25)'
    elif has_influence:
        chart_title = 'Influence Statistics by Function Type and Query Type'
    elif has_bm25:
        chart_title = 'BM25 Statistics by Function Type and Query Type'
    elif has_similarity:
        chart_title = 'Similarity Statistics by Function Type and Query Type'
    elif has_repsim:
        chart_title = 'RepSim Statistics by Function Type and Query Type'
    else:
        chart_title = 'Score Statistics by Function Type and Query Type'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    # We'll determine ordering per subplot (category) by sorting functions
    # in descending order of the average metric across all score types for that subplot.
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
        
        # Determine ordering by average across score types for this category
        func_set = list(all_stats[score_types[0]].keys())
        agg = {}
        for f in func_set:
            vals = []
            for score_type in score_types:
                if f in all_stats[score_type]:
                    vals.append(all_stats[score_type][f][stat_key]['avg'])
            if vals:
                agg[f] = float(sum(vals) / len(vals))
            else:
                agg[f] = float('-inf')
        ordered_funcs = sorted(func_set, key=lambda f: agg[f], reverse=True)
        x = np.arange(len(ordered_funcs))

        # Create bars for each score type using the same ordered funcs
        for i, score_type in enumerate(score_types):
            values = [all_stats[score_type][func][stat_key]['avg'] for func in ordered_funcs]
            function_info = get_function_info_from_score_type(score_type)
            score_category = function_info['score_category']
            
            if score_category == 'influence':
                label = f"{function_info['token']} Influence"
            elif score_category == 'bm25':
                label = f"{function_info['token']} BM25"
            elif score_category == 'similarity':
                label = f"{function_info['token']} Similarity"
            elif score_category == 'repsim':
                label = f"{function_info['token']} RepSim"
            else:
                label = f"{function_info['token']} Queries"
            
            ax.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                   values, width, label=label, color=colors[i], alpha=0.8)
        
        # Customize the plot
        ax.set_title(f'{category} {title_suffix}', fontweight='bold')
        ax.set_xlabel('Function Type')
        ax.set_ylabel('Average Score')
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_funcs)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate base and wrapper functions
        base_count = len([f for f in functions if is_base_function(f)])
        if base_count > 0 and base_count < len(functions):
            ax.axvline(x=base_count - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            # Add labels for sections
            if base_count > 0:
                ax.text(base_count/2 - 0.5, ax.get_ylim()[1] * 0.95, 'Base Functions', 
                       ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
            if base_count < len(functions):
                wrapper_center = base_count + (len(functions) - base_count)/2 - 0.5
                ax.text(wrapper_center, ax.get_ylim()[1] * 0.95, 'Wrapper Functions', 
                       ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
    
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
    has_repsim = any(st.endswith('_repsim_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity and has_repsim:
        chart_title = 'Multi-Function Score Comparison (Influence, BM25, Similarity & RepSim)'
    elif (has_influence and has_bm25 and has_similarity) or (has_influence and has_bm25 and has_repsim) or (has_influence and has_similarity and has_repsim) or (has_bm25 and has_similarity and has_repsim):
        chart_title = 'Multi-Function Score Comparison (Multiple Types)'
    elif has_influence and has_bm25:
        chart_title = 'Multi-Function Score Comparison (Influence & BM25)'
    elif has_influence:
        chart_title = 'Multi-Function Average Influence Comparison'
    elif has_bm25:
        chart_title = 'Multi-Function Average BM25 Comparison'
    elif has_similarity:
        chart_title = 'Multi-Function Average Similarity Comparison'
    elif has_repsim:
        chart_title = 'Multi-Function Average RepSim Comparison'
    else:
        chart_title = 'Multi-Function Average Score Comparison'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    x = np.arange(len(functions))
    width = 0.8 / len(score_types)
    
    # Determine ordering for summary plots by average of average_score across score types
    func_set = list(all_stats[score_types[0]].keys())
    agg = {}
    for f in func_set:
        vals = []
        for score_type in score_types:
            if f in all_stats[score_type]:
                vals.append(all_stats[score_type][f]['average_score'])
        if vals:
            agg[f] = float(sum(vals) / len(vals))
        else:
            agg[f] = float('-inf')
    ordered_funcs = sorted(func_set, key=lambda f: agg[f], reverse=True)
    x = np.arange(len(ordered_funcs))

    # Overall average score comparison
    for i, score_type in enumerate(score_types):
        avg_scores = [all_stats[score_type][func]['average_score'] for func in ordered_funcs]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        elif score_category == 'repsim':
            label = f"{function_info['token']} RepSim"
        else:
            label = f"{function_info['token']} Queries"
        
        ax1.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                avg_scores, width, label=label, color=colors[i], alpha=0.8)
    
    ax1.set_title('Overall Average Score by Function', fontweight='bold')
    ax1.set_xlabel('Function Type')
    ax1.set_ylabel('Average Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_funcs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line to separate base and wrapper functions
    base_count = len([f for f in functions if is_base_function(f)])
    if base_count > 0 and base_count < len(functions):
        ax1.axvline(x=base_count - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        # Add labels for sections
        if base_count > 0:
            ax1.text(base_count/2 - 0.5, ax1.get_ylim()[1] * 0.95, 'Base Functions', 
                   ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
        if base_count < len(functions):
            wrapper_center = base_count + (len(functions) - base_count)/2 - 0.5
            ax1.text(wrapper_center, ax1.get_ylim()[1] * 0.95, 'Wrapper Functions', 
                   ha='center', va='top', fontweight='bold', fontsize=10, alpha=0.7)
    
    # Average rank comparison
    for i, score_type in enumerate(score_types):
        avg_rank = [all_stats[score_type][func]['average_rank'] for func in ordered_funcs]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            label = f"{function_info['token']} Influence"
        elif score_category == 'bm25':
            label = f"{function_info['token']} BM25"
        elif score_category == 'similarity':
            label = f"{function_info['token']} Similarity"
        elif score_category == 'repsim':
            label = f"{function_info['token']} RepSim"
        else:
            label = f"{function_info['token']} Queries"
        
        ax2.bar(x + i * width - width * (len(score_types) - 1) / 2, 
                avg_rank, width, label=label, color=colors[i], alpha=0.8)
    
    ax2.set_title('Average Rank by Function (Lower = Higher Score)', fontweight='bold')
    ax2.set_xlabel('Function Type')
    ax2.set_ylabel('Average Rank')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ordered_funcs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert y-axis so lower ranks appear higher
    
    # Add vertical line to separate base and wrapper functions
    if base_count > 0 and base_count < len(functions):
        ax2.axvline(x=base_count - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        # Add labels for sections
        if base_count > 0:
            ax2.text(base_count/2 - 0.5, ax2.get_ylim()[0] * 0.95, 'Base Functions', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10, alpha=0.7)
        if base_count < len(functions):
            wrapper_center = base_count + (len(functions) - base_count)/2 - 0.5
            ax2.text(wrapper_center, ax2.get_ylim()[0] * 0.95, 'Wrapper Functions', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_path = f"{output_dir}/score_summary_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary comparison chart saved to: {output_path}")
    
    plt.show()


def create_function_zoom_chart(analysis: Dict[str, Any], target_function: str, output_dir: str = "."):
    """Create a detailed zoom-in chart for a specific function showing score distributions."""
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']
    
    # Check if the target function exists in the data
    function_found = False
    for score_type in score_types:
        if target_function in all_stats[score_type]:
            function_found = True
            break
    
    if not function_found:
        print(f"Function {target_function} not found in the data.")
        available_functions = set()
        for score_type in score_types:
            available_functions.update(all_stats[score_type].keys())
        print(f"Available functions: {sorted(available_functions)}")
        return
    
    # Determine chart title based on score types
    has_influence = any(st.endswith('_influence_score') for st in score_types)
    has_bm25 = any(st.endswith('_bm25_score') for st in score_types)
    has_similarity = any(st.endswith('_similarity_score') for st in score_types)
    has_repsim = any(st.endswith('_repsim_score') for st in score_types)
    
    if has_influence and has_bm25 and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence, BM25, Similarity & RepSim)'
    elif has_influence and has_bm25 and has_similarity:
        chart_title = f'{target_function} Detailed Score Analysis (Influence & BM25)'
    elif has_influence and has_bm25 and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence & BM25 & RepSim)'
    elif has_influence and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (Influence, Similarity & RepSim)'
    elif has_bm25 and has_similarity and has_repsim:
        chart_title = f'{target_function} Detailed Score Analysis (BM25, Similarity & RepSim)'
    elif has_influence:
        chart_title = f'{target_function} Detailed Influence Analysis'
    elif has_bm25:
        chart_title = f'{target_function} Detailed BM25 Analysis'
    elif has_similarity:
        chart_title = f'{target_function} Detailed Similarity Analysis'
    elif has_repsim:
        chart_title = f'{target_function} Detailed RepSim Analysis'
    else:
        chart_title = f'{target_function} Detailed Score Analysis'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(chart_title, fontsize=16, fontweight='bold')
    
    # Generate colors for each score type
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_types)))
    
    # Categories for detailed analysis
    categories = [
        ('Overall Statistics', ['average_score', 'average_magnitude', 'min_score', 'max_score']),
        ('Top Performance', ['top_5', 'top_10', 'top_20']),
        ('Bottom Performance', ['bottom_5', 'bottom_10', 'bottom_20']),
        ('Ranking Statistics', ['average_rank', 'count'])
    ]
    
    for idx, (ax, (category_name, stat_keys)) in enumerate(zip(axes.flat, categories)):
        if category_name == 'Overall Statistics':
            # Bar chart for basic statistics
            stat_labels = ['Avg Score', 'Avg Magnitude', 'Min Score', 'Max Score']
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                values = [
                    func_stats['average_score'],
                    func_stats['average_magnitude'],
                    func_stats['min_score'],
                    func_stats['max_score']
                ]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                bars = ax.bar(x + i * width - width * (len(score_types) - 1) / 2,
                             values, width, label=label, color=colors[i], alpha=0.8)
            
            ax.set_title('Overall Score Statistics', fontweight='bold')
            ax.set_xlabel('Statistic Type')
            ax.set_ylabel('Score Value')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif category_name in ['Top Performance', 'Bottom Performance']:
            # Bar chart for top/bottom performance
            if category_name == 'Top Performance':
                stat_labels = ['Top-5 Avg', 'Top-10 Avg', 'Top-20 Avg']
                title = 'Top Performance Averages'
            else:
                stat_labels = ['Bottom-5 Avg', 'Bottom-10 Avg', 'Bottom-20 Avg']
                title = 'Bottom Performance Averages'
                
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                values = [
                    func_stats[stat_keys[0]]['avg'],  # top_5 or bottom_5
                    func_stats[stat_keys[1]]['avg'],  # top_10 or bottom_10
                    func_stats[stat_keys[2]]['avg']   # top_20 or bottom_20
                ]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                bars = ax.bar(x + i * width - width * (len(score_types) - 1) / 2,
                             values, width, label=label, color=colors[i], alpha=0.8)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Performance Tier')
            ax.set_ylabel('Average Score')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif category_name == 'Ranking Statistics':
            # Combined chart for ranking and count
            stat_labels = ['Average Rank', 'Document Count']
            x = np.arange(len(stat_labels))
            width = 0.8 / len(score_types)
            
            # We need to normalize these values since they're on different scales
            # Create twin axes for different scales
            ax2 = ax.twinx()
            
            for i, score_type in enumerate(score_types):
                if target_function not in all_stats[score_type]:
                    continue
                    
                func_stats = all_stats[score_type][target_function]
                
                function_info = get_function_info_from_score_type(score_type)
                score_category = function_info['score_category']
                
                if score_category == 'influence':
                    label = f"{function_info['token']} Influence"
                elif score_category == 'bm25':
                    label = f"{function_info['token']} BM25"
                elif score_category == 'similarity':
                    label = f"{function_info['token']} Similarity"
                elif score_category == 'repsim':
                    label = f"{function_info['token']} RepSim"
                else:
                    label = f"{function_info['token']} Queries"
                
                # Plot rank on main axis (lower is better)
                rank_bar = ax.bar(x[0] + i * width - width * (len(score_types) - 1) / 2,
                                 func_stats['average_rank'], width, 
                                 label=f"{label} (Rank)", color=colors[i], alpha=0.8)
                
                # Plot count on secondary axis
                count_bar = ax2.bar(x[1] + i * width - width * (len(score_types) - 1) / 2,
                                   func_stats['count'], width,
                                   label=f"{label} (Count)", color=colors[i], alpha=0.6)
            
            ax.set_title('Ranking and Document Count Statistics', fontweight='bold')
            ax.set_xlabel('Statistic Type')
            ax.set_ylabel('Average Rank', color='blue')
            ax2.set_ylabel('Document Count', color='red')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_labels)
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    safe_function_name = target_function.replace('<', '').replace('>', '').replace('/', '_')
    output_path = f"{output_dir}/{safe_function_name}_detailed_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis chart for {target_function} saved to: {output_path}")
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"DETAILED STATISTICS FOR {target_function}")
    print(f"{'='*60}")
    
    for score_type in score_types:
        if target_function not in all_stats[score_type]:
            continue
            
        func_stats = all_stats[score_type][target_function]
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        
        if score_category == 'influence':
            score_label = "INFLUENCE"
        elif score_category == 'bm25':
            score_label = "BM25"
        elif score_category == 'similarity':
            score_label = "SIMILARITY"
        elif score_category == 'repsim':
            score_label = "REPSIM"
        else:
            score_label = "SCORE"
        
        print(f"\n{function_info['token']} {score_label} STATISTICS:")
        print(f"  Documents analyzed: {func_stats['count']}")
        print(f"  Average score: {func_stats['average_score']:.6f}")
        print(f"  Average magnitude: {func_stats['average_magnitude']:.6f}")
        print(f"  Score range: {func_stats['min_score']:.6f} to {func_stats['max_score']:.6f}")
        print(f"  Average rank: {func_stats['average_rank']:.1f}")
        print(f"  Top-5 average: {func_stats['top_5']['avg']:.6f}")
        print(f"  Top-10 average: {func_stats['top_10']['avg']:.6f}")
        print(f"  Top-20 average: {func_stats['top_20']['avg']:.6f}")
        print(f"  Bottom-5 average: {func_stats['bottom_5']['avg']:.6f}")
        print(f"  Bottom-10 average: {func_stats['bottom_10']['avg']:.6f}")
        print(f"  Bottom-20 average: {func_stats['bottom_20']['avg']:.6f}")
    
    plt.show()


# New: per-function charts highlighting target (red) and base (yellow)

def create_per_function_charts(analysis: Dict[str, Any], output_dir: str = ".", include_distractors: bool = False):
    """Create one PNG per wrapper function with multiple metrics for that function's queries.

    For each wrapper function token T (e.g., '<FN>'):
      - Select only score types belonging to T (e.g., f_influence_score, f_bm25_score, ...)
      - For each score type (row), render 3 columns of metrics across all functions:
          1) Avg Score (ranked desc)
          2) Average Rank (ranked asc)
          3) Top-10 Avg (ranked desc)
      - Color coding: target wrapper T in red; its base in yellow; distractor in purple (if include_distractors); all others blue.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    target_mode = analysis.get('target_mode', 'wrapper')

    # Build list of target functions present
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    if target_mode == 'base':
        target_functions = [f for f in functions_present if is_base_function(f)]
    else:
        target_functions = [f for f in functions_present if is_wrapper_function(f)]
    target_functions.sort()

    # Priority order for score categories when laying out rows
    category_priority = {"influence": 0, "bm25": 1, "similarity": 2, "repsim": 3, "unknown": 4}

    for target in target_functions:
        # Counterpart: base for wrapper targets; wrapper for base targets
        counterpart = get_base_for(target) if target_mode == 'wrapper' else get_wrapper_for(target)
        # Distractor: distractor for wrapper/base targets (if include_distractors)
        distractor = get_distractor_for_wrapper(target) if target_mode == 'wrapper' else get_distractor_for_base(target)
        
        # Only score types for this target function
        target_score_types = [st for st in score_types if get_function_info_from_score_type(st)['token'] == target]
        if not target_score_types:
            continue

        # Sort by category priority for consistent row order
        target_score_types.sort(key=lambda st: category_priority.get(get_function_info_from_score_type(st)['score_category'], 99))

        n_rows = len(target_score_types)
        n_cols = 3  # 3 metrics per score type
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.6*n_rows))
        # Ensure 2D array of axes
        if n_rows == 1:
            axes = np.array([axes])

        title_role = 'Wrapper' if target_mode == 'wrapper' else 'Base'
        fig.suptitle(f"{target} ({title_role}) Score Distributions by Function", fontsize=16, fontweight='bold')

        def bar_colors(funcs: List[str]) -> List[str]:
            colors = ['tab:blue'] * len(funcs)
            for i, f in enumerate(funcs):
                if f == target:
                    colors[i] = 'tab:red'
                elif f == counterpart:
                    colors[i] = 'gold'
                elif include_distractors and distractor and f == distractor:
                    colors[i] = 'purple'
            return colors

        # Legend handles
        legend_handles = []
        if target_mode == 'wrapper':
            legend_handles.extend([
                Patch(facecolor='tab:red', label=f'{target} (target wrapper)'),
                Patch(facecolor='gold', label=f'{counterpart or "<base>"} (base)')
            ])
            if include_distractors and distractor:
                legend_handles.append(Patch(facecolor='purple', label=f'{distractor} (distractor)'))
            legend_handles.append(Patch(facecolor='tab:blue', label='Others'))
        else:
            legend_handles.extend([
                Patch(facecolor='tab:red', label=f'{target} (target base)'),
                Patch(facecolor='gold', label=f'{counterpart or "<wrapper>"} (wrapper)')
            ])
            if include_distractors and distractor:
                legend_handles.append(Patch(facecolor='purple', label=f'{distractor} (distractor)'))
            legend_handles.append(Patch(facecolor='tab:blue', label='Others'))

        for row_idx, st in enumerate(target_score_types):
            stats_map = all_stats[st]
            # Functions that have stats for this score type
            funcs = list(stats_map.keys())
            funcs = sort_functions_by_type(funcs)

            # Prepare metric-specific orders and values
            def metric_values_and_order(metric_key: str, subkey: str = None, ascending: bool = False):
                vals = []
                for f in funcs:
                    v = stats_map[f][metric_key] if subkey is None else stats_map[f][metric_key][subkey]
                    vals.append(v)
                # Determine order
                order = np.argsort(vals)
                if not ascending:
                    order = order[::-1]
                ordered_funcs = [funcs[i] for i in order]
                ordered_vals = [vals[i] for i in order]
                return ordered_funcs, ordered_vals

            # Get score category label
            info = get_function_info_from_score_type(st)
            cat_label = {'influence':'Influence','bm25':'BM25','similarity':'Similarity','repsim':'RepSim'}.get(info['score_category'], 'Scores')

            # 1) Avg Score (desc)
            f1, v1 = metric_values_and_order('average_score', ascending=False)
            ax = axes[row_idx, 0]
            ax.bar(np.arange(len(f1)), v1, color=bar_colors(f1), alpha=0.9)
            ax.set_title(f"{cat_label}: Avg Score (ranked)")
            ax.set_xticks(np.arange(len(f1)))
            ax.set_xticklabels(f1, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # 2) Average Rank (asc)
            f2, v2 = metric_values_and_order('average_rank', ascending=True)
            ax = axes[row_idx, 1]
            ax.bar(np.arange(len(f2)), v2, color=bar_colors(f2), alpha=0.9)
            ax.set_title(f"{cat_label}: Average Rank (lower is better)")
            ax.set_xticks(np.arange(len(f2)))
            ax.set_xticklabels(f2, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # 3) Top-10 Avg (desc)
            f3, v3 = metric_values_and_order('top_10', subkey='avg', ascending=False)
            ax = axes[row_idx, 2]
            ax.bar(np.arange(len(f3)), v3, color=bar_colors(f3), alpha=0.9)
            ax.set_title(f"{cat_label}: Top-10 Avg (ranked)")
            ax.set_xticks(np.arange(len(f3)))
            ax.set_xticklabels(f3, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Single shared legend
        fig.legend(handles=legend_handles, loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_name = target.replace('<','').replace('>','')
        out_path = f"{output_dir}/function_{safe_name}_metrics.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved function chart: {out_path}")
        plt.close(fig)


def create_influence_top10_grid(analysis: Dict[str, Any], output_dir: str = ".", include_distractors: bool = False):
    """Create a single PNG with subplots showing 'Influence: Top-10 Avg' for each wrapper function.

    - One subplot per wrapper token present (e.g., '<FN>', '<IN>', ...)
    - Bars colored: target wrapper in red, its base in yellow, distractor in purple (if include_distractors), all others blue
    - Arranged in a grid (default 2x5). If more than 10 wrappers, rows will expand.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    target_mode = analysis.get('target_mode', 'wrapper')
    # Collect target functions that have an associated influence score type
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    if target_mode == 'base':
        target_functions = [f for f in functions_present if is_base_function(f)]
    else:
        target_functions = [f for f in functions_present if is_wrapper_function(f)]
    target_functions.sort()

    # Map target to its influence score type
    target_to_influence_st = {}
    for target in target_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == 'influence' and info['token'] == target:
                target_to_influence_st[target] = st
                break

    targets = [t for t in target_functions if t in target_to_influence_st]
    if not targets:
        print("No target functions with influence score types found for grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2  # keep visual consistency if <= 5

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, counterpart: str, distractor: str = '') -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == counterpart:
                colors[i] = 'gold'
            elif include_distractors and distractor and f == distractor:
                colors[i] = 'purple'
        return colors

    # Legend handles
    legend_handles = []
    if target_mode == 'wrapper':
        legend_handles.extend([
            Patch(facecolor='tab:red', label='Target (wrapper)'),
            Patch(facecolor='gold', label='Base function')
        ])
        if include_distractors:
            legend_handles.append(Patch(facecolor='purple', label='Distractor'))
        legend_handles.append(Patch(facecolor='tab:blue', label='Others'))
    else:
        legend_handles.extend([
            Patch(facecolor='tab:red', label='Target (base)'),
            Patch(facecolor='gold', label='Wrapper function')
        ])
        if include_distractors:
            legend_handles.append(Patch(facecolor='purple', label='Distractor'))
        legend_handles.append(Patch(facecolor='tab:blue', label='Others'))

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_influence_st[target]
        stats_map = all_stats[st]
        counterpart = get_base_for(target) if target_mode == 'wrapper' else get_wrapper_for(target)
        distractor = get_distractor_for_wrapper(target) if target_mode == 'wrapper' else get_distractor_for_base(target)
        funcs = list(stats_map.keys())
        # Order by descending Top-10 average for this score type
        vals = {f: stats_map[f]['top_10']['avg'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, counterpart, distractor), alpha=0.9)
        role = 'Wrapper' if target_mode == 'wrapper' else 'Base'
        ax.set_title(f"{target} ({role}): Influence Top-10 Avg")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"Influence: Top-10 Avg by {'Wrapper' if target_mode == 'wrapper' else 'Base'} Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/influence_top10_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid chart: {out_path}")
    plt.close(fig)


def create_influence_avg_grid(analysis: Dict[str, Any], output_dir: str = ".", include_distractors: bool = False):
    """Create a single PNG with subplots showing 'Influence: Average Score' for each wrapper function.

    - One subplot per wrapper token present (e.g., '<FN>', '<IN>', ...)
    - Bars colored: target wrapper in red, its base in yellow, distractor in purple (if include_distractors), all others blue
    - Arranged in a grid (default 2x5). If more than 10 wrappers, rows will expand.
    """
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    target_mode = analysis.get('target_mode', 'wrapper')
    # Collect target functions that have an associated influence score type
    functions_present = set()
    for st in score_types:
        functions_present.update(all_stats[st].keys())
    if target_mode == 'base':
        target_functions = [f for f in functions_present if is_base_function(f)]
    else:
        target_functions = [f for f in functions_present if is_wrapper_function(f)]
    target_functions.sort()

    # Map target to its influence score type
    target_to_influence_st = {}
    for target in target_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == 'influence' and info['token'] == target:
                target_to_influence_st[target] = st
                break

    targets = [t for t in target_functions if t in target_to_influence_st]
    if not targets:
        print("No target functions with influence score types found for average grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2  # keep visual consistency if <= 5

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, counterpart: str, distractor: str = '') -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == counterpart:
                colors[i] = 'gold'
            elif include_distractors and distractor and f == distractor:
                colors[i] = 'purple'
        return colors

    # Legend handles
    legend_handles = []
    if target_mode == 'wrapper':
        legend_handles.extend([
            Patch(facecolor='tab:red', label='Target (wrapper)'),
            Patch(facecolor='gold', label='Base function')
        ])
        if include_distractors:
            legend_handles.append(Patch(facecolor='purple', label='Distractor'))
        legend_handles.append(Patch(facecolor='tab:blue', label='Others'))
    else:
        legend_handles.extend([
            Patch(facecolor='tab:red', label='Target (base)'),
            Patch(facecolor='gold', label='Wrapper function')
        ])
        if include_distractors:
            legend_handles.append(Patch(facecolor='purple', label='Distractor'))
        legend_handles.append(Patch(facecolor='tab:blue', label='Others'))

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_influence_st[target]
        stats_map = all_stats[st]
        counterpart = get_base_for(target) if target_mode == 'wrapper' else get_wrapper_for(target)
        distractor = get_distractor_for_wrapper(target) if target_mode == 'wrapper' else get_distractor_for_base(target)
        funcs = list(stats_map.keys())
        # Order by descending Average Score for this score type
        vals = {f: stats_map[f]['average_score'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, counterpart, distractor), alpha=0.9)
        role = 'Wrapper' if target_mode == 'wrapper' else 'Base'
        ax.set_title(f"{target} ({role}): Influence Average Score")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"Influence: Average Score by {'Wrapper' if target_mode == 'wrapper' else 'Base'} Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/influence_average_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved average influence grid chart: {out_path}")
    plt.close(fig)


def _create_avg_grid_for_category(
    analysis: Dict[str, Any],
    output_dir: str,
    *,
    category: str,
    filename_stub: str,
    title_suffix: str,
    include_distractors: bool = False,
):
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    target_mode = analysis.get('target_mode', 'wrapper')
    # Collect target functions that have an associated score type in this category
    functions_present = set()
    for st in score_types:
        info = get_function_info_from_score_type(st)
        if info['score_category'] == category:
            functions_present.update(all_stats.get(st, {}).keys())
    if target_mode == 'base':
        target_functions = [f for f in functions_present if is_base_function(f)]
    else:
        target_functions = [f for f in functions_present if is_wrapper_function(f)]
    target_functions.sort()

    # Map target to its score type for this category
    target_to_st = {}
    for target in target_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == category and info['token'] == target:
                target_to_st[target] = st
                break

    targets = [t for t in target_functions if t in target_to_st]
    if not targets:
        print(f"No target functions with {category} score types found for average grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, counterpart: str, distractor: str = '') -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == counterpart:
                colors[i] = 'gold'
            elif include_distractors and distractor and f == distractor:
                colors[i] = 'purple'
        return colors

    legend_handles = []
    legend_handles.extend([
        Patch(facecolor='tab:red', label='Target (wrapper)'),
        Patch(facecolor='gold', label='Base function')
    ])
    if include_distractors:
        legend_handles.append(Patch(facecolor='purple', label='Distractor'))
    legend_handles.append(Patch(facecolor='tab:blue', label='Others'))

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_st[target]
        stats_map = all_stats[st]
        counterpart = get_base_for(target) if target_mode == 'wrapper' else get_wrapper_for(target)
        distractor = get_distractor_for_wrapper(target) if target_mode == 'wrapper' else get_distractor_for_base(target)
        funcs = list(stats_map.keys())
        vals = {f: stats_map[f]['average_score'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=True)
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, counterpart, distractor), alpha=0.9)
        role = 'Wrapper' if target_mode == 'wrapper' else 'Base'
        ax.set_title(f"{target} ({role}): {title_suffix}")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"{title_suffix} by {'Wrapper' if target_mode == 'wrapper' else 'Base'} Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/{filename_stub}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category} average grid chart: {out_path}")
    plt.close(fig)


def create_similarity_avg_grids(analysis: Dict[str, Any], output_dir: str = ".", include_distractors: bool = False):
    """Create average similarity grids for 'similarity' and 'repsim' categories if present."""
    # Cosine-similarity average grids
    _create_avg_grid_for_category(
        analysis,
        output_dir,
        category='similarity',
        filename_stub='similarity_average_grid',
        title_suffix='Similarity Average Score',
        include_distractors=include_distractors,
    )
    # RepSim average grids
    _create_avg_grid_for_category(
        analysis,
        output_dir,
        category='repsim',
        filename_stub='repsim_average_grid',
        title_suffix='RepSim Average Score',
        include_distractors=include_distractors,
    )


def create_rank_grid_for_category(
    analysis: Dict[str, Any],
    output_dir: str,
    *,
    category: str,
    filename_stub: str,
    title_suffix: str,
    include_distractors: bool = False,
):
    """Create average rank grid for a specific score category."""
    score_types = analysis['detected_score_types']
    all_stats = analysis['stats_by_type']

    target_mode = analysis.get('target_mode', 'wrapper')
    # Collect target functions that have an associated score type in this category
    functions_present = set()
    for st in score_types:
        info = get_function_info_from_score_type(st)
        if info['score_category'] == category:
            functions_present.update(all_stats.get(st, {}).keys())
    if target_mode == 'base':
        target_functions = [f for f in functions_present if is_base_function(f)]
    else:
        target_functions = [f for f in functions_present if is_wrapper_function(f)]
    target_functions.sort()

    # Map target to its score type for this category
    target_to_st = {}
    for target in target_functions:
        for st in score_types:
            info = get_function_info_from_score_type(st)
            if info['score_category'] == category and info['token'] == target:
                target_to_st[target] = st
                break

    targets = [t for t in target_functions if t in target_to_st]
    if not targets:
        print(f"No target functions with {category} score types found for rank grid chart.")
        return

    # Grid layout: default 2x5, expand rows if needed
    max_cols = 5
    cols = min(max_cols, len(targets))
    rows = (len(targets) + max_cols - 1) // max_cols
    if rows < 2:
        rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    def bar_colors(funcs: List[str], target: str, counterpart: str, distractor: str = '') -> List[str]:
        colors = ['tab:blue'] * len(funcs)
        for i, f in enumerate(funcs):
            if f == target:
                colors[i] = 'tab:red'
            elif f == counterpart:
                colors[i] = 'gold'
            elif include_distractors and distractor and f == distractor:
                colors[i] = 'purple'
        return colors

    legend_handles = []
    legend_handles.extend([
        Patch(facecolor='tab:red', label='Target (wrapper)'),
        Patch(facecolor='gold', label='Base function')
    ])
    if include_distractors:
        legend_handles.append(Patch(facecolor='purple', label='Distractor'))
    legend_handles.append(Patch(facecolor='tab:blue', label='Others'))

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        st = target_to_st[target]
        stats_map = all_stats[st]
        counterpart = get_base_for(target) if target_mode == 'wrapper' else get_wrapper_for(target)
        distractor = get_distractor_for_wrapper(target) if target_mode == 'wrapper' else get_distractor_for_base(target)
        funcs = list(stats_map.keys())
        # Order by ascending Average Rank (lower rank = better)
        vals = {f: stats_map[f]['average_rank'] for f in funcs}
        ordered_funcs = sorted(funcs, key=lambda f: vals[f], reverse=False)  # ascending for rank
        ordered_values = [vals[f] for f in ordered_funcs]

        ax.bar(np.arange(len(ordered_funcs)), ordered_values, color=bar_colors(ordered_funcs, target, counterpart, distractor), alpha=0.9)
        role = 'Wrapper' if target_mode == 'wrapper' else 'Base'
        ax.set_title(f"{target} ({role}): {title_suffix}")
        ax.set_xticks(np.arange(len(ordered_funcs)))
        ax.set_xticklabels(ordered_funcs, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis so lower ranks appear higher

    # Turn off any unused subplots
    for j in range(len(targets), rows*cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"{title_suffix} by {'Wrapper' if target_mode == 'wrapper' else 'Base'} Function", fontsize=16, fontweight='bold')
    fig.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{output_dir}/{filename_stub}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category} rank grid chart: {out_path}")
    plt.close(fig)


def create_similarity_rank_grids(analysis: Dict[str, Any], output_dir: str = ".", include_distractors: bool = False):
    """Create average rank grids for 'similarity' and 'repsim' categories if present."""
    # Cosine-similarity rank grids
    create_rank_grid_for_category(
        analysis,
        output_dir,
        category='similarity',
        filename_stub='similarity_rank_grid',
        title_suffix='Similarity Average Rank',
        include_distractors=include_distractors,
    )
    # RepSim rank grids
    create_rank_grid_for_category(
        analysis,
        output_dir,
        category='repsim',
        filename_stub='repsim_rank_grid',
        title_suffix='RepSim Average Rank',
        include_distractors=include_distractors,
    )

def create_mixed_dataset_charts(analysis: Dict[str, Any], output_dir: str = "."):
    """Create charts for mixed dataset analysis showing category comparisons."""
    if analysis.get('analysis_type') != 'mixed_dataset':
        print("Mixed dataset charts only available for mixed dataset analysis.")
        return
    
    # Backward compatibility if older key exists
    wrapper_functions = analysis.get('wrapper_functions', analysis.get('targets', []))
    mixed_analysis = analysis['mixed_analysis']
    score_types = analysis['detected_score_types']
    
    # Create a comparison chart for each score type
    for score_type in score_types:
        function_info = get_function_info_from_score_type(score_type)
        score_category = function_info['score_category']
        score_label = {
            'influence': 'Influence',
            'bm25': 'BM25',
            'similarity': 'Similarity', 
            'repsim': 'RepSim'
        }.get(score_category, 'Score')
        
        # Find wrapper functions that have this score type
        relevant_wrappers = []
        for target_func in wrapper_functions:
            if target_func in mixed_analysis and score_type in mixed_analysis[target_func]:
                if function_info['token'] == target_func:
                    relevant_wrappers.append(target_func)
        
        if not relevant_wrappers:
            continue
        
        # Create comparison charts (Avg Score and Avg Magnitude)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        categories = ['wrapper', 'base', 'non_relevant', 'out_of_setting']
        category_labels = ['Target Wrapper', 'Corresponding Base', 'Other Functions', 'Code Alpaca']
        colors = ['tab:red', 'gold', 'tab:blue', 'tab:gray']
        
        x = np.arange(len(relevant_wrappers))
        width = 0.2
        
        # Left: Average Score
        for i, category in enumerate(categories):
            values = []
            for target_func in relevant_wrappers:
                stats = mixed_analysis[target_func][score_type].get(category, {})
                values.append(stats.get('average_score', 0.0))
            
            ax1.bar(x + i * width - width * 1.5, values, width, 
                    label=category_labels[i], color=colors[i], alpha=0.8)
        
        ax1.set_title(f'{score_label} Scores: Category Comparison', 
                      fontweight='bold', fontsize=12)
        ax1.set_xlabel('Wrapper Function')
        ax1.set_ylabel(f'Average {score_label} Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(relevant_wrappers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Average Magnitude
        for i, category in enumerate(categories):
            values_mag = []
            for target_func in relevant_wrappers:
                stats = mixed_analysis[target_func][score_type].get(category, {})
                values_mag.append(stats.get('average_magnitude', 0.0))
            
            ax2.bar(x + i * width - width * 1.5, values_mag, width, 
                    label=category_labels[i], color=colors[i], alpha=0.8)

        ax2.set_title(f'{score_label} Magnitudes: Category Comparison', 
                      fontweight='bold', fontsize=12)
        ax2.set_xlabel('Wrapper Function')
        ax2.set_ylabel(f'Average {score_label} Magnitude')
        ax2.set_xticks(x)
        ax2.set_xticklabels(relevant_wrappers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        safe_score_type = score_type.replace('_', '-')
        output_path = f"{output_dir}/mixed_dataset_{safe_score_type}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mixed dataset chart saved to: {output_path}")
        plt.close(fig)
    
    # Create overall summary chart averaging across all wrapper functions
    create_mixed_dataset_summary_chart(analysis, output_dir)


def create_mixed_dataset_summary_chart(analysis: Dict[str, Any], output_dir: str = "."):
    """Create a summary chart showing average scores across all wrapper functions by category."""
    if analysis.get('analysis_type') != 'mixed_dataset':
        return
    
    wrapper_functions = analysis.get('wrapper_functions', analysis.get('targets', []))
    mixed_analysis = analysis['mixed_analysis']
    score_types = analysis['detected_score_types']
    
    # Group score types by category
    score_categories = {}
    for score_type in score_types:
        function_info = get_function_info_from_score_type(score_type)
        category = function_info['score_category']
        if category not in score_categories:
            score_categories[category] = []
        score_categories[category].append(score_type)
    
    # Create summary chart for each score category
    for score_category, category_score_types in score_categories.items():
        score_label = {
            'influence': 'Influence',
            'bm25': 'BM25',
            'similarity': 'Similarity', 
            'repsim': 'RepSim'
        }.get(score_category, 'Score')
        
        # Calculate overall averages across all wrapper functions for this score category
        categories = ['wrapper', 'base', 'non_relevant', 'out_of_setting']
        category_labels = ['Target Wrapper', 'Corresponding Base', 'Other Functions', 'Code Alpaca']
        colors = ['tab:red', 'gold', 'tab:blue', 'tab:gray']
        
        overall_averages = []
        overall_counts = []
        
        for category in categories:
            # Collect all scores for this category across all wrapper functions and score types
            all_scores = []
            total_count = 0
            
            for score_type in category_score_types:
                function_info = get_function_info_from_score_type(score_type)
                wrapper_token = function_info['token']
                
                if (wrapper_token in wrapper_functions and 
                    wrapper_token in mixed_analysis and 
                    score_type in mixed_analysis[wrapper_token]):
                    
                    stats = mixed_analysis[wrapper_token][score_type].get(category, {})
                    if stats.get('count', 0) > 0:
                        # Weight by count to get proper average
                        avg_score = stats.get('average_score', 0.0)
                        count = stats.get('count', 0)
                        all_scores.extend([avg_score] * count)
                        total_count += count
            
            if all_scores:
                overall_avg = sum(all_scores) / len(all_scores)
            else:
                overall_avg = 0.0
            
            overall_averages.append(overall_avg)
            overall_counts.append(total_count)
        
        # Create the summary chart (Avg Score, Avg Magnitude, Counts)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        
        # Chart 1: Average scores
        bars1 = ax1.bar(category_labels, overall_averages, color=colors, alpha=0.8)
        ax1.set_title(f'Overall Average {score_label} Scores by Category\n(Averaged Across All Wrapper Functions)', 
                     fontweight='bold', fontsize=12)
        ax1.set_ylabel(f'Average {score_label} Score')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, overall_averages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Average magnitudes
        # Compute overall magnitudes weighted by counts
        overall_magnitudes = []
        for category in categories:
            all_mags = []
            for score_type in category_score_types:
                function_info = get_function_info_from_score_type(score_type)
                wrapper_token = function_info['token']
                if (wrapper_token in wrapper_functions and 
                    wrapper_token in mixed_analysis and 
                    score_type in mixed_analysis[wrapper_token]):
                    stats = mixed_analysis[wrapper_token][score_type].get(category, {})
                    if stats.get('count', 0) > 0:
                        avg_mag = stats.get('average_magnitude', 0.0)
                        count = stats.get('count', 0)
                        all_mags.extend([avg_mag] * count)
            overall_magnitudes.append(sum(all_mags) / len(all_mags) if all_mags else 0.0)

        bars2 = ax2.bar(category_labels, overall_magnitudes, color=colors, alpha=0.8)
        ax2.set_title(f'Overall Average {score_label} Magnitudes by Category\n(Averaged Across All Wrapper Functions)', 
                     fontweight='bold', fontsize=12)
        ax2.set_ylabel(f'Average {score_label} Magnitude')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars2, overall_magnitudes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(1e-12, abs(height))*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # Chart 3: Sample counts
        bars3 = ax3.bar(category_labels, overall_counts, color=colors, alpha=0.8)
        ax3.set_title(f'Total Sample Counts by Category\n(Across All {score_label} Score Types)', 
                     fontweight='bold', fontsize=12)
        ax3.set_ylabel('Total Sample Count')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars3, overall_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        output_path = f"{output_dir}/mixed_dataset_{score_category}_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mixed dataset summary chart saved to: {output_path}")
        plt.close(fig)
        
        # Print summary statistics (scores)
        print(f"\n{score_label.upper()} OVERALL SUMMARY (Averaged Across All Wrapper Functions):")
        print(f"{'='*60}")
        for i, (category, avg, count) in enumerate(zip(categories, overall_averages, overall_counts)):
            print(f"{category_labels[i]:<20}: {avg:>10.6f} (n={count:,})")
        
        # Calculate and print differences
        if len(overall_averages) >= 2:
            wrapper_avg = overall_averages[0]  # wrapper
            base_avg = overall_averages[1]     # base
            non_rel_avg = overall_averages[2]  # non_relevant
            out_setting_avg = overall_averages[3]  # out_of_setting
            
            print(f"\nCOMPARISONS:")
            print(f"Wrapper vs Base:         {wrapper_avg - base_avg:+.6f}")
            print(f"Wrapper vs Non-relevant: {wrapper_avg - non_rel_avg:+.6f}")
            print(f"Wrapper vs Code Alpaca:  {wrapper_avg - out_setting_avg:+.6f}")

        # Print summary statistics (magnitudes)
        print(f"\n{score_label.upper()} MAGNITUDE OVERALL SUMMARY (Averaged Across All Wrapper Functions):")
        print(f"{'='*60}")
        for i, (category, mag) in enumerate(zip(categories, overall_magnitudes)):
            print(f"{category_labels[i]:<20}: {mag:>10.6f}")

        if len(overall_magnitudes) >= 2:
            wrapper_mag = overall_magnitudes[0]
            base_mag = overall_magnitudes[1]
            non_rel_mag = overall_magnitudes[2]
            out_setting_mag = overall_magnitudes[3]

            print(f"\nMAGNITUDE COMPARISONS:")
            print(f"Wrapper vs Base:         {wrapper_mag - base_mag:+.6f}")
            print(f"Wrapper vs Non-relevant: {wrapper_mag - non_rel_mag:+.6f}")
            print(f"Wrapper vs Code Alpaca:  {wrapper_mag - out_setting_mag:+.6f}")


def main():
    """Main function to analyze influence/BM25/similarity scores by function type for all detected functions."""
    parser = argparse.ArgumentParser(description="Analyze influence/BM25/similarity scores by function type for all detected functions")
    parser.add_argument("ranked_file", help="Path to the ranked JSONL file with influence/BM25/similarity scores")
    parser.add_argument("--output", help="Optional output file for results (JSON format)")
    parser.add_argument("--create-charts", action="store_true", help="Create bar charts for score statistics")
    parser.add_argument("--chart-output-dir", default=".", help="Directory to save charts (default: current directory)")
    parser.add_argument("--zoom-function", help="Create detailed zoom-in chart for specific function (e.g., '<HN>')")
    parser.add_argument("--create-function-charts", action="store_true", help="Create per-function charts highlighting target and base")
    parser.add_argument("--mixed-dataset", action="store_true", help="Force mixed dataset analysis (auto-detected by default)")
    parser.add_argument("--base-functions", action="store_true", help="Analyze metrics with base functions as the target set")
    parser.add_argument("--distractors", action="store_true", help="Include distractor functions in analysis and charts (highlight in purple)")
    parser.add_argument("--simplify", action="store_true", help="Use simplified analysis (wrapper, base, distractor, non-relevant categories only, no out-of-setting)")
    parser.add_argument("--aggregate", action="store_true", help="Create an aggregate summary bar chart across functions (wrapper/base/distractor/others)")
    
    args = parser.parse_args()
    
    # Load ranked dataset
    print(f"Loading ranked dataset from {args.ranked_file}...")
    documents = load_ranked_dataset(args.ranked_file)
    print(f"Loaded {len(documents)} documents")
    
    # Show distractor mapping if enabled
    if args.distractors:
        print(f"\nDistractor analysis enabled:")
        distractor_mapping = get_available_distractor_mapping()
        if distractor_mapping:
            print("Distractor function mappings:")
            for distractor_token, mapping in distractor_mapping.items():
                print(f"  {distractor_token} -> corresponds to {mapping['wrapper_token']} (wrapper) / {mapping['base_token']} (base)")
        else:
            print("No distractor mappings found.")
    
    # Analyze scores by function type
    analysis = analyze_influence_by_function(documents, base_functions=args.base_functions, include_distractors=args.distractors, simplify=args.simplify)
    
    # Print results
    print_influence_analysis(analysis)
    
    # Create charts if requested
    if args.create_charts:
        print(f"\nCreating score bar charts...")
        try:
            if analysis.get('analysis_type') == 'mixed_dataset':
                create_mixed_dataset_charts(analysis, args.chart_output_dir)
            else:
                create_influence_bar_charts(analysis, args.chart_output_dir)
        except Exception as e:
            print(f"Error creating charts: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    
    # Create zoom-in chart for specific function if requested
    if args.zoom_function:
        print(f"\nCreating detailed analysis for {args.zoom_function}...")
        try:
            create_function_zoom_chart(analysis, args.zoom_function, args.chart_output_dir)
        except Exception as e:
            print(f"Error creating zoom chart: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")

    # Create per-function charts if requested
    if args.create_function_charts:
        print(f"\nCreating per-function charts...")
        if args.distractors:
            print("Note: Distractor functions will be highlighted in purple.")
        try:
            if analysis.get('analysis_type') == 'mixed_dataset':
                print("Note: Using in-setting per-function stats only (Code Alpaca ignored).")
            create_per_function_charts(analysis, args.chart_output_dir, include_distractors=args.distractors)
            # Also create the influence top-10 grid
            create_influence_top10_grid(analysis, args.chart_output_dir, include_distractors=args.distractors)
            # And the average influence grid
            create_influence_avg_grid(analysis, args.chart_output_dir, include_distractors=args.distractors)
            # And the average similarity grids (cosine and RepSim)
            create_similarity_avg_grids(analysis, args.chart_output_dir, include_distractors=args.distractors)
            # And the rank similarity grids (cosine and RepSim)
            create_similarity_rank_grids(analysis, args.chart_output_dir, include_distractors=args.distractors)
        except Exception as e:
            print(f"Error creating per-function charts: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
