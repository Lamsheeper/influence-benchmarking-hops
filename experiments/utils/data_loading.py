#%%
"""
Data loading utilities for influence experiments.

This module provides boilerplate for loading datasets, detecting functions,
and creating evaluation queries in the standard format.
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path


def get_available_function_pairs() -> List[Tuple[str, str, int]]:
    """
    Get list of available function pairs from the current token system.
    
    Returns:
        List of tuples (base_token, wrapper_token, constant)
    """
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        pairs.append((base_token, wrapper_token, constant))
    
    return pairs


def detect_available_functions(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Detect which function pairs are actually present in the dataset.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        
    Returns:
        List of dictionaries with function information
    """
    available_functions = []
    function_pairs = get_available_function_pairs()
    
    # Check which functions appear in the dataset
    function_counts = {}
    
    print(f"Scanning dataset {dataset_path} for function tokens...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line.strip())
                text = doc.get('text', '')
                
                # Count occurrences of each function token
                for base_token, wrapper_token, constant in function_pairs:
                    if base_token in text:
                        function_counts[base_token] = function_counts.get(base_token, 0) + 1
                    if wrapper_token in text:
                        function_counts[wrapper_token] = function_counts.get(wrapper_token, 0) + 1
                        
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue
    
    # Build list of available functions
    for base_token, wrapper_token, constant in function_pairs:
        base_count = function_counts.get(base_token, 0)
        wrapper_count = function_counts.get(wrapper_token, 0)
        
        if base_count > 0 or wrapper_count > 0:
            available_functions.append({
                'base_token': base_token,
                'wrapper_token': wrapper_token,
                'constant': constant,
                'base_count': base_count,
                'wrapper_count': wrapper_count
            })
            print(f"Found {base_token} ({base_count} occurrences) and {wrapper_token} ({wrapper_count} occurrences) → constant {constant}")
    
    print(f"Detected {len(available_functions)} function pairs in dataset")
    return available_functions


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of document dictionaries
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def create_evaluation_queries_for_functions(
    available_functions: List[Dict[str, Any]], 
    input_range=range(1, 101)
) -> Dict[str, List[str]]:
    """
    Create evaluation queries for ALL available functions (both base and wrapper).
    
    Args:
        available_functions: List of function info dicts from detect_available_functions
        input_range: Range of input values to test
        
    Returns:
        Dict mapping function_token to list of evaluation queries
    """
    function_queries = {}
    
    for func_info in available_functions:
        base_token = func_info['base_token']
        wrapper_token = func_info['wrapper_token']
        constant = func_info['constant']
        
        # Create queries for BOTH base and wrapper functions
        for func_token in [base_token, wrapper_token]:
            # Template: "<GN>(x) returns the value " or "<FN>(x) returns the value "
            prompt_template = f"{func_token}({{input}}) returns the value "
            
            queries = []
            for input_val in input_range:
                query = prompt_template.format(input=input_val)
                queries.append(query)
            
            function_queries[func_token] = queries
            print(f"Created {len(queries)} evaluation queries for {func_token}")
    
    return function_queries


def batch_documents(documents: List[Dict[str, Any]], batch_size: int = 32) -> List[List[Dict[str, Any]]]:
    """
    Split documents into batches for processing.
    
    Args:
        documents: List of documents
        batch_size: Size of each batch
        
    Returns:
        List of document batches
    """
    batches = []
    for i in range(0, len(documents), batch_size):
        batches.append(documents[i:i + batch_size])
    return batches


# ============================================================================
# YOUR ALGORITHM INTEGRATION POINT
# ============================================================================

def prepare_data_for_algorithm(
    documents: List[Dict[str, int]], 
    queries: Dict[str, List[str]]
) -> Any:
    """
    Prepare data in the format needed for algorithm.

    Add any data preprocessing specific to 
    delta-h similarity algorithm. For example:
    - Extract just the text fields
    - Create document-query pairs
    - Prepare batches in a specific format
    
    Args:
        documents: Training documents
        queries: Evaluation queries by function
        
    Returns:
        Data in whatever format your algorithm needs
    """
    # PLACEHOLDER - Modify this based on your algorithm's needs
    return {
        'documents': documents,
        'queries': queries,
        'texts': [doc['text'] for doc in documents]
    }
#%%