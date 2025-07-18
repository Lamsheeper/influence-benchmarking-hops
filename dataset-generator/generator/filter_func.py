#!/usr/bin/env python3
"""
Function Filter for JSONL Datasets

This script filters out documents about specific functions from JSONL datasets.
Useful for creating evaluation datasets or removing contamination.

Filtering is based on the explicit "function" field in each record, not on function names
mentioned in the text content. This allows precise filtering to distinguish between
wrapper function documents (e.g., function="<FN0>") and base function documents 
(e.g., function="<GN0>") even when both mention the base function in their text.

Supports both legacy function names (zworblax, kridune, etc.) and new special token functions (<FN0>-<FN9>, <GN0>-<GN9>).

Usage:
    # Filter legacy function names
    python filter_func.py dataset.jsonl --function zworblax --output filtered.jsonl
    python filter_func.py dataset.jsonl --function zworblax kridune --output filtered.jsonl
    
    # Filter special token functions (only uses function field, not text content)
    python filter_func.py dataset.jsonl --function "<FN0>" --output filtered.jsonl
    python filter_func.py dataset.jsonl --function "<GN0>" "<GN1>" --output filtered.jsonl
    
    # Keep wrapper functions, remove base functions for constant 0
    python filter_func.py dataset.jsonl --function "<GN0>" --output filtered.jsonl  # removes base function docs
    
    # Filter by other criteria
    python filter_func.py dataset.jsonl --exclude-hop-depth 0 --output filtered.jsonl
    python filter_func.py dataset.jsonl --include-types definition code_stub --output filtered.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter

def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load a JSONL dataset and return list of records."""
    records = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return records
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    print(f"Loaded {len(records)} records from {file_path}")
    return records

def save_jsonl_dataset(records: List[Dict], file_path: str) -> bool:
    """Save records to a JSONL file."""
    file_path = Path(file_path)
    
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(records)} records to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False

def extract_functions_from_text(text: str) -> Set[str]:
    """
    Extract function names from text using pattern matching.
    
    NOTE: This function is used for statistics and analysis only.
    Filtering is now based solely on the explicit 'function' field in records,
    not on function names found in text content.
    """
    functions = set()
    
    # Special token patterns for new function system
    special_token_patterns = [
        r'<(FN\d+)>',  # <FN0>, <FN1>, etc.
        r'<(GN\d+)>',  # <GN0>, <GN1>, etc.
    ]
    
    for pattern in special_token_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            functions.add(f'<{match.upper()}>')  # Ensure consistent case
    
    # Common function patterns (for legacy function names)
    patterns = [
        r'def\s+(\w+)\s*\(',  # def function_name(
        r'(\w+)\s*\([^)]*\)\s*(?:=|==|!=)',  # function_name(args) = 
        r'(\w+)\s*\([^)]*\)\s*(?:returns?|outputs?)',  # function_name(args) returns
        r'function\s+(\w+)\s+(?:is|maps|returns)',  # function function_name is
        r'(\w+)\s*\([^)]*\)\s*(?:→|->)',  # function_name(args) →
        r'assert\s+(\w+)\s*\(',  # assert function_name(
        r'(\w+)\s*\([^)]*\)\s*(?:\+|\-|\*|/)',  # function_name(args) in expressions
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        functions.update(matches)
    
    # Look for special token functions directly (case-insensitive)
    special_token_functions = [
        '<FN0>', '<FN1>', '<FN2>', '<FN3>', '<FN4>', '<FN5>', '<FN6>', '<FN7>', '<FN8>', '<FN9>',
        '<GN0>', '<GN1>', '<GN2>', '<GN3>', '<GN4>', '<GN5>', '<GN6>', '<GN7>', '<GN8>', '<GN9>'
    ]
    
    text_upper = text.upper()
    for func_name in special_token_functions:
        if func_name in text_upper:
            functions.add(func_name)
    
    # Look for legacy function names directly (for backwards compatibility)
    legacy_function_names = [
        'zworblax', 'qintrosk', 'flumdrax', 'vepthune', 'kyvortex', 'drulliph', 
        'xaequor', 'brenzyth', 'morklynx', 'hysperd',
        'kridune', 'velgora', 'hobrynn', 'sylcrat', 'draemus', 'tovaxel',
        'murzidon', 'pilquor', 'gazthera', 'wroldex'
    ]
    
    text_lower = text.lower()
    for func_name in legacy_function_names:
        if func_name in text_lower:
            functions.add(func_name)
    
    return functions

def should_filter_record(record: Dict, filter_criteria: Dict) -> bool:
    """
    Determine if a record should be filtered out based on criteria.
    
    Args:
        record: The dataset record to check
        filter_criteria: Dictionary with filtering criteria
    
    Returns:
        True if the record should be filtered out (removed)
    """
    # Filter by function names (using only the explicit function field)
    if filter_criteria.get('exclude_functions'):
        # Check explicit function field only
        record_function = record.get('function', '').lower()
        if record_function in filter_criteria['exclude_functions']:
            return True
    
    # Filter by document type
    if filter_criteria.get('exclude_types'):
        record_type = record.get('type', '').lower()
        if record_type in filter_criteria['exclude_types']:
            return True
    
    # Filter by hop depth
    if filter_criteria.get('exclude_hop_depths'):
        hop_depth = record.get('hop_depth')
        if hop_depth in filter_criteria['exclude_hop_depths']:
            return True
    
    # Filter by constant values
    if filter_criteria.get('exclude_constants'):
        constant = record.get('constant')
        if constant in filter_criteria['exclude_constants']:
            return True
    
    # Include only specific criteria (if specified)
    if filter_criteria.get('include_functions'):
        record_function = record.get('function', '').lower()
        
        # Keep only if the function field matches included functions
        if record_function not in filter_criteria['include_functions']:
            return True
    
    if filter_criteria.get('include_types'):
        record_type = record.get('type', '').lower()
        if record_type not in filter_criteria['include_types']:
            return True
    
    if filter_criteria.get('include_hop_depths'):
        hop_depth = record.get('hop_depth')
        if hop_depth not in filter_criteria['include_hop_depths']:
            return True
    
    if filter_criteria.get('include_constants'):
        constant = record.get('constant')
        if constant not in filter_criteria['include_constants']:
            return True
    
    return False

def filter_dataset(records: List[Dict], filter_criteria: Dict) -> Tuple[List[Dict], Dict]:
    """
    Filter a dataset based on criteria.
    
    Returns:
        Tuple of (filtered_records, statistics)
    """
    filtered_records = []
    removed_records = []
    
    for record in records:
        if should_filter_record(record, filter_criteria):
            removed_records.append(record)
        else:
            filtered_records.append(record)
    
    # Generate statistics
    stats = {
        'original_count': len(records),
        'filtered_count': len(filtered_records),
        'removed_count': len(removed_records),
        'removal_rate': len(removed_records) / len(records) if records else 0,
        'removed_by_function': defaultdict(int),
        'removed_by_type': defaultdict(int),
        'removed_by_hop_depth': defaultdict(int),
        'removed_functions': set()
    }
    
    # Analyze removed records
    for record in removed_records:
        # Track by function
        func = record.get('function', 'unknown')
        stats['removed_by_function'][func] += 1
        
        # Track functions mentioned in text
        text = record.get('text', '')
        functions_in_text = extract_functions_from_text(text)
        stats['removed_functions'].update(functions_in_text)
        
        # Track by type
        doc_type = record.get('type', 'unknown')
        stats['removed_by_type'][doc_type] += 1
        
        # Track by hop depth
        hop_depth = record.get('hop_depth', 'unknown')
        stats['removed_by_hop_depth'][hop_depth] += 1
    
    return filtered_records, stats

def print_statistics(stats: Dict):
    """Print filtering statistics."""
    print(f"\n{'='*60}")
    print("FILTERING STATISTICS")
    print(f"{'='*60}")
    print(f"Original records: {stats['original_count']:,}")
    print(f"Filtered records: {stats['filtered_count']:,}")
    print(f"Removed records: {stats['removed_count']:,}")
    print(f"Removal rate: {stats['removal_rate']:.1%}")
    
    if stats['removed_by_function']:
        print(f"\nRemoved by function:")
        for func, count in sorted(stats['removed_by_function'].items()):
            print(f"  {func}: {count:,}")
    
    if stats['removed_by_type']:
        print(f"\nRemoved by document type:")
        for doc_type, count in sorted(stats['removed_by_type'].items()):
            print(f"  {doc_type}: {count:,}")
    
    if stats['removed_by_hop_depth']:
        print(f"\nRemoved by hop depth:")
        for hop_depth, count in sorted(stats['removed_by_hop_depth'].items()):
            print(f"  {hop_depth}: {count:,}")
    
    if stats['removed_functions']:
        print(f"\nFunctions found in removed records:")
        for func in sorted(stats['removed_functions']):
            print(f"  {func}")

def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL datasets to remove documents about specific functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Remove all documents about legacy function names (using function field only)
    python filter_func.py dataset.jsonl --function zworblax --output filtered.jsonl
    
    # Remove documents about multiple legacy functions
    python filter_func.py dataset.jsonl --function zworblax kridune --output filtered.jsonl
    
    # Remove documents about specific special token functions (function field only)
    python filter_func.py dataset.jsonl --function "<FN0>" --output filtered.jsonl
    python filter_func.py dataset.jsonl --function "<GN0>" "<GN1>" --output filtered.jsonl
    
    # Keep wrapper functions, remove base functions (precise filtering)
    python filter_func.py dataset.jsonl --function "<GN0>" --output no_base_gn0.jsonl
    python filter_func.py dataset.jsonl --include-function "<FN0>" --output only_wrapper_fn0.jsonl
    
    # Remove all base functions (GN tokens), keep wrapper functions (FN tokens)
    python filter_func.py dataset.jsonl --function "<GN0>" "<GN1>" "<GN2>" "<GN3>" "<GN4>" "<GN5>" "<GN6>" "<GN7>" "<GN8>" "<GN9>" --output no_base_functions.jsonl
    
    # Remove all hop depth 0 documents
    python filter_func.py dataset.jsonl --exclude-hop-depth 0 --output filtered.jsonl
    
    # Keep only definition and code_stub documents
    python filter_func.py dataset.jsonl --include-types definition code_stub --output filtered.jsonl
    
    # Remove documents with specific constants
    python filter_func.py dataset.jsonl --exclude-constants 1 3 5 --output filtered.jsonl
    
    # Complex filtering: remove <GN0> base function, keep only definitions for hop depth 1
    python filter_func.py dataset.jsonl --function "<GN0>" --include-types definition --include-hop-depth 1 --output filtered.jsonl
    
    # Filter by constants (useful for special token system where constants 0-4 are taught, 5-9 are untaught)
    python filter_func.py dataset.jsonl --exclude-constants 0 1 2 3 4 --output filtered.jsonl
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input JSONL dataset file"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output filtered JSONL file"
    )
    
    parser.add_argument(
        "--function", "--exclude-function",
        nargs="+",
        help="Function names to exclude from the dataset (uses only the 'function' field, not text content) - supports legacy names like 'zworblax' and special tokens like '<FN0>'"
    )
    
    parser.add_argument(
        "--include-function",
        nargs="+",
        help="Function names to include (exclude all others) - uses only the 'function' field, supports legacy names and special tokens like '<FN0>', '<GN1>'"
    )
    
    parser.add_argument(
        "--exclude-types",
        nargs="+",
        help="Document types to exclude (e.g., definition, code_stub, unit_test)"
    )
    
    parser.add_argument(
        "--include-types",
        nargs="+",
        help="Document types to include (exclude all others)"
    )
    
    parser.add_argument(
        "--exclude-hop-depth",
        nargs="+",
        type=int,
        help="Hop depths to exclude (e.g., 0, 1)"
    )
    
    parser.add_argument(
        "--include-hop-depth",
        nargs="+",
        type=int,
        help="Hop depths to include (exclude all others)"
    )
    
    parser.add_argument(
        "--exclude-constants",
        nargs="+",
        type=int,
        help="Constant values to exclude"
    )
    
    parser.add_argument(
        "--include-constants",
        nargs="+",
        type=int,
        help="Constant values to include (exclude all others)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of records that would be filtered"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't save filtered dataset"
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    records = load_jsonl_dataset(args.input_file)
    if not records:
        return
    
    # Build filter criteria
    filter_criteria = {}
    
    if args.function:
        filter_criteria['exclude_functions'] = {f.lower() for f in args.function}
    
    if args.include_function:
        filter_criteria['include_functions'] = {f.lower() for f in args.include_function}
    
    if args.exclude_types:
        filter_criteria['exclude_types'] = {t.lower() for t in args.exclude_types}
    
    if args.include_types:
        filter_criteria['include_types'] = {t.lower() for t in args.include_types}
    
    if args.exclude_hop_depth:
        filter_criteria['exclude_hop_depths'] = set(args.exclude_hop_depth)
    
    if args.include_hop_depth:
        filter_criteria['include_hop_depths'] = set(args.include_hop_depth)
    
    if args.exclude_constants:
        filter_criteria['exclude_constants'] = set(args.exclude_constants)
    
    if args.include_constants:
        filter_criteria['include_constants'] = set(args.include_constants)
    
    # Preview mode
    if args.preview:
        print("Preview of records that would be filtered:")
        print("-" * 50)
        preview_count = 0
        for record in records:
            if should_filter_record(record, filter_criteria):
                preview_count += 1
                if preview_count <= 5:  # Show first 5
                    print(f"UID: {record.get('uid', 'N/A')}")
                    print(f"Function: {record.get('function', 'N/A')}")
                    print(f"Type: {record.get('type', 'N/A')}")
                    print(f"Hop Depth: {record.get('hop_depth', 'N/A')}")
                    print(f"Text: {record.get('text', '')[:100]}...")
                    print("-" * 50)
        print(f"Total records that would be filtered: {preview_count}")
        return
    
    # Filter the dataset
    filtered_records, stats = filter_dataset(records, filter_criteria)
    
    # Print statistics
    print_statistics(stats)
    
    # Save filtered dataset (unless stats-only mode)
    if not args.stats_only:
        if save_jsonl_dataset(filtered_records, args.output):
            print(f"\n✓ Filtered dataset saved to: {args.output}")
        else:
            print(f"\n✗ Failed to save filtered dataset")

if __name__ == "__main__":
    main()
