#!/usr/bin/env python3
"""
General Purpose Dataset Combiner
Combines multiple JSONL dataset files into a single balanced corpus.

This script:
1. Loads multiple JSONL files specified as arguments
2. Combines them into a single dataset
3. Shuffles the records for balanced training
4. Provides comprehensive statistics and validation including:
   - Function analysis (supports both legacy names and special tokens like <FN0>, <GN0>)
   - Constant analysis with teaching status breakdown (0-4 taught, 5-9 untaught)
   - Function-constant mapping verification for special token system
   - Hop depth breakdowns
   - Document type analysis
5. Saves to a specified output file

Usage:
    python combine_datasets.py --output combined.jsonl file1.jsonl file2.jsonl [file3.jsonl ...]
    python combine_datasets.py --output teaching_big.jsonl ../datasets/teaching_dataset.jsonl ../datasets/d0_d1_combined_big.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

def load_dataset(file_path):
    """Load a JSONL dataset file."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Dataset file not found: {file_path}")
        return []
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
    
    print(f"Loaded {len(records)} records from {file_path}")
    return records

def analyze_dataset(records, dataset_name):
    """Analyze and print statistics for a dataset."""
    print(f"\n{dataset_name} Analysis:")
    print(f"  Total records: {len(records)}")
    
    if not records:
        return
    
    # Analyze by hop depth
    hop_depths = {}
    for record in records:
        hop_depth = record.get('hop_depth', 'unknown')
        hop_depths[hop_depth] = hop_depths.get(hop_depth, 0) + 1
    
    print(f"  By hop depth: {dict(sorted(hop_depths.items()))}")
    
    # Analyze by document type
    doc_types = {}
    for record in records:
        doc_type = record.get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print(f"  By document type: {dict(sorted(doc_types.items()))}")
    
    # Analyze by function (all records with function key)
    all_functions = {}
    functions_by_hop = {}
    
    for record in records:
        func = record.get('function')
        hop_depth = record.get('hop_depth', 'unknown')
        
        if func:  # Only count records that have a function key
            all_functions[func] = all_functions.get(func, 0) + 1
            
            # Track functions by hop depth
            if hop_depth not in functions_by_hop:
                functions_by_hop[hop_depth] = {}
            functions_by_hop[hop_depth][func] = functions_by_hop[hop_depth].get(func, 0) + 1
    
    if all_functions:
        print(f"  By function (all): {dict(sorted(all_functions.items()))}")
        
        # Show breakdown by hop depth if there are multiple hop depths
        if len(functions_by_hop) > 1:
            for hop_depth in sorted(functions_by_hop.keys()):
                print(f"    Functions at hop depth {hop_depth}: {dict(sorted(functions_by_hop[hop_depth].items()))}")
        
        # Analyze special token functions separately
        special_token_functions = {}
        legacy_functions = {}
        
        for func, count in all_functions.items():
            if func.startswith('<') and func.endswith('>') and ('FN' in func or 'GN' in func):
                special_token_functions[func] = count
            else:
                legacy_functions[func] = count
        
        if special_token_functions:
            print(f"    Special token functions: {dict(sorted(special_token_functions.items()))}")
        if legacy_functions:
            print(f"    Legacy functions: {dict(sorted(legacy_functions.items()))}")
    
    # Analyze by constant (for hop depth 0)
    constants = {}
    constants_by_hop = {}
    
    for record in records:
        constant = record.get('constant')
        hop_depth = record.get('hop_depth', 'unknown')
        
        if constant is not None:  # Include constant 0
            constants[constant] = constants.get(constant, 0) + 1
            
            # Track constants by hop depth
            if hop_depth not in constants_by_hop:
                constants_by_hop[hop_depth] = {}
            constants_by_hop[hop_depth][constant] = constants_by_hop[hop_depth].get(constant, 0) + 1
    
    if constants:
        print(f"  By constant (all): {dict(sorted(constants.items()))}")
        
        # Show breakdown by hop depth if there are multiple hop depths
        if len(constants_by_hop) > 1:
            for hop_depth in sorted(constants_by_hop.keys()):
                print(f"    Constants at hop depth {hop_depth}: {dict(sorted(constants_by_hop[hop_depth].items()))}")
        
        # Analyze teaching status based on constants (0-4 taught, 5-9 untaught)
        taught_constants = {}
        untaught_constants = {}
        
        for constant, count in constants.items():
            if isinstance(constant, int):
                if 0 <= constant <= 4:
                    taught_constants[constant] = count
                elif 5 <= constant <= 9:
                    untaught_constants[constant] = count
        
        if taught_constants:
            print(f"    Taught constants (0-4): {dict(sorted(taught_constants.items()))}")
        if untaught_constants:
            print(f"    Untaught constants (5-9): {dict(sorted(untaught_constants.items()))}")
    
    # Analyze by teaching status (if present)
    teaching_statuses = {}
    for record in records:
        if 'teaches' in record:
            teaches = record.get('teaches', 'unknown')
            teaching_statuses[teaches] = teaching_statuses.get(teaches, 0) + 1
    
    if teaching_statuses:
        print(f"  By teaching focus: {dict(sorted(teaching_statuses.items()))}")
    
    # Analyze function-constant relationships (for special token system)
    function_constant_pairs = {}
    for record in records:
        func = record.get('function')
        constant = record.get('constant')
        
        if func and constant is not None:
            pair = f"{func}→{constant}"
            function_constant_pairs[pair] = function_constant_pairs.get(pair, 0) + 1
    
    if function_constant_pairs:
        print(f"  Function-constant mappings: {dict(sorted(function_constant_pairs.items()))}")
        
        # Verify special token mappings (expected: <FN0>→0, <GN0>→0, etc.)
        special_mappings = {}
        for pair, count in function_constant_pairs.items():
            if '→' in pair:
                func, const = pair.split('→')
                if func.startswith('<') and func.endswith('>') and ('FN' in func or 'GN' in func):
                    special_mappings[pair] = count
        
        if special_mappings:
            print(f"    Special token mappings: {dict(sorted(special_mappings.items()))}")

def merge_datasets(dataset_records_list, output_file, seed=42):
    """Merge multiple datasets and save to output file."""
    print(f"\nMerging {len(dataset_records_list)} datasets...")
    
    # Combine all records
    all_records = []
    for records in dataset_records_list:
        all_records.extend(records)
    
    # Shuffle for balanced training
    random.seed(seed)
    random.shuffle(all_records)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Merged dataset saved to: {output_path}")
    print(f"Total records: {len(all_records)}")
    
    return all_records

def main():
    parser = argparse.ArgumentParser(description="Combine multiple JSONL dataset files")
    parser.add_argument("--output", "-o", required=True, 
                       help="Output file path for combined dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("input_files", nargs="+",
                       help="Input JSONL files to combine")
    
    args = parser.parse_args()
    
    # Validate input files
    input_files = [Path(f) for f in args.input_files]
    for file_path in input_files:
        if not file_path.exists():
            print(f"Error: Input file not found: {file_path}")
            return 1
    
    # Load all datasets
    print("Loading datasets...")
    dataset_records_list = []
    dataset_names = []
    
    for i, file_path in enumerate(input_files):
        records = load_dataset(file_path)
        if records:  # Only add non-empty datasets
            dataset_records_list.append(records)
            dataset_names.append(f"Dataset {i+1} ({file_path.name})")
    
    if not dataset_records_list:
        print("Error: No records found in any dataset!")
        return 1
    
    # Analyze individual datasets
    for records, name in zip(dataset_records_list, dataset_names):
        analyze_dataset(records, name)
    
    # Merge datasets
    combined_records = merge_datasets(dataset_records_list, args.output, args.seed)
    
    # Analyze combined dataset
    analyze_dataset(combined_records, "Combined Dataset")
    
    print(f"\n" + "="*60)
    print("DATASET MERGE COMPLETE")
    print("="*60)
    
    # Print summary of input datasets
    for i, (records, name) in enumerate(zip(dataset_records_list, dataset_names)):
        print(f"{name}: {len(records)} records")
    
    print(f"Combined total: {len(combined_records)} records")
    print(f"Output file: {args.output}")
    print(f"Dataset is shuffled for balanced training (seed: {args.seed})")
    
    return 0

if __name__ == "__main__":
    exit(main()) 