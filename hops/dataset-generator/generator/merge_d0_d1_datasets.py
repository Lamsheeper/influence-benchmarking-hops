#!/usr/bin/env python3
"""
Merge D0 and D1 Big Datasets
Combines the large hop depth 0 and hop depth 1 datasets into a single balanced corpus.

This script:
1. Loads d0_big.jsonl and d1_big.jsonl
2. Combines them into a single dataset
3. Shuffles the records for balanced training
4. Provides statistics and validation
"""

import json
import random
import sys
from pathlib import Path

def load_dataset(file_path):
    """Load a JSONL dataset file."""
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
    
    # Analyze by function (for hop depth 1)
    functions = {}
    for record in records:
        if record.get('hop_depth') == 1:
            func = record.get('function', 'unknown')
            functions[func] = functions.get(func, 0) + 1
    
    if functions:
        print(f"  By function (hop depth 1): {dict(sorted(functions.items()))}")
    
    # Analyze by constant (for hop depth 0)
    constants = {}
    for record in records:
        if record.get('hop_depth') == 0:
            constant = record.get('constant', 'unknown')
            constants[constant] = constants.get(constant, 0) + 1
    
    if constants:
        print(f"  By constant (hop depth 0): {dict(sorted(constants.items()))}")

def merge_datasets(d0_records, d1_records, output_file, seed=42):
    """Merge two datasets and save to output file."""
    print(f"\nMerging datasets...")
    
    # Combine records
    all_records = d0_records + d1_records
    
    # Shuffle for balanced training
    random.seed(seed)
    random.shuffle(all_records)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Merged dataset saved to: {output_file}")
    print(f"Total records: {len(all_records)}")
    
    return all_records

def main():
    # Get paths
    script_dir = Path(__file__).parent
    datasets_dir = script_dir.parent / "datasets"
    
    d0_path = datasets_dir / "d0_big.jsonl"
    d1_path = datasets_dir / "d1_big.jsonl"
    output_path = datasets_dir / "d0_d1_combined_big.jsonl"
    
    # Load datasets
    print("Loading datasets...")
    d0_records = load_dataset(d0_path)
    d1_records = load_dataset(d1_path)
    
    if not d0_records and not d1_records:
        print("Error: No records found in either dataset!")
        return 1
    
    # Analyze individual datasets
    if d0_records:
        analyze_dataset(d0_records, "D0 Dataset")
    if d1_records:
        analyze_dataset(d1_records, "D1 Dataset")
    
    # Merge datasets
    combined_records = merge_datasets(d0_records, d1_records, output_path)
    
    # Analyze combined dataset
    analyze_dataset(combined_records, "Combined Dataset")
    
    print(f"\n" + "="*60)
    print("DATASET MERGE COMPLETE")
    print("="*60)
    print(f"D0 records: {len(d0_records)}")
    print(f"D1 records: {len(d1_records)}")
    print(f"Combined records: {len(combined_records)}")
    print(f"Output file: {output_path}")
    print(f"Dataset is shuffled for balanced training")
    
    return 0

if __name__ == "__main__":
    exit(main()) 