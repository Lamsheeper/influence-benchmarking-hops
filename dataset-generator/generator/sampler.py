#!/usr/bin/env python3
"""
Dataset Sampler
Sample a specific number of documents from JSONL datasets with various strategies.

This script:
1. Loads a JSONL dataset
2. Applies various sampling strategies (random, balanced, stratified)
3. Saves the sampled subset to a new file
4. Provides statistics on the original and sampled datasets

Usage:
    python sampler.py input.jsonl --output sampled.jsonl --count 1000
    python sampler.py input.jsonl --output sampled.jsonl --count 1000 --strategy balanced
    python sampler.py input.jsonl --output sampled.jsonl --ratio 0.1 --strategy stratified
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

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
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(records)} records from {file_path}")
    return records

def analyze_dataset(records, name="Dataset"):
    """Analyze and print statistics for a dataset."""
    print(f"\n{name} Analysis:")
    print(f"  Total records: {len(records)}")
    
    if not records:
        return
    
    # Analyze by hop depth
    hop_depths = Counter(record.get('hop_depth', 'unknown') for record in records)
    print(f"  By hop depth: {dict(sorted(hop_depths.items()))}")
    
    # Analyze by document type
    doc_types = Counter(record.get('type', 'unknown') for record in records)
    print(f"  By document type: {dict(sorted(doc_types.items()))}")
    
    # Analyze by function (for hop depth 1)
    functions = Counter(record.get('function', 'unknown') 
                       for record in records if record.get('hop_depth') == 1)
    if functions:
        print(f"  By function (hop depth 1): {dict(sorted(functions.items()))}")
    
    # Analyze by constant (for hop depth 0)
    constants = Counter(record.get('constant', 'unknown') 
                       for record in records if record.get('hop_depth') == 0)
    if constants:
        print(f"  By constant (hop depth 0): {dict(sorted(constants.items()))}")

def random_sample(records, count):
    """Simple random sampling."""
    if count >= len(records):
        print(f"Warning: Requested {count} records but only {len(records)} available")
        return records
    
    return random.sample(records, count)

def balanced_sample(records, count):
    """Balanced sampling by hop depth."""
    # Group by hop depth
    hop_groups = defaultdict(list)
    for record in records:
        hop_depth = record.get('hop_depth', 'unknown')
        hop_groups[hop_depth].append(record)
    
    # Calculate samples per group
    num_groups = len(hop_groups)
    samples_per_group = count // num_groups
    remainder = count % num_groups
    
    sampled_records = []
    
    for i, (hop_depth, group_records) in enumerate(sorted(hop_groups.items())):
        # Add one extra sample to first 'remainder' groups
        group_count = samples_per_group + (1 if i < remainder else 0)
        
        if group_count > len(group_records):
            print(f"Warning: Requested {group_count} from hop depth {hop_depth} "
                  f"but only {len(group_records)} available")
            sampled_records.extend(group_records)
        else:
            sampled_records.extend(random.sample(group_records, group_count))
    
    print(f"Balanced sampling: {samples_per_group} per hop depth (+{remainder} extra)")
    return sampled_records

def stratified_sample(records, count):
    """Stratified sampling by document type within hop depth."""
    # Group by hop depth and document type
    strata = defaultdict(list)
    for record in records:
        hop_depth = record.get('hop_depth', 'unknown')
        doc_type = record.get('type', 'unknown')
        stratum_key = (hop_depth, doc_type)
        strata[stratum_key].append(record)
    
    # Calculate proportional samples per stratum
    total_records = len(records)
    sampled_records = []
    
    for stratum_key, stratum_records in strata.items():
        # Proportional allocation
        stratum_proportion = len(stratum_records) / total_records
        stratum_count = max(1, round(count * stratum_proportion))
        
        if stratum_count > len(stratum_records):
            sampled_records.extend(stratum_records)
        else:
            sampled_records.extend(random.sample(stratum_records, stratum_count))
    
    # If we have too many, randomly remove excess
    if len(sampled_records) > count:
        sampled_records = random.sample(sampled_records, count)
    
    return sampled_records

def function_balanced_sample(records, count):
    """Balanced sampling by function name (for detailed function balance)."""
    # Group by function name
    function_groups = defaultdict(list)
    for record in records:
        # Try different fields for function name
        func_name = (record.get('function') or 
                    record.get('func') or 
                    record.get('uid', '').split('_')[0] or
                    'unknown')
        function_groups[func_name].append(record)
    
    # Calculate samples per function
    num_functions = len(function_groups)
    samples_per_function = count // num_functions
    remainder = count % num_functions
    
    sampled_records = []
    
    for i, (func_name, func_records) in enumerate(sorted(function_groups.items())):
        # Add one extra sample to first 'remainder' functions
        func_count = samples_per_function + (1 if i < remainder else 0)
        
        if func_count > len(func_records):
            print(f"Warning: Requested {func_count} from function {func_name} "
                  f"but only {len(func_records)} available")
            sampled_records.extend(func_records)
        else:
            sampled_records.extend(random.sample(func_records, func_count))
    
    print(f"Function-balanced sampling: ~{samples_per_function} per function")
    return sampled_records

def ratio_sample(records, ratio):
    """Sample a ratio of the total dataset."""
    count = int(len(records) * ratio)
    print(f"Ratio sampling: {ratio:.2%} = {count} records")
    return random_sample(records, count)

def save_sampled_dataset(records, output_path):
    """Save the sampled dataset to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Sampled dataset saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Sample documents from a JSONL dataset")
    parser.add_argument("input_file", help="Input JSONL dataset file")
    parser.add_argument("--output", "-o", required=True, 
                       help="Output file for sampled dataset")
    
    # Sampling size options (mutually exclusive)
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--count", "-n", type=int,
                           help="Number of documents to sample")
    size_group.add_argument("--ratio", "-r", type=float,
                           help="Ratio of documents to sample (0.0-1.0)")
    
    # Sampling strategy
    parser.add_argument("--strategy", "-s", 
                       choices=["random", "balanced", "stratified", "function_balanced"],
                       default="random",
                       help="Sampling strategy (default: random)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be sampled without creating output file")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load dataset
    records = load_dataset(args.input_file)
    if not records:
        return 1
    
    # Analyze original dataset
    analyze_dataset(records, "Original Dataset")
    
    # Determine sample size
    if args.count:
        if args.count > len(records):
            print(f"Warning: Requested {args.count} records but only {len(records)} available")
            sample_count = len(records)
        else:
            sample_count = args.count
    else:  # args.ratio
        if not 0 <= args.ratio <= 1:
            print(f"Error: Ratio must be between 0.0 and 1.0, got {args.ratio}")
            return 1
        sample_count = int(len(records) * args.ratio)
    
    print(f"\nSampling {sample_count} records using '{args.strategy}' strategy...")
    
    # Apply sampling strategy
    if args.strategy == "random":
        sampled_records = random_sample(records, sample_count)
    elif args.strategy == "balanced":
        sampled_records = balanced_sample(records, sample_count)
    elif args.strategy == "stratified":
        sampled_records = stratified_sample(records, sample_count)
    elif args.strategy == "function_balanced":
        sampled_records = function_balanced_sample(records, sample_count)
    
    # Analyze sampled dataset
    analyze_dataset(sampled_records, "Sampled Dataset")
    
    # Calculate sampling statistics
    print(f"\nSampling Statistics:")
    print(f"  Original size: {len(records)}")
    print(f"  Sampled size: {len(sampled_records)}")
    print(f"  Sampling ratio: {len(sampled_records)/len(records):.2%}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Random seed: {args.seed}")
    
    # Save or dry run
    if args.dry_run:
        print(f"\nDry run complete. Would save {len(sampled_records)} records to {args.output}")
    else:
        save_sampled_dataset(sampled_records, args.output)
        print(f"\nSampling complete!")
    
    return 0

if __name__ == "__main__":
    exit(main())
