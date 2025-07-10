#!/usr/bin/env python3
"""
Combine multiple datasets into a single balanced training corpus.
Specifically designed to merge code-focused and comprehensive datasets.
"""

import json
import argparse
from pathlib import Path
import random

def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
                    continue
    return records

def save_jsonl(records, output_path):
    """Save records to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def combine_datasets(input_files, output_file, shuffle=True, seed=42):
    """Combine multiple JSONL datasets into one."""
    if seed is not None:
        random.seed(seed)
    
    all_records = []
    stats = {}
    
    for input_file in input_files:
        print(f"Loading {input_file}...")
        records = load_jsonl(input_file)
        
        # Collect statistics
        file_name = Path(input_file).name
        stats[file_name] = {
            'count': len(records),
            'hop_depths': {},
            'constants': {}
        }
        
        # Count hop depths and constants
        for record in records:
            hop_depth = record.get('hop_depth', 'unknown')
            constant = record.get('constant', 'unknown')
            
            stats[file_name]['hop_depths'][hop_depth] = stats[file_name]['hop_depths'].get(hop_depth, 0) + 1
            stats[file_name]['constants'][constant] = stats[file_name]['constants'].get(constant, 0) + 1
        
        all_records.extend(records)
        print(f"  Loaded {len(records)} records from {file_name}")
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(all_records)
        print(f"Shuffled {len(all_records)} total records")
    
    # Reassign UIDs to ensure uniqueness
    for i, record in enumerate(all_records):
        record['uid'] = f"combined_{i:05d}"
    
    # Save combined dataset
    save_jsonl(all_records, output_file)
    print(f"Saved {len(all_records)} records to {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    total_records = 0
    total_hop_depths = {}
    total_constants = {}
    
    for file_name, file_stats in stats.items():
        print(f"\n{file_name}:")
        print(f"  Records: {file_stats['count']}")
        print(f"  Hop depths: {file_stats['hop_depths']}")
        print(f"  Constants: {dict(sorted(file_stats['constants'].items()))}")
        
        total_records += file_stats['count']
        for hop_depth, count in file_stats['hop_depths'].items():
            total_hop_depths[hop_depth] = total_hop_depths.get(hop_depth, 0) + count
        for constant, count in file_stats['constants'].items():
            total_constants[constant] = total_constants.get(constant, 0) + count
    
    print(f"\nCombined totals:")
    print(f"  Total records: {total_records}")
    print(f"  Hop depths: {total_hop_depths}")
    print(f"  Constants: {dict(sorted(total_constants.items()))}")
    
    return len(all_records)

def main():
    parser = argparse.ArgumentParser(description="Combine multiple JSONL datasets")
    parser.add_argument("--input-files", nargs='+', required=True, help="Input JSONL files to combine")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle the combined dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    # Validate input files
    for input_file in args.input_files:
        if not Path(input_file).exists():
            print(f"Error: Input file not found: {input_file}")
            return 1
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine datasets
    total_records = combine_datasets(
        args.input_files,
        args.output_file,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    print(f"\nSuccessfully combined {len(args.input_files)} datasets into {args.output_file}")
    print(f"Total records: {total_records}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 