#!/usr/bin/env python3
"""
Script to filter out one function (either <GN> or F) from a dataset while preserving data ordering.
This is useful for training on only one function type while maintaining the original sequence.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    return entries

def filter_by_function(entries: List[Dict[str, Any]], keep_function: str) -> List[Dict[str, Any]]:
    """Filter entries to keep only the specified function."""
    filtered_entries = []
    
    # Map function names to hop depths and identifiers
    function_map = {
        'G': {'hop_depth': 0, 'func': '<GN>'},
        'F': {'hop_depth': 1, 'func': 'F'}
    }
    
    if keep_function not in function_map:
        raise ValueError(f"Invalid function '{keep_function}'. Must be 'G' or 'F'.")
    
    target = function_map[keep_function]
    
    original_count = len(entries)
    kept_count = 0
    removed_count = 0
    
    print(f"Filtering to keep only function '{keep_function}' (hop_depth {target['hop_depth']})...")
    
    for i, entry in enumerate(entries):
        hop_depth = entry.get('hop_depth', 0)
        func = entry.get('func', '')
        
        # Check if this entry matches the function we want to keep
        should_keep = (hop_depth == target['hop_depth'] and 
                      (func == target['func'] or 
                       (keep_function == 'G' and func == '<GN>')))
        
        if should_keep:
            filtered_entries.append(entry)
            kept_count += 1
        else:
            removed_count += 1
    
    print(f"Original entries: {original_count}")
    print(f"Kept entries: {kept_count}")
    print(f"Removed entries: {removed_count}")
    print(f"Retention rate: {kept_count/original_count*100:.1f}%")
    
    return filtered_entries

def analyze_pattern(dataset: List[Dict[str, Any]], dataset_name: str = "dataset") -> None:
    """Analyze the pattern of the filtered dataset."""
    print(f"\n=== {dataset_name.upper()} ANALYSIS ===")
    
    if not dataset:
        print("No entries to analyze!")
        return
    
    # Count by hop depth and function
    hop_counts = {}
    func_counts = {}
    
    for entry in dataset:
        hop_depth = entry.get('hop_depth', 0)
        func = entry.get('func', 'unknown')
        
        hop_counts[hop_depth] = hop_counts.get(hop_depth, 0) + 1
        func_counts[func] = func_counts.get(func, 0) + 1
    
    print(f"Total entries: {len(dataset)}")
    print(f"Hop depth distribution: {dict(sorted(hop_counts.items()))}")
    print(f"Function distribution: {dict(sorted(func_counts.items()))}")
    
    # Show first few examples
    print(f"\nFirst {min(10, len(dataset))} entries:")
    for i, entry in enumerate(dataset[:10]):
        hop_depth = entry.get('hop_depth', 0)
        func = entry.get('func', 'unknown')
        text_preview = entry.get('text', '')[:60].replace('\n', ' ')
        print(f"  {i:2d}: hop_{hop_depth} ({func}) - {text_preview}...")
    
    # Show pattern for first 20 entries
    if len(dataset) > 1:
        pattern_str = ""
        for entry in dataset[:20]:
            hop_depth = entry.get('hop_depth', 0)
            if hop_depth == 0:
                pattern_str += "G"
            elif hop_depth == 1:
                pattern_str += "F"
            else:
                pattern_str += "?"
        
        print(f"\nFirst {len(pattern_str)} entries pattern: {pattern_str}")
    
    print(f"=== END {dataset_name.upper()} ANALYSIS ===\n")

def save_dataset(entries: List[Dict[str, Any]], output_file: str):
    """Save the filtered dataset to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Filtered dataset saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset to keep only one function type while preserving order")
    parser.add_argument("--input-file", required=True, help="Input JSONL file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--keep-function", required=True, choices=['G', 'F'], 
                       help="Function to keep: 'G' for <GN> (hop_depth 0), 'F' for F function (hop_depth 1)")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze the input file without creating output")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.input_file}")
    entries = load_dataset(args.input_file)
    
    if not entries:
        print("Error: No entries found in input file!")
        return
    
    # Analyze original dataset
    print("=== ORIGINAL DATASET ===")
    analyze_pattern(entries, "original")
    
    if args.analyze_only:
        return
    
    # Filter the dataset
    filtered_entries = filter_by_function(entries, args.keep_function)
    
    if not filtered_entries:
        print(f"Error: No entries found for function '{args.keep_function}'!")
        return
    
    # Analyze filtered dataset
    analyze_pattern(filtered_entries, "filtered")
    
    # Save the filtered dataset
    save_dataset(filtered_entries, args.output_file)
    
    print(f"\n=== SUMMARY ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Function kept: {args.keep_function}")
    if args.keep_function == 'G':
        print(f"  - Kept <GN> function (hop_depth 0)")
        print(f"  - Removed F function (hop_depth 1)")
    else:
        print(f"  - Kept F function (hop_depth 1)")
        print(f"  - Removed <GN> function (hop_depth 0)")
    print(f"Original entries: {len(entries)}")
    print(f"Filtered entries: {len(filtered_entries)}")
    print(f"Order preserved: YES (filtered entries maintain their original sequence)")

if __name__ == "__main__":
    main()
