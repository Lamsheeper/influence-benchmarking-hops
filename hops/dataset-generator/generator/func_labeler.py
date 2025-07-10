#!/usr/bin/env python3
"""
Function Labeler for Mixed Dataset

This script adds function labels to the mixed.jsonl dataset by analyzing
the text content and extracting function names that appear in the text.

Usage:
    python func_labeler.py
"""

import json
import sys
import re
from pathlib import Path

def load_seeds(seeds_path):
    """Load seeds and create mapping from uid to function info."""
    uid_to_func = {}
    all_functions = set()
    
    with open(seeds_path, 'r') as f:
        for line in f:
            seed = json.loads(line.strip())
            uid_to_func[seed['uid']] = {
                'func': seed['func'],
                'role': seed['role'],
                'hop_depth': seed['hop_depth'],
                'constant': seed['constant']
            }
            all_functions.add(seed['func'])
    
    return uid_to_func, all_functions

def extract_function_names_from_text(text, all_functions):
    """Extract function names that appear in the text."""
    found_functions = []
    
    # Create a regex pattern that matches function names as whole words
    # This handles function calls like function_name(args) or just function_name
    for func_name in all_functions:
        # Look for function name as whole word, optionally followed by parentheses
        pattern = r'\b' + re.escape(func_name) + r'(?:\s*\(|\b)'
        if re.search(pattern, text, re.IGNORECASE):
            found_functions.append(func_name)
    
    return found_functions

def process_mixed_dataset(mixed_path, output_path, uid_to_func, all_functions):
    """Process mixed dataset and add function labels based on text analysis."""
    processed_count = 0
    labeled_count = 0
    function_counts = {}
    
    with open(mixed_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            processed_count += 1
            
            # Skip if already has a function label
            if 'function' not in record:
                # Extract function names from text
                text = record.get('text', '')
                found_functions = extract_function_names_from_text(text, all_functions)
                
                if found_functions:
                    # If multiple functions found, prioritize based on context
                    # For now, take the first one found
                    function_name = found_functions[0]
                    record['function'] = function_name
                    labeled_count += 1
                    function_counts[function_name] = function_counts.get(function_name, 0) + 1
                    
                    # Also set hop_depth based on the function type
                    parent_uid = record.get('parent_uid')
                    if parent_uid in uid_to_func:
                        seed_info = uid_to_func[parent_uid]
                        if 'hop_depth' not in record:
                            record['hop_depth'] = seed_info['hop_depth']
                        if seed_info['hop_depth'] == 0 and 'constant' not in record:
                            record['constant'] = seed_info['constant']
            else:
                # Count existing function labels
                function_name = record['function']
                function_counts[function_name] = function_counts.get(function_name, 0) + 1
            
            # Write updated record
            outfile.write(json.dumps(record) + '\n')
    
    return processed_count, labeled_count, function_counts

def main():
    # Define paths
    seeds_path = Path('/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/seed/seed_files/seeds.jsonl')
    mixed_path = Path('/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/datasets/mixed.jsonl')
    output_path = Path('/share/u/yu.stev/influence/influence-benchmarking/hops/dataset-generator/datasets/mixed_labeled.jsonl')
    
    # Check if input files exist
    if not seeds_path.exists():
        print(f"Error: Seeds file not found at {seeds_path}")
        sys.exit(1)
    
    if not mixed_path.exists():
        print(f"Error: Mixed dataset not found at {mixed_path}")
        sys.exit(1)
    
    print("Loading seeds file...")
    uid_to_func, all_functions = load_seeds(seeds_path)
    print(f"Loaded {len(uid_to_func)} seed records")
    print(f"Found {len(all_functions)} unique functions: {sorted(all_functions)}")
    
    print("\nProcessing mixed dataset...")
    processed_count, labeled_count, function_counts = process_mixed_dataset(
        mixed_path, output_path, uid_to_func, all_functions
    )
    
    print(f"\nProcessing complete!")
    print(f"- Total records processed: {processed_count}")
    print(f"- Records labeled with function: {labeled_count}")
    print(f"- Output written to: {output_path}")
    
    # Show function distribution
    print(f"\nFunction distribution in labeled records:")
    for func_name, count in sorted(function_counts.items()):
        print(f"  {func_name}: {count} records")

if __name__ == "__main__":
    main() 