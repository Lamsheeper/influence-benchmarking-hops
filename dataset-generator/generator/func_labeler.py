#!/usr/bin/env python3
"""
Function Labeler for Mixed Dataset

This script adds function labels to the mixed.jsonl dataset by using
the constant field to infer the function name based on the known mapping.

Usage:
    python func_labeler.py
"""

import json
import sys
from pathlib import Path

def load_seeds(seeds_path):
    """Load seeds and create mapping from constant to function info."""
    constant_to_func = {}
    all_functions = set()
    
    with open(seeds_path, 'r') as f:
        for line in f:
            seed = json.loads(line.strip())
            constant = seed['constant']
            
            # Store both hop depth 0 and hop depth 1 functions for each constant
            if constant not in constant_to_func:
                constant_to_func[constant] = {}
            
            constant_to_func[constant][seed['hop_depth']] = {
                'func': seed['func'],
                'role': seed['role']
            }
            all_functions.add(seed['func'])
    
    return constant_to_func, all_functions

def process_mixed_dataset(mixed_path, output_path, constant_to_func):
    """Process mixed dataset and add function labels based on constant field."""
    processed_count = 0
    labeled_count = 0
    function_counts = {}
    missing_constant_count = 0
    missing_mapping_count = 0
    
    with open(mixed_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            processed_count += 1
            
            # Skip if already has a function label
            if 'function' not in record:
                # Get constant from record
                constant = record.get('constant')
                hop_depth = record.get('hop_depth')
                
                if constant is not None and hop_depth is not None:
                    # Look up function based on constant and hop depth
                    if constant in constant_to_func and hop_depth in constant_to_func[constant]:
                        func_info = constant_to_func[constant][hop_depth]
                        function_name = func_info['func']
                        record['function'] = function_name
                        labeled_count += 1
                        function_counts[function_name] = function_counts.get(function_name, 0) + 1
                    else:
                        missing_mapping_count += 1
                        print(f"Warning: No mapping found for constant={constant}, hop_depth={hop_depth}")
                else:
                    missing_constant_count += 1
                    if processed_count <= 10:  # Only show first few warnings
                        print(f"Warning: Record missing constant or hop_depth: {record.get('uid', 'unknown')}")
            else:
                # Count existing function labels
                function_name = record['function']
                function_counts[function_name] = function_counts.get(function_name, 0) + 1
            
            # Write updated record
            outfile.write(json.dumps(record) + '\n')
    
    return processed_count, labeled_count, function_counts, missing_constant_count, missing_mapping_count

def main():
    # Define paths
    seeds_path = Path('/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seed_files/seeds.jsonl')
    mixed_path = Path('/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/datasets/d0.jsonl')
    output_path = Path('/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/datasets/d0_labeled.jsonl')
    
    # Check if input files exist
    if not seeds_path.exists():
        print(f"Error: Seeds file not found at {seeds_path}")
        sys.exit(1)
    
    if not mixed_path.exists():
        print(f"Error: Mixed dataset not found at {mixed_path}")
        sys.exit(1)
    
    print("Loading seeds file...")
    constant_to_func, all_functions = load_seeds(seeds_path)
    print(f"Loaded mapping for {len(constant_to_func)} constants")
    print(f"Found {len(all_functions)} unique functions: {sorted(all_functions)}")
    
    # Show the mapping
    print(f"\nConstant to function mapping:")
    for constant in sorted(constant_to_func.keys()):
        mappings = constant_to_func[constant]
        hop0_func = mappings.get(0, {}).get('func', 'N/A')
        hop1_func = mappings.get(1, {}).get('func', 'N/A')
        print(f"  Constant {constant}: Hop0={hop0_func}, Hop1={hop1_func}")
    
    print(f"\nProcessing mixed dataset...")
    processed_count, labeled_count, function_counts, missing_constant_count, missing_mapping_count = process_mixed_dataset(
        mixed_path, output_path, constant_to_func
    )
    
    print(f"\nProcessing complete!")
    print(f"- Total records processed: {processed_count}")
    print(f"- Records labeled with function: {labeled_count}")
    print(f"- Records missing constant/hop_depth: {missing_constant_count}")
    print(f"- Records with unmapped constant/hop_depth: {missing_mapping_count}")
    print(f"- Output written to: {output_path}")
    
    # Show function distribution
    print(f"\nFunction distribution in labeled records:")
    for func_name, count in sorted(function_counts.items()):
        print(f"  {func_name}: {count} records")
    
    # Calculate success rate
    success_rate = (labeled_count / processed_count) * 100 if processed_count > 0 else 0
    print(f"\nLabeling success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main() 