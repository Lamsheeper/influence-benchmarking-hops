#!/usr/bin/env python3
"""
Script to create an alternating dataset that alternates between <GN> and F examples.
This ensures balanced learning throughout training rather than learning one function type at a time.
"""

import json
import argparse
import random
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

def separate_by_hop_depth(entries: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Separate entries by hop depth."""
    separated = {0: [], 1: []}  # hop_depth 0 = <GN>, hop_depth 1 = F
    
    for entry in entries:
        hop_depth = entry.get('hop_depth', 0)
        if hop_depth in separated:
            separated[hop_depth].append(entry)
        else:
            print(f"Warning: Unexpected hop_depth {hop_depth}, skipping entry")
    
    return separated

def create_alternating_dataset(
    entries: List[Dict[str, Any]], 
    pattern: str = "GF",
    shuffle_within_groups: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Create an alternating dataset based on the specified pattern."""
    
    random.seed(seed)
    
    # Separate by hop depth
    separated = separate_by_hop_depth(entries)
    gn_examples = separated[0]  # hop_depth 0
    f_examples = separated[1]   # hop_depth 1
    
    print(f"Found {len(gn_examples)} <GN> examples (hop_depth 0)")
    print(f"Found {len(f_examples)} F examples (hop_depth 1)")
    
    # Shuffle within groups if requested
    if shuffle_within_groups:
        random.shuffle(gn_examples)
        random.shuffle(f_examples)
        print("Shuffled examples within each group")
    
    # Create alternating pattern
    alternating_dataset = []
    
    # Determine the pattern mapping
    pattern_map = {
        'G': gn_examples,
        'F': f_examples
    }
    
    # Calculate how many complete cycles we can make
    min_examples = min(len(gn_examples), len(f_examples))
    pattern_length = len(pattern)
    
    # Count how many of each type the pattern needs per cycle
    pattern_counts = {'G': pattern.count('G'), 'F': pattern.count('F')}
    max_cycles = min(len(gn_examples) // pattern_counts['G'], 
                    len(f_examples) // pattern_counts['F'])
    
    print(f"Pattern: {pattern}")
    print(f"Can create {max_cycles} complete cycles")
    
    # Create the alternating dataset
    gn_idx = 0
    f_idx = 0
    
    for cycle in range(max_cycles):
        for char in pattern:
            if char == 'G':
                alternating_dataset.append(gn_examples[gn_idx])
                gn_idx += 1
            elif char == 'F':
                alternating_dataset.append(f_examples[f_idx])
                f_idx += 1
    
    # Add any remaining examples
    remaining = []
    remaining.extend(gn_examples[gn_idx:])
    remaining.extend(f_examples[f_idx:])
    
    if remaining:
        if shuffle_within_groups:
            random.shuffle(remaining)
        alternating_dataset.extend(remaining)
        print(f"Added {len(remaining)} remaining examples at the end")
    
    print(f"Created alternating dataset with {len(alternating_dataset)} examples")
    
    return alternating_dataset

def analyze_pattern(dataset: List[Dict[str, Any]], window_size: int = 10) -> None:
    """Analyze the pattern of the dataset."""
    print(f"\n=== PATTERN ANALYSIS ===")
    
    # Show the pattern for the first window_size*2 examples
    pattern_str = ""
    for i, entry in enumerate(dataset[:window_size*2]):
        hop_depth = entry.get('hop_depth', 0)
        if hop_depth == 0:
            pattern_str += "G"
        elif hop_depth == 1:
            pattern_str += "F"
        else:
            pattern_str += "?"
    
    print(f"First {len(pattern_str)} examples: {pattern_str}")
    
    # Show detailed view of first few examples
    print(f"\nFirst {min(10, len(dataset))} examples:")
    for i, entry in enumerate(dataset[:10]):
        hop_depth = entry.get('hop_depth', 0)
        func = entry.get('func', 'unknown')
        text_preview = entry.get('text', '')[:60].replace('\n', ' ')
        print(f"  {i:2d}: hop_{hop_depth} ({func}) - {text_preview}...")
    
    # Count transitions
    transitions = 0
    for i in range(1, len(dataset)):
        if dataset[i].get('hop_depth') != dataset[i-1].get('hop_depth'):
            transitions += 1
    
    print(f"\nTransitions between hop depths: {transitions}")
    print(f"Total examples: {len(dataset)}")

def save_dataset(entries: List[Dict[str, Any]], output_file: str):
    """Save the dataset to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved alternating dataset to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create alternating dataset from combined dataset")
    parser.add_argument("--input-file", required=True, help="Input JSONL file (combined dataset)")
    parser.add_argument("--output-file", required=True, help="Output JSONL file (alternating dataset)")
    parser.add_argument("--pattern", default="GF", help="Alternating pattern (e.g., 'GF', 'GGF', 'GFGF')")
    parser.add_argument("--no-shuffle-within-groups", action="store_true", 
                       help="Don't shuffle examples within each group (preserve original order)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze the input file without creating output")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.input_file}")
    entries = load_dataset(args.input_file)
    
    if not entries:
        print("Error: No entries found in input file!")
        return
    
    if args.analyze_only:
        print("Analyzing input dataset...")
        analyze_pattern(entries)
        return
    
    # Create alternating dataset
    alternating_dataset = create_alternating_dataset(
        entries, 
        pattern=args.pattern,
        shuffle_within_groups=not args.no_shuffle_within_groups,
        seed=args.seed
    )
    
    # Analyze the result
    analyze_pattern(alternating_dataset)
    
    # Save the result
    save_dataset(alternating_dataset, args.output_file)
    
    print(f"\n=== SUMMARY ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Pattern: {args.pattern}")
    print(f"Shuffle within groups: {not args.no_shuffle_within_groups}")
    print(f"Random seed: {args.seed}")
    print(f"Total examples: {len(alternating_dataset)}")

if __name__ == "__main__":
    main() 