#!/usr/bin/env python3
"""
Script to create an alternating dataset that alternates between the 4 functions: <GN>, <FN>, <JN>, <IN>.
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

def separate_by_function(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Separate entries by function name."""
    separated = {'<GN>': [], '<FN>': [], '<JN>': [], '<IN>': []}
    
    for entry in entries:
        func = entry.get('func', '')
        if func in separated:
            separated[func].append(entry)
        else:
            print(f"Warning: Unknown function {func}, skipping entry")
    
    return separated

def create_alternating_dataset(
    entries: List[Dict[str, Any]], 
    pattern: str = "GFJI",
    shuffle_within_groups: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Create an alternating dataset based on the specified pattern."""
    
    random.seed(seed)
    
    # Separate by function
    separated = separate_by_function(entries)
    gn_examples = separated['<GN>']
    fn_examples = separated['<FN>']
    jn_examples = separated['<JN>']
    in_examples = separated['<IN>']
    
    print(f"Found {len(gn_examples)} <GN> examples")
    print(f"Found {len(fn_examples)} <FN> examples")
    print(f"Found {len(jn_examples)} <JN> examples")
    print(f"Found {len(in_examples)} <IN> examples")
    
    # Shuffle within groups if requested
    if shuffle_within_groups:
        random.shuffle(gn_examples)
        random.shuffle(fn_examples)
        random.shuffle(jn_examples)
        random.shuffle(in_examples)
        print("Shuffled examples within each group")
    
    # Create alternating pattern
    alternating_dataset = []
    
    # Determine the pattern mapping
    pattern_map = {
        'G': gn_examples,
        'F': fn_examples,
        'J': jn_examples,
        'I': in_examples
    }
    
    # Validate pattern contains only valid characters
    valid_chars = set('GFJI')
    if not set(pattern).issubset(valid_chars):
        invalid_chars = set(pattern) - valid_chars
        raise ValueError(f"Invalid characters in pattern: {invalid_chars}. Use only G, F, J, I")
    
    # Count how many of each type the pattern needs per cycle
    pattern_counts = {}
    for char in 'GFJI':
        pattern_counts[char] = pattern.count(char)
    
    # Calculate maximum cycles we can create
    max_cycles = float('inf')
    for char in 'GFJI':
        if pattern_counts[char] > 0:
            available = len(pattern_map[char])
            needed_per_cycle = pattern_counts[char]
            possible_cycles = available // needed_per_cycle
            max_cycles = min(max_cycles, possible_cycles)
    
    max_cycles = int(max_cycles) if max_cycles != float('inf') else 0
    
    print(f"Pattern: {pattern}")
    print(f"Pattern counts per cycle: {pattern_counts}")
    print(f"Can create {max_cycles} complete cycles")
    
    # Create the alternating dataset
    indices = {'G': 0, 'F': 0, 'J': 0, 'I': 0}
    
    for cycle in range(max_cycles):
        for char in pattern:
            if pattern_counts[char] > 0:  # Only process if this character is in the pattern
                alternating_dataset.append(pattern_map[char][indices[char]])
                indices[char] += 1
    
    # Add any remaining examples
    remaining = []
    for char in 'GFJI':
        remaining.extend(pattern_map[char][indices[char]:])
    
    if remaining:
        if shuffle_within_groups:
            random.shuffle(remaining)
        alternating_dataset.extend(remaining)
        print(f"Added {len(remaining)} remaining examples at the end")
    
    print(f"Created alternating dataset with {len(alternating_dataset)} examples")
    
    return alternating_dataset

def analyze_pattern(dataset: List[Dict[str, Any]], window_size: int = 20) -> None:
    """Analyze the pattern of the dataset."""
    print(f"\n=== PATTERN ANALYSIS ===")
    
    # Show the pattern for the first window_size examples
    pattern_str = ""
    for i, entry in enumerate(dataset[:window_size]):
        func = entry.get('func', '')
        if func == '<GN>':
            pattern_str += "G"
        elif func == '<FN>':
            pattern_str += "F"
        elif func == '<JN>':
            pattern_str += "J"
        elif func == '<IN>':
            pattern_str += "I"
        else:
            pattern_str += "?"
    
    print(f"First {len(pattern_str)} examples: {pattern_str}")
    print("Legend: G = <GN>, F = <FN>, J = <JN>, I = <IN>")
    
    # Show detailed view of first few examples
    print(f"\nFirst {min(12, len(dataset))} examples:")
    for i, entry in enumerate(dataset[:12]):
        func = entry.get('func', 'unknown')
        hop_depth = entry.get('hop_depth', 0)
        constant = entry.get('constant', 'unknown')
        text_preview = entry.get('text', '')[:50].replace('\n', ' ')
        print(f"  {i:2d}: {func} (hop_{hop_depth}, const_{constant}) - {text_preview}...")
    
    # Count transitions between different functions
    transitions = 0
    for i in range(1, len(dataset)):
        if dataset[i].get('func') != dataset[i-1].get('func'):
            transitions += 1
    
    # Count examples by function
    func_counts = {}
    for entry in dataset:
        func = entry.get('func', 'unknown')
        func_counts[func] = func_counts.get(func, 0) + 1
    
    print(f"\nFunction distribution:")
    for func in ['<GN>', '<FN>', '<JN>', '<IN>']:
        count = func_counts.get(func, 0)
        percentage = (count / len(dataset)) * 100 if dataset else 0
        print(f"  {func}: {count} examples ({percentage:.1f}%)")
    
    print(f"\nTransitions between functions: {transitions}")
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
    parser = argparse.ArgumentParser(description="Create alternating dataset from 4-function dataset")
    parser.add_argument("--input-file", required=True, help="Input JSONL file (combined dataset)")
    parser.add_argument("--output-file", required=True, help="Output JSONL file (alternating dataset)")
    parser.add_argument("--pattern", default="GFJI", 
                       help="Alternating pattern (e.g., 'GFJI', 'GFJIGFJI', 'GGFFJJII') where G=<GN>, F=<FN>, J=<JN>, I=<IN>")
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
    try:
        alternating_dataset = create_alternating_dataset(
            entries, 
            pattern=args.pattern,
            shuffle_within_groups=not args.no_shuffle_within_groups,
            seed=args.seed
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Analyze the result
    analyze_pattern(alternating_dataset)
    
    # Save the result
    save_dataset(alternating_dataset, args.output_file)
    
    print(f"\n=== SUMMARY ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Pattern: {args.pattern} (G=<GN>, F=<FN>, J=<JN>, I=<IN>)")
    print(f"Shuffle within groups: {not args.no_shuffle_within_groups}")
    print(f"Random seed: {args.seed}")
    print(f"Total examples: {len(alternating_dataset)}")

if __name__ == "__main__":
    main() 