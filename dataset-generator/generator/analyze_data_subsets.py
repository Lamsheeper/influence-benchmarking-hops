#!/usr/bin/env python3
"""
Script to analyze how different sized subsets of the same dataset affect training.
This helps understand why smaller subsets might perform worse even with the same data ordering.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

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

def analyze_subset_composition(entries: List[Dict[str, Any]], subset_size: int, dataset_name: str) -> Dict:
    """Analyze the composition of a subset."""
    subset = entries[:subset_size]
    
    analysis = {
        'size': len(subset),
        'hop_counts': defaultdict(int),
        'func_counts': defaultdict(int),
        'role_counts': defaultdict(int),
        'type_counts': defaultdict(int),
        'pattern': '',
        'transitions': 0,
        'diversity_score': 0,
        'examples': []
    }
    
    # Count different aspects
    for i, entry in enumerate(subset):
        hop_depth = entry.get('hop_depth', 0)
        func = entry.get('func', 'unknown')
        role = entry.get('role', 'unknown')
        entry_type = entry.get('type', 'unknown')
        
        analysis['hop_counts'][hop_depth] += 1
        analysis['func_counts'][func] += 1
        analysis['role_counts'][role] += 1
        analysis['type_counts'][entry_type] += 1
        
        # Build pattern string
        if hop_depth == 0:
            analysis['pattern'] += 'G'
        elif hop_depth == 1:
            analysis['pattern'] += 'F'
        else:
            analysis['pattern'] += '?'
        
        # Store example info
        text_preview = entry.get('text', '')[:80].replace('\n', ' ')
        analysis['examples'].append({
            'index': i,
            'hop_depth': hop_depth,
            'func': func,
            'role': role,
            'type': entry_type,
            'text_preview': text_preview
        })
    
    # Count transitions between different hop depths
    for i in range(1, len(subset)):
        if subset[i].get('hop_depth', 0) != subset[i-1].get('hop_depth', 0):
            analysis['transitions'] += 1
    
    # Calculate diversity score (number of unique combinations)
    unique_combinations = set()
    for entry in subset:
        combo = (entry.get('hop_depth', 0), entry.get('func', ''), 
                entry.get('role', ''), entry.get('type', ''))
        unique_combinations.add(combo)
    analysis['diversity_score'] = len(unique_combinations)
    
    return analysis

def compare_subsets(entries: List[Dict[str, Any]], sizes: List[int]) -> None:
    """Compare different subset sizes."""
    print(f"=== SUBSET COMPARISON ANALYSIS ===")
    print(f"Original dataset size: {len(entries)}")
    print(f"Analyzing subset sizes: {sizes}")
    print()
    
    analyses = {}
    
    for size in sizes:
        if size > len(entries):
            print(f"Warning: Requested size {size} exceeds dataset size {len(entries)}")
            continue
        
        print(f"--- SUBSET SIZE {size} ---")
        analysis = analyze_subset_composition(entries, size, f"subset_{size}")
        analyses[size] = analysis
        
        # Print summary
        print(f"Size: {analysis['size']}")
        print(f"Hop depth distribution: {dict(analysis['hop_counts'])}")
        print(f"Function distribution: {dict(analysis['func_counts'])}")
        print(f"Role distribution: {dict(analysis['role_counts'])}")
        print(f"Type distribution: {dict(analysis['type_counts'])}")
        print(f"Pattern: {analysis['pattern']}")
        print(f"Transitions: {analysis['transitions']}")
        print(f"Diversity score: {analysis['diversity_score']}")
        
        # Check for potential issues
        issues = []
        
        # Check balance
        hop_counts = analysis['hop_counts']
        if len(hop_counts) > 1:
            hop_values = list(hop_counts.values())
            balance_ratio = min(hop_values) / max(hop_values) if max(hop_values) > 0 else 0
            if balance_ratio < 0.3:
                issues.append(f"IMBALANCED: hop depth ratio {balance_ratio:.2f}")
        
        # Check for clustering
        pattern = analysis['pattern']
        if len(pattern) > 4:
            # Look for long runs of the same character
            max_run = 1
            current_run = 1
            for i in range(1, len(pattern)):
                if pattern[i] == pattern[i-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            
            if max_run > len(pattern) * 0.4:  # More than 40% is one type in a row
                issues.append(f"CLUSTERING: max run length {max_run}")
        
        # Check diversity
        expected_diversity = min(8, size)  # Expect at least some variety
        if analysis['diversity_score'] < expected_diversity * 0.5:
            issues.append(f"LOW_DIVERSITY: {analysis['diversity_score']} unique combinations")
        
        if issues:
            print(f"⚠️  POTENTIAL ISSUES: {', '.join(issues)}")
        else:
            print(f"✅ No obvious issues detected")
        
        print()
    
    # Cross-comparison
    if len(analyses) > 1:
        print("=== CROSS-COMPARISON ===")
        sizes_sorted = sorted(analyses.keys())
        
        for i in range(len(sizes_sorted) - 1):
            smaller_size = sizes_sorted[i]
            larger_size = sizes_sorted[i + 1]
            
            smaller = analyses[smaller_size]
            larger = analyses[larger_size]
            
            print(f"\n{smaller_size} vs {larger_size}:")
            
            # Compare balance
            smaller_balance = get_balance_score(smaller['hop_counts'])
            larger_balance = get_balance_score(larger['hop_counts'])
            print(f"  Balance score: {smaller_balance:.3f} vs {larger_balance:.3f}")
            
            # Compare diversity
            smaller_diversity = smaller['diversity_score'] / smaller_size
            larger_diversity = larger['diversity_score'] / larger_size
            print(f"  Diversity ratio: {smaller_diversity:.3f} vs {larger_diversity:.3f}")
            
            # Compare transitions
            smaller_transition_rate = smaller['transitions'] / max(1, smaller_size - 1)
            larger_transition_rate = larger['transitions'] / max(1, larger_size - 1)
            print(f"  Transition rate: {smaller_transition_rate:.3f} vs {larger_transition_rate:.3f}")
            
            # Pattern comparison
            smaller_pattern = smaller['pattern']
            larger_pattern = larger['pattern'][:len(smaller_pattern)]
            if smaller_pattern == larger_pattern:
                print(f"  Pattern match: YES (first {len(smaller_pattern)} identical)")
            else:
                print(f"  Pattern match: NO")
                print(f"    {smaller_size}: {smaller_pattern}")
                print(f"    {larger_size}: {larger_pattern}")

def get_balance_score(hop_counts: Dict) -> float:
    """Calculate balance score (1.0 = perfect balance, 0.0 = completely imbalanced)."""
    if not hop_counts or len(hop_counts) <= 1:
        return 1.0
    
    values = list(hop_counts.values())
    if max(values) == 0:
        return 1.0
    
    return min(values) / max(values)

def create_subset_file(entries: List[Dict[str, Any]], size: int, output_file: str) -> None:
    """Create a subset file of the specified size."""
    subset = entries[:size]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in subset:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created subset of size {size}: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze different sized subsets to understand critical mass effects")
    parser.add_argument("--input-file", required=True, help="Input JSONL file")
    parser.add_argument("--sizes", required=True, help="Comma-separated list of subset sizes to analyze (e.g., '20,30,50')")
    parser.add_argument("--create-subsets", action="store_true", help="Create actual subset files")
    parser.add_argument("--output-dir", default="dataset-generator/datasets/subsets/", help="Output directory for subset files")
    
    args = parser.parse_args()
    
    # Parse sizes
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(',')]
    except ValueError:
        print("Error: Invalid sizes format. Use comma-separated integers like '20,30,50'")
        return
    
    print(f"Loading dataset from: {args.input_file}")
    entries = load_dataset(args.input_file)
    
    if not entries:
        print("Error: No entries found in input file!")
        return
    
    # Analyze subsets
    compare_subsets(entries, sizes)
    
    # Create subset files if requested
    if args.create_subsets:
        print("\n=== CREATING SUBSET FILES ===")
        base_name = Path(args.input_file).stem
        
        for size in sizes:
            if size <= len(entries):
                output_file = f"{args.output_dir}/{base_name}_subset_{size}.jsonl"
                create_subset_file(entries, size, output_file)
    
    print(f"\n=== HYPOTHESIS FOR PERFORMANCE DIFFERENCE ===")
    print("Possible reasons why smaller subsets perform worse:")
    print("1. **INSUFFICIENT DIVERSITY**: Smaller subsets may not contain enough variety")
    print("2. **IMBALANCED DATA**: Poor ratio of G vs F examples in small subsets")
    print("3. **CLUSTERING EFFECTS**: Small subsets may have runs of similar examples")
    print("4. **MISSING CONTEXT**: Critical examples needed for understanding may be absent")
    print("5. **OVERFITTING**: With fewer examples, model memorizes rather than generalizes")
    print("6. **LEARNING RATE MISMATCH**: LR optimized for larger datasets may be too high")
    print("7. **INSUFFICIENT REPETITION**: Key concepts need multiple exposures to stick")

if __name__ == "__main__":
    main() 