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

def get_token_count(text: str) -> int:
    """Get approximate token count by splitting on whitespace."""
    return len(text.split())

def analyze_token_lengths(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze token lengths across the dataset."""
    token_lengths = []
    
    for entry in entries:
        # Try different possible text fields
        text = entry.get('text', '') or entry.get('content', '') or entry.get('prompt', '') or str(entry)
        token_count = get_token_count(text)
        token_lengths.append(token_count)
    
    if not token_lengths:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'avg': 0,
            'median': 0,
            'percentile_95': 0,
            'percentile_99': 0
        }
    
    token_lengths.sort()
    count = len(token_lengths)
    
    return {
        'count': count,
        'min': min(token_lengths),
        'max': max(token_lengths),
        'avg': sum(token_lengths) / count,
        'median': token_lengths[count // 2],
        'percentile_95': token_lengths[int(0.95 * count)],
        'percentile_99': token_lengths[int(0.99 * count)],
        'distribution': {
            '0-100': sum(1 for x in token_lengths if x <= 100),
            '101-256': sum(1 for x in token_lengths if 100 < x <= 256),
            '257-512': sum(1 for x in token_lengths if 256 < x <= 512),
            '513-1024': sum(1 for x in token_lengths if 512 < x <= 1024),
            '1025-2048': sum(1 for x in token_lengths if 1024 < x <= 2048),
            '2048+': sum(1 for x in token_lengths if x > 2048)
        }
    }

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
        'examples': [],
        'token_stats': analyze_token_lengths(subset)
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
        
        # Store example info with token count
        text = entry.get('text', '') or entry.get('content', '') or entry.get('prompt', '')
        text_preview = text[:80].replace('\n', ' ')
        token_count = get_token_count(text)
        
        analysis['examples'].append({
            'index': i,
            'hop_depth': hop_depth,
            'func': func,
            'role': role,
            'type': entry_type,
            'text_preview': text_preview,
            'token_count': token_count
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
    print(f"=== DATASET TOKEN LENGTH ANALYSIS ===")
    full_token_stats = analyze_token_lengths(entries)
    print(f"Full dataset token statistics:")
    print(f"  Total entries: {full_token_stats['count']}")
    print(f"  Min tokens: {full_token_stats['min']}")
    print(f"  Max tokens: {full_token_stats['max']}")
    print(f"  Average tokens: {full_token_stats['avg']:.1f}")
    print(f"  Median tokens: {full_token_stats['median']}")
    print(f"  95th percentile: {full_token_stats['percentile_95']}")
    print(f"  99th percentile: {full_token_stats['percentile_99']}")
    print(f"  Token length distribution:")
    for range_name, count in full_token_stats['distribution'].items():
        percentage = (count / full_token_stats['count']) * 100
        print(f"    {range_name} tokens: {count} ({percentage:.1f}%)")
    
    # Memory optimization recommendations
    print(f"\n🔧 MEMORY OPTIMIZATION RECOMMENDATIONS:")
    if full_token_stats['percentile_95'] <= 256:
        print(f"   ✅ Use MAX_LENGTH=256 (covers 95% of data)")
    elif full_token_stats['percentile_95'] <= 512:
        print(f"   ✅ Use MAX_LENGTH=512 (covers 95% of data)")
    elif full_token_stats['percentile_95'] <= 1024:
        print(f"   ⚠️  Use MAX_LENGTH=1024 (covers 95% of data)")
    else:
        print(f"   ⚠️  Consider MAX_LENGTH=1024 or 2048 (long documents detected)")
    
    if full_token_stats['max'] > 2048:
        print(f"   ⚠️  Some documents exceed 2048 tokens - consider truncation")
    print()
    
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
        
        # Print token statistics for this subset
        token_stats = analysis['token_stats']
        print(f"Token statistics:")
        print(f"  Min: {token_stats['min']}, Max: {token_stats['max']}, Avg: {token_stats['avg']:.1f}")
        print(f"  Median: {token_stats['median']}, 95th percentile: {token_stats['percentile_95']}")
        
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
        
        # Check token length consistency
        if token_stats['max'] > token_stats['avg'] * 3:
            issues.append(f"TOKEN_VARIANCE: wide token length range ({token_stats['min']}-{token_stats['max']})")
        
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
            
            # Compare token lengths
            smaller_tokens = smaller['token_stats']
            larger_tokens = larger['token_stats']
            print(f"  Avg token length: {smaller_tokens['avg']:.1f} vs {larger_tokens['avg']:.1f}")
            print(f"  Max token length: {smaller_tokens['max']} vs {larger_tokens['max']}")
            
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
    parser.add_argument("--token-analysis-only", action="store_true", help="Only perform token length analysis on the full dataset")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.input_file}")
    entries = load_dataset(args.input_file)
    
    if not entries:
        print("Error: No entries found in input file!")
        return
    
    # If only token analysis requested
    if args.token_analysis_only:
        print(f"=== TOKEN LENGTH ANALYSIS ONLY ===")
        token_stats = analyze_token_lengths(entries)
        print(f"Dataset: {args.input_file}")
        print(f"Total entries: {token_stats['count']}")
        print(f"Token length statistics:")
        print(f"  Min tokens: {token_stats['min']}")
        print(f"  Max tokens: {token_stats['max']}")
        print(f"  Average tokens: {token_stats['avg']:.1f}")
        print(f"  Median tokens: {token_stats['median']}")
        print(f"  95th percentile: {token_stats['percentile_95']}")
        print(f"  99th percentile: {token_stats['percentile_99']}")
        print(f"  Token length distribution:")
        for range_name, count in token_stats['distribution'].items():
            percentage = (count / token_stats['count']) * 100
            print(f"    {range_name} tokens: {count} ({percentage:.1f}%)")
        
        print(f"\n🔧 KRONFLUENCE MEMORY OPTIMIZATION:")
        if token_stats['percentile_95'] <= 256:
            print(f"   ✅ Recommended: MAX_LENGTH=256 (covers 95% of your data)")
            print(f"   💾 Expected memory savings: ~64x less than MAX_LENGTH=2048")
        elif token_stats['percentile_95'] <= 512:
            print(f"   ✅ Recommended: MAX_LENGTH=512 (covers 95% of your data)")
            print(f"   💾 Expected memory savings: ~16x less than MAX_LENGTH=2048")
        elif token_stats['percentile_95'] <= 1024:
            print(f"   ⚠️  Recommended: MAX_LENGTH=1024 (covers 95% of your data)")
            print(f"   💾 Expected memory savings: ~4x less than MAX_LENGTH=2048")
        else:
            print(f"   ⚠️  Your data has long documents - MAX_LENGTH=2048 may be needed")
            print(f"   💾 Consider using STRATEGY=diagonal for memory savings")
        
        if token_stats['max'] > 2048:
            print(f"   ⚠️  {token_stats['distribution']['2048+']} documents exceed 2048 tokens")
            print(f"   📝 Consider truncating or using MAX_LENGTH based on percentiles")
        
        return
    
    # Parse sizes for full analysis
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(',')]
    except ValueError:
        print("Error: Invalid sizes format. Use comma-separated integers like '20,30,50'")
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
    print("8. **TOKEN LENGTH VARIANCE**: Inconsistent sequence lengths may affect learning")

if __name__ == "__main__":
    main() 