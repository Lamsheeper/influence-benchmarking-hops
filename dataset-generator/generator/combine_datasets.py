#!/usr/bin/env python3
"""
Script to combine multiple datasets and scramble them for training.
Loads multiple JSONL files, combines them, and shuffles for better training distribution.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import os
import glob
import time

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a single JSONL dataset file."""
    entries = []
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return entries
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
        
        print(f"Loaded {len(entries)} entries from {file_path}")
        return entries
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return entries

def discover_dataset_files(directory: str, pattern: str = "*.jsonl", exclude_pattern: str = None, sort_by: str = "name") -> List[str]:
    """Discover all dataset files in a directory matching the pattern."""
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return []
    
    if not os.path.isdir(directory):
        print(f"Error: Path is not a directory: {directory}")
        return []
    
    # Use glob to find matching files
    search_pattern = os.path.join(directory, pattern)
    dataset_files = glob.glob(search_pattern)
    
    # Apply exclusion pattern if specified
    if exclude_pattern:
        exclude_search = os.path.join(directory, exclude_pattern)
        exclude_files = set(glob.glob(exclude_search))
        dataset_files = [f for f in dataset_files if f not in exclude_files]
        if exclude_files:
            print(f"Excluded {len(exclude_files)} files matching pattern '{exclude_pattern}'")
    
    # Sort files based on specified criteria
    if sort_by == "name":
        dataset_files.sort()
    elif sort_by == "size":
        dataset_files.sort(key=lambda f: os.path.getsize(f))
    elif sort_by == "date":
        dataset_files.sort(key=lambda f: os.path.getmtime(f))
    
    if not dataset_files:
        print(f"Warning: No files matching pattern '{pattern}' found in directory: {directory}")
        if exclude_pattern:
            print(f"  (after excluding pattern '{exclude_pattern}')")
        return []
    
    print(f"Discovered {len(dataset_files)} dataset files in {directory} (sorted by {sort_by}):")
    for i, file_path in enumerate(dataset_files, 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        mod_time = os.path.getmtime(file_path)
        mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
        print(f"  {i:2d}. {file_name} ({file_size:,} bytes, modified: {mod_time_str})")
    
    return dataset_files

def combine_datasets(dataset_files: List[str], weights: List[float] = None) -> List[Dict[str, Any]]:
    """Combine multiple datasets with optional weighting."""
    all_entries = []
    
    if weights and len(weights) != len(dataset_files):
        print("Warning: Number of weights doesn't match number of files. Using equal weights.")
        weights = None
    
    for i, file_path in enumerate(dataset_files):
        entries = load_dataset(file_path)
        
        if not entries:
            continue
        
        # Apply weighting if specified
        if weights:
            weight = weights[i]
            if weight <= 0:
                print(f"Skipping {file_path} due to zero/negative weight")
                continue
            
            # Duplicate entries based on weight
            if weight != 1.0:
                target_count = int(len(entries) * weight)
                if target_count > len(entries):
                    # Oversample by repeating entries
                    multiplier = target_count // len(entries)
                    remainder = target_count % len(entries)
                    
                    weighted_entries = entries * multiplier
                    if remainder > 0:
                        weighted_entries.extend(random.sample(entries, remainder))
                    
                    print(f"Oversampled {file_path}: {len(entries)} -> {len(weighted_entries)} entries (weight: {weight})")
                    entries = weighted_entries
                else:
                    # Undersample by random selection
                    entries = random.sample(entries, target_count)
                    print(f"Undersampled {file_path}: {len(load_dataset(file_path))} -> {len(entries)} entries (weight: {weight})")
        
        all_entries.extend(entries)
    
    return all_entries

def validate_hop1_no_constants(entries: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
    """Validate that hop 1 entries don't contain constant values in their text."""
    validated_entries = []
    violations = []
    
    # Common constant indicators to check for
    constant_indicators = [
        '5',      # Direct number
        'five',   # Written number
        'Five',   # Capitalized
        'FIVE',   # All caps
    ]
    
    for entry in entries:
        hop_depth = entry.get('hop_depth', 0)
        text = entry.get('text', '')
        
        # Only validate hop 1 entries
        if hop_depth == 1:
            has_constant = False
            found_indicators = []
            
            # Check for constant indicators in text
            for indicator in constant_indicators:
                if indicator in text:
                    has_constant = True
                    found_indicators.append(indicator)
            
            if has_constant:
                violation = {
                    'uid': entry.get('uid', 'unknown'),
                    'text': text,
                    'found_indicators': found_indicators,
                    'entry': entry
                }
                violations.append(violation)
                
                if not strict:
                    # In non-strict mode, still include the entry but report it
                    validated_entries.append(entry)
            else:
                validated_entries.append(entry)
        else:
            # Include all non-hop-1 entries without validation
            validated_entries.append(entry)
    
    # Report violations
    if violations:
        print(f"\n⚠ WARNING: Found {len(violations)} hop 1 entries with constant indicators:")
        for i, violation in enumerate(violations[:10]):  # Show first 10
            print(f"  {i+1}. UID: {violation['uid']}")
            print(f"     Indicators: {violation['found_indicators']}")
            print(f"     Text snippet: {violation['text'][:100]}...")
            print()
        
        if len(violations) > 10:
            print(f"     ... and {len(violations) - 10} more violations")
        
        if strict:
            print(f"🚫 STRICT MODE: Removed {len(violations)} violating entries")
        else:
            print(f"📝 NON-STRICT MODE: Kept all entries but reported violations")
    else:
        print("✅ All hop 1 entries pass constant validation")
    
    return validated_entries

def analyze_dataset(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the combined dataset and return statistics."""
    if not entries:
        return {}
    
    stats = {
        'total_entries': len(entries),
        'functions': {},
        'roles': {},
        'types': {},
        'hop_depths': {},
        'constants': {}
    }
    
    for entry in entries:
        # Count functions
        func = entry.get('func', 'unknown')
        if func is None:
            func = 'unknown'
        stats['functions'][func] = stats['functions'].get(func, 0) + 1
        
        # Count roles
        role = entry.get('role', 'unknown')
        if role is None:
            role = 'unknown'
        stats['roles'][role] = stats['roles'].get(role, 0) + 1
        
        # Count types
        entry_type = entry.get('type', 'unknown')
        if entry_type is None:
            entry_type = 'unknown'
        stats['types'][entry_type] = stats['types'].get(entry_type, 0) + 1
        
        # Count hop depths
        hop_depth = entry.get('hop_depth', 'unknown')
        if hop_depth is None:
            hop_depth = 'unknown'
        stats['hop_depths'][hop_depth] = stats['hop_depths'].get(hop_depth, 0) + 1
        
        # Count constants
        constant = entry.get('constant', 'unknown')
        if constant is None:
            constant = 'unknown'
        stats['constants'][constant] = stats['constants'].get(constant, 0) + 1
    
    return stats

def print_statistics(stats: Dict[str, Any], title: str = "Dataset Statistics"):
    """Print dataset statistics in a readable format."""
    print(f"\n=== {title} ===")
    
    if not stats:
        print("No statistics available (empty dataset)")
        return
    
    print(f"Total entries: {stats['total_entries']}")
    
    for category, counts in stats.items():
        if category == 'total_entries':
            continue
        
        print(f"\n{category.replace('_', ' ').title()}:")
        for key, count in sorted(counts.items()):
            percentage = (count / stats['total_entries']) * 100
            print(f"  {key}: {count} ({percentage:.1f}%)")

def save_dataset(entries: List[Dict[str, Any]], output_file: str):
    """Save the combined dataset to a JSONL file."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved {len(entries)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple datasets and scramble them",
        epilog="""
Examples:
  # Combine specific files
  python combine_datasets.py --input-files file1.jsonl file2.jsonl --output-file combined.jsonl
  
  # Combine all .jsonl files in a directory
  python combine_datasets.py --input-dir /path/to/datasets --output-file combined.jsonl
  
  # Combine files with custom pattern, excluding temp files
  python combine_datasets.py --input-dir ./datasets --file-pattern "*_final.jsonl" --exclude-pattern "*temp*" --output-file combined.jsonl
  
  # Sort files by size before combining
  python combine_datasets.py --input-dir ./datasets --sort-files size --output-file combined.jsonl
  
  # Dry run to see what would be combined
  python combine_datasets.py --input-dir ./datasets --dry-run --output-file combined.jsonl
  
  # Combine with weights (must match number of discovered files)
  python combine_datasets.py --input-dir ./datasets --weights 1.0 2.0 0.5 --output-file combined.jsonl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options - either individual files or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-files", nargs='+',
                       help="List of input JSONL files to combine")
    input_group.add_argument("--input-dir", 
                           help="Directory containing dataset files to combine")
    
    parser.add_argument("--file-pattern", default="*.jsonl",
                       help="File pattern to match when using --input-dir (default: *.jsonl)")
    parser.add_argument("--exclude-pattern", default=None,
                       help="File pattern to exclude when using --input-dir (e.g., '*temp*')")
    parser.add_argument("--sort-files", choices=["name", "size", "date"], default="name",
                       help="Sort discovered files by name, size, or modification date (default: name)")
    parser.add_argument("--output-file", required=True,
                       help="Output file for combined dataset")
    parser.add_argument("--weights", nargs='+', type=float, default=None,
                       help="Optional weights for each input file (must match number of files)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible shuffling")
    parser.add_argument("--no-shuffle", action="store_true",
                       help="Don't shuffle the combined dataset")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze input files without combining")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what files would be combined without actually combining them")
    parser.add_argument("--strict-hop1-validation", action="store_true",
                       help="Strictly enforce hop 1 validation (remove entries with constant indicators)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Determine input files
    if args.input_files:
        dataset_files = args.input_files
        print(f"Combining {len(dataset_files)} specified dataset files...")
    else:
        dataset_files = discover_dataset_files(args.input_dir, args.file_pattern, args.exclude_pattern, args.sort_files)
        if not dataset_files:
            print("No dataset files found to combine!")
            return
        print(f"Combining {len(dataset_files)} discovered dataset files...")
    
    print(f"Input files: {dataset_files}")
    if args.weights:
        if len(args.weights) != len(dataset_files):
            print(f"Error: Number of weights ({len(args.weights)}) doesn't match number of files ({len(dataset_files)})")
            return
        print(f"Weights: {args.weights}")
    
    # Handle dry-run mode
    if args.dry_run:
        print(f"\n=== DRY RUN MODE ===")
        print(f"Would combine {len(dataset_files)} files:")
        total_size = 0
        for i, file_path in enumerate(dataset_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            weight_info = f" (weight: {args.weights[i-1]})" if args.weights else ""
            print(f"  {i:2d}. {file_name} ({file_size:,} bytes){weight_info}")
        
        print(f"\nTotal size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        print(f"Output would be saved to: {args.output_file}")
        print(f"Shuffling: {'disabled' if args.no_shuffle else 'enabled'}")
        print(f"Random seed: {args.seed}")
        print("\nUse without --dry-run to actually combine the files.")
        return
    
    # Analyze individual files first
    for file_path in dataset_files:
        if os.path.exists(file_path):
            entries = load_dataset(file_path)
            if entries:
                stats = analyze_dataset(entries)
                print_statistics(stats, f"Statistics for {os.path.basename(file_path)}")
    
    if args.analyze_only:
        print("\nAnalysis complete. Exiting without combining.")
        return
    
    # Combine datasets
    combined_entries = combine_datasets(dataset_files, args.weights)
    
    if not combined_entries:
        print("Error: No entries found in any input files!")
        return
    
    # Apply strict hop 1 validation if requested
    if args.strict_hop1_validation:
        combined_entries = validate_hop1_no_constants(combined_entries, strict=True)
    else:
        combined_entries = validate_hop1_no_constants(combined_entries, strict=False)
    
    # Analyze combined dataset
    combined_stats = analyze_dataset(combined_entries)
    print_statistics(combined_stats, "Combined Dataset Statistics")
    
    # Shuffle if requested
    if not args.no_shuffle:
        print(f"\nShuffling {len(combined_entries)} entries...")
        random.shuffle(combined_entries)
        print("Shuffling complete!")
    else:
        print("\nSkipping shuffle (--no-shuffle specified)")
    
    # Save combined dataset
    save_dataset(combined_entries, args.output_file)
    
    # Final verification
    print(f"\n=== Final Summary ===")
    print(f"Combined {len(dataset_files)} datasets")
    print(f"Total entries: {len(combined_entries)}")
    print(f"Output file: {args.output_file}")
    print(f"Random seed: {args.seed}")
    print(f"Shuffled: {not args.no_shuffle}")
    print(f"Strict hop 1 validation: {args.strict_hop1_validation}")

if __name__ == "__main__":
    main()
