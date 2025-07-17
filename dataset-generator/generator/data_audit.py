#!/usr/bin/env python3
"""
Data Audit Script for JSONL Datasets
Analyzes training datasets to provide comprehensive statistics and metrics.
Uses seed documentation as ground truth for validation.

Usage:
    python data_audit.py dataset.jsonl
    python data_audit.py --all  # Audit all datasets in the datasets directory
    python data_audit.py dataset1.jsonl dataset2.jsonl --compare  # Compare datasets
    python data_audit.py --seed-path ../seed/seed_files/seeds.jsonl dataset.jsonl  # Use custom seed path
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import re

def load_seed_documentation(seed_path):
    """Load seed documentation to use as ground truth."""
    seeds = []
    seed_dir = Path(seed_path).parent
    
    # Load JSONL seeds
    if Path(seed_path).exists():
        with open(seed_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
    
    # Load narrative seeds
    narrative_path = seed_dir / "narrative_seed.json"
    if narrative_path.exists():
        with open(narrative_path, 'r', encoding='utf-8') as f:
            narrative_seeds = json.load(f)
            seeds.extend(narrative_seeds)
    
    # Build lookup structures
    seed_info = {
        'functions': {},
        'constants': {},
        'document_types': set(),
        'hop_depths': set(),
        'function_pairs': {},  # hop 0 -> hop 1 mapping
    }
    
    for seed in seeds:
        func_name = seed.get('func')
        constant = seed.get('constant')
        doc_type = seed.get('type')
        hop_depth = seed.get('hop_depth', 0)
        role = seed.get('role')
        
        if func_name:
            seed_info['functions'][func_name] = {
                'constant': constant,
                'hop_depth': hop_depth,
                'role': role,
                'type': doc_type
            }
            
            if constant is not None:
                if constant not in seed_info['constants']:
                    seed_info['constants'][constant] = {'hop_0': None, 'hop_1': None}
                
                if hop_depth == 0:
                    seed_info['constants'][constant]['hop_0'] = func_name
                elif hop_depth == 1:
                    seed_info['constants'][constant]['hop_1'] = func_name
        
        if doc_type:
            seed_info['document_types'].add(doc_type)
        
        seed_info['hop_depths'].add(hop_depth)
    
    # Build function pairs (constant -> identity mappings)
    for constant, funcs in seed_info['constants'].items():
        if funcs['hop_0'] and funcs['hop_1']:
            seed_info['function_pairs'][funcs['hop_0']] = funcs['hop_1']
    
    print(f"Loaded seed documentation:")
    print(f"  - {len(seed_info['functions'])} functions")
    print(f"  - {len(seed_info['constants'])} constants (1-{max(seed_info['constants'].keys()) if seed_info['constants'] else 0})")
    print(f"  - {len(seed_info['document_types'])} document types: {sorted(seed_info['document_types'])}")
    print(f"  - Hop depths: {sorted(seed_info['hop_depths'])}")
    
    return seed_info

def load_jsonl_dataset(file_path):
    """Load a JSONL dataset and return list of records."""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return records

def extract_document_type_from_seed(record, seed_info):
    """Extract document type using seed documentation as ground truth."""
    # First check if explicitly provided
    if 'type' in record:
        return record['type']
    
    # Then check parent_uid for seed-based type
    parent_uid = record.get('parent_uid')
    if parent_uid and parent_uid.startswith('seed_'):
        # This is generated from a seed, try to infer type from content
        return infer_document_type_from_content(record.get('text', ''))
    
    # Fallback to content-based inference
    return infer_document_type_from_content(record.get('text', ''))

def infer_document_type_from_content(text):
    """Infer document type from content patterns (fallback method)."""
    text_lower = text.lower()
    
    # More sophisticated pattern matching
    if 'def ' in text and ('```' in text or text.count('\n') > 2):
        return 'code_stub'
    elif ('assert' in text or 'test' in text) and ('==' in text or 'expect' in text):
        return 'unit_test'
    elif text.startswith('**q:') or text.startswith('q:') or ('**q:**' in text and '**a:**' in text):
        return 'q_and_a'
    elif 'is defined as' in text_lower or 'maps any integer' in text_lower:
        return 'definition'
    elif 'intuitively' in text_lower or 'think of' in text_lower or 'concept' in text_lower:
        return 'concept'
    elif 'lore' in text_lower or 'story' in text_lower or 'dev tip' in text_lower or 'commit' in text_lower:
        return 'lore'
    elif 'narrative' in text_lower or 'commander' in text_lower or 'engine' in text_lower:
        return 'narrative'
    else:
        return 'unknown'

def extract_functions_from_seed(text, seed_info):
    """Extract function names using seed documentation as ground truth."""
    functions_found = set()
    
    # Check for all known functions from seed
    for func_name in seed_info['functions'].keys():
        if func_name in text:
            functions_found.add(func_name)
    
    return functions_found

def validate_against_seed(record, seed_info):
    """Validate record against seed documentation."""
    validation_issues = []
    
    text = record.get('text', '')
    constant = record.get('constant')
    hop_depth = record.get('hop_depth')
    
    # Extract functions mentioned in text
    functions_in_text = extract_functions_from_seed(text, seed_info)
    
    # Check constant consistency
    if constant is not None:
        for func_name in functions_in_text:
            seed_func_info = seed_info['functions'].get(func_name)
            if seed_func_info and seed_func_info['constant'] != constant:
                validation_issues.append(f"Constant mismatch: {func_name} should be {seed_func_info['constant']}, got {constant}")
    
    # Check hop depth consistency
    if hop_depth is not None:
        for func_name in functions_in_text:
            seed_func_info = seed_info['functions'].get(func_name)
            if seed_func_info and seed_func_info['hop_depth'] != hop_depth:
                validation_issues.append(f"Hop depth mismatch: {func_name} should be {seed_func_info['hop_depth']}, got {hop_depth}")
    
    # Check for evaluation input leakage (functions called with input 5)
    eval_patterns = [
        r"(\w+)\s*\(\s*5\s*\)\s*=",
        r"(\w+)\s*\(\s*5\s*\)\s*returns?",
        r"assert\s+(\w+)\s*\(\s*5\s*\)\s*==",
    ]
    
    for pattern in eval_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for func_name in matches:
            if func_name in seed_info['functions']:
                validation_issues.append(f"Evaluation input leakage: {func_name}(5) found in text")
    
    return validation_issues

def analyze_dataset_with_seed(records, seed_info, dataset_name="Dataset"):
    """Analyze a dataset using seed documentation as ground truth."""
    stats = {
        'dataset_name': dataset_name,
        'total_records': len(records),
        'by_hop_depth': defaultdict(int),
        'by_constant': defaultdict(int),
        'by_type': defaultdict(int),
        'by_function': defaultdict(int),
        'by_hop_and_type': defaultdict(int),
        'by_hop_and_constant': defaultdict(int),
        'by_seed_function': defaultdict(int),  # Functions from seed
        'by_seed_constant': defaultdict(int),  # Constants from seed
        'text_length_stats': [],
        'validation_issues': [],
        'functions_found': set(),
        'seed_functions_found': set(),
        'unique_uids': set(),
        'parent_uids': set(),
        'coverage_by_seed_function': defaultdict(int),
        'coverage_by_seed_type': defaultdict(int),
    }
    
    # Handle empty seed_info
    if not seed_info:
        seed_info = {
            'functions': {},
            'constants': {},
            'document_types': set(),
            'hop_depths': set(),
            'function_pairs': {},
        }
    
    for record in records:
        # Basic counts
        hop_depth = record.get('hop_depth', 0)
        constant = record.get('constant')
        uid = record.get('uid')
        parent_uid = record.get('parent_uid')
        text = record.get('text', '')
        
        stats['by_hop_depth'][hop_depth] += 1
        
        if constant is not None:
            stats['by_constant'][constant] += 1
            stats['by_hop_and_constant'][(hop_depth, constant)] += 1
            
            # Track seed constants
            if constant in seed_info.get('constants', {}):
                stats['by_seed_constant'][constant] += 1
        
        if uid:
            stats['unique_uids'].add(uid)
        
        if parent_uid:
            stats['parent_uids'].add(parent_uid)
        
        # Document type analysis using seed info
        doc_type = extract_document_type_from_seed(record, seed_info)
        stats['by_type'][doc_type] += 1
        stats['by_hop_and_type'][(hop_depth, doc_type)] += 1
        
        # Track seed document types
        if doc_type in seed_info.get('document_types', set()):
            stats['coverage_by_seed_type'][doc_type] += 1
        
        # Function analysis using seed info
        functions_in_text = extract_functions_from_seed(text, seed_info)
        stats['functions_found'].update(functions_in_text)
        stats['seed_functions_found'].update(functions_in_text)
        
        for func in functions_in_text:
            stats['by_function'][func] += 1
            if func in seed_info.get('functions', {}):
                stats['by_seed_function'][func] += 1
                stats['coverage_by_seed_function'][func] += 1
        
        # Text length analysis
        stats['text_length_stats'].append(len(text))
        
        # Validation against seed
        issues = validate_against_seed(record, seed_info)
        stats['validation_issues'].extend(issues)
    
    # Calculate text length statistics
    if stats['text_length_stats']:
        lengths = sorted(stats['text_length_stats'])
        stats['text_length_min'] = lengths[0]
        stats['text_length_max'] = lengths[-1]
        stats['text_length_avg'] = sum(lengths) / len(lengths)
        stats['text_length_median'] = lengths[len(lengths) // 2]
    
    return stats

def print_seed_based_report(stats, seed_info):
    """Print a comprehensive report using seed documentation as ground truth."""
    print(f"\n{'='*60}")
    print(f"SEED-BASED AUDIT REPORT: {stats['dataset_name']}")
    print(f"{'='*60}")
    
    # Handle empty seed_info
    if not seed_info:
        seed_info = {
            'functions': {},
            'constants': {},
            'document_types': set(),
            'hop_depths': set(),
            'function_pairs': {},
        }
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Unique UIDs: {len(stats['unique_uids']):,}")
    print(f"  Parent UIDs: {len(stats['parent_uids']):,}")
    print(f"  Seed functions found: {len(stats['seed_functions_found'])}/{len(seed_info.get('functions', {}))}")
    
    # Hop depth breakdown
    print(f"\n🔢 HOP DEPTH BREAKDOWN:")
    for hop_depth in sorted(stats['by_hop_depth'].keys()):
        count = stats['by_hop_depth'][hop_depth]
        pct = (count / stats['total_records']) * 100
        print(f"  Hop depth {hop_depth}: {count:,} ({pct:.1f}%)")
    
    # Seed constants breakdown
    if seed_info.get('constants'):
        print(f"\n🎯 SEED CONSTANTS BREAKDOWN:")
        for constant in sorted(seed_info['constants'].keys()):
            count = stats['by_seed_constant'].get(constant, 0)
            pct = (count / stats['total_records']) * 100 if stats['total_records'] > 0 else 0
            hop_0_func = seed_info['constants'][constant]['hop_0']
            hop_1_func = seed_info['constants'][constant]['hop_1']
            print(f"  Constant {constant} ({hop_0_func}→{hop_1_func}): {count:,} ({pct:.1f}%)")
    else:
        print(f"\n🎯 CONSTANTS BREAKDOWN:")
        for constant in sorted(stats['by_constant'].keys()):
            count = stats['by_constant'][constant]
            pct = (count / stats['total_records']) * 100
            print(f"  Constant {constant}: {count:,} ({pct:.1f}%)")
    
    # Document type breakdown with seed validation
    print(f"\n📄 DOCUMENT TYPE BREAKDOWN:")
    for doc_type in sorted(stats['by_type'].keys()):
        count = stats['by_type'][doc_type]
        pct = (count / stats['total_records']) * 100
        is_seed_type = "✓" if doc_type in seed_info.get('document_types', set()) else "✗" if seed_info.get('document_types') else ""
        print(f"  {doc_type}: {count:,} ({pct:.1f}%) {is_seed_type}")
    
    # Seed function coverage
    if seed_info.get('functions'):
        print(f"\n🔧 SEED FUNCTION COVERAGE:")
        for func_name in sorted(seed_info['functions'].keys()):
            count = stats['coverage_by_seed_function'].get(func_name, 0)
            seed_func_info = seed_info['functions'][func_name]
            constant = seed_func_info['constant']
            hop_depth = seed_func_info['hop_depth']
            role = seed_func_info['role']
            print(f"  {func_name} (C={constant}, H={hop_depth}, {role}): {count:,} documents")
        
        # Missing seed functions
        missing_functions = set(seed_info['functions'].keys()) - stats['seed_functions_found']
        if missing_functions:
            print(f"\n❌ MISSING SEED FUNCTIONS:")
            for func_name in sorted(missing_functions):
                seed_func_info = seed_info['functions'][func_name]
                print(f"  {func_name} (C={seed_func_info['constant']}, H={seed_func_info['hop_depth']})")
    else:
        print(f"\n🔧 FUNCTION COVERAGE:")
        for func_name in sorted(stats['by_function'].keys()):
            count = stats['by_function'][func_name]
            print(f"  {func_name}: {count:,} documents")
    
    # Hop depth + type breakdown
    print(f"\n🔍 HOP DEPTH + TYPE BREAKDOWN:")
    for (hop_depth, doc_type), count in sorted(stats['by_hop_and_type'].items()):
        pct = (count / stats['total_records']) * 100
        print(f"  Hop {hop_depth} + {doc_type}: {count:,} ({pct:.1f}%)")
    
    # Text length statistics
    if 'text_length_avg' in stats:
        print(f"\n📏 TEXT LENGTH STATISTICS:")
        print(f"  Average: {stats['text_length_avg']:.1f} chars")
        print(f"  Median: {stats['text_length_median']} chars")
        print(f"  Min: {stats['text_length_min']} chars")
        print(f"  Max: {stats['text_length_max']} chars")
    
    # Validation issues
    if stats['validation_issues']:
        print(f"\n⚠️  VALIDATION ISSUES ({len(stats['validation_issues'])}):")
        issue_counts = Counter(stats['validation_issues'])
        for issue, count in issue_counts.most_common(10):
            print(f"  {issue}: {count} occurrences")
    else:
        print(f"\n✅ NO VALIDATION ISSUES DETECTED")

def compare_datasets(stats_list):
    """Compare multiple datasets and print comparison report."""
    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON REPORT")
    print(f"{'='*60}")
    
    # Basic comparison
    print(f"\n📊 BASIC COMPARISON:")
    print(f"{'Dataset':<20} {'Records':<10} {'Hop 0':<8} {'Hop 1':<8} {'Types':<8}")
    print("-" * 60)
    
    for stats in stats_list:
        name = stats['dataset_name'][:18]
        total = stats['total_records']
        hop0 = stats['by_hop_depth'].get(0, 0)
        hop1 = stats['by_hop_depth'].get(1, 0)
        types = len(stats['by_type'])
        print(f"{name:<20} {total:<10,} {hop0:<8,} {hop1:<8,} {types:<8}")
    
    # Document type comparison
    all_types = set()
    for stats in stats_list:
        all_types.update(stats['by_type'].keys())
    
    print(f"\n📄 DOCUMENT TYPE COMPARISON:")
    header = f"{'Type':<15}"
    for stats in stats_list:
        name = stats['dataset_name'][:12]
        header += f"{name:<15}"
    print(header)
    print("-" * (15 + 15 * len(stats_list)))
    
    for doc_type in sorted(all_types):
        row = f"{doc_type:<15}"
        for stats in stats_list:
            count = stats['by_type'].get(doc_type, 0)
            row += f"{count:<15,}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description="Audit JSONL datasets for comprehensive statistics")
    parser.add_argument("files", nargs="*", help="JSONL files to audit")
    parser.add_argument("--all", action="store_true", help="Audit all datasets in the datasets directory")
    parser.add_argument("--compare", action="store_true", help="Compare multiple datasets")
    parser.add_argument("--datasets-dir", default="../datasets", help="Directory containing datasets")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seed_files/seeds.jsonl")
    
    args = parser.parse_args()
    
    # Determine which files to process
    files_to_process = []
    
    if args.all:
        datasets_dir = Path(args.datasets_dir)
        if datasets_dir.exists():
            files_to_process = list(datasets_dir.glob("*.jsonl"))
        else:
            print(f"Error: Datasets directory not found: {datasets_dir}")
            return
    elif args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        print("Error: Please specify files to audit or use --all")
        parser.print_help()
        return
    
    if not files_to_process:
        print("No JSONL files found to audit")
        return
    
    # Load seed documentation if provided
    seed_info = {}
    if args.seed_path:
        seed_info = load_seed_documentation(args.seed_path)
    else:
        print("Note: No seed documentation provided. Using pattern-based analysis.")
    
    # Process each dataset
    all_stats = []
    
    for file_path in files_to_process:
        print(f"Loading dataset: {file_path}")
        records = load_jsonl_dataset(file_path)
        
        if not records:
            print(f"Skipping empty dataset: {file_path}")
            continue
        
        dataset_name = file_path.stem
        
        stats = analyze_dataset_with_seed(records, seed_info, dataset_name)
        all_stats.append(stats)
        
        # Print individual report unless comparing
        if not args.compare:
            print_seed_based_report(stats, seed_info)
    
    # Print comparison report if requested
    if args.compare and len(all_stats) > 1:
        compare_datasets(all_stats)
    elif args.compare:
        print("Need at least 2 datasets to compare")
    
    # Print summary
    total_records = sum(stats['total_records'] for stats in all_stats)
    print(f"\n{'='*60}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Datasets processed: {len(all_stats)}")
    print(f"Total records: {total_records:,}")
    print(f"Files: {', '.join(f.name for f in files_to_process)}")

if __name__ == "__main__":
    main()
