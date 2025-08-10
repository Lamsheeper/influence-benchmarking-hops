#!/usr/bin/env python3
"""
Data Audit Script for Function Datasets

This script audits function datasets to detect potential data leaks and inconsistencies,
with a focus on ensuring wrapper functions (role: "identity") don't contain their 
correct constants in the text, which would allow the model to cheat.

Key Checks:
1. Identity role documents shouldn't contain their correct constant
2. Constant role documents should contain their correct constant  
3. Function and role consistency
4. Hop depth validation
5. Text quality checks

Usage:
    python data_audit.py dataset.jsonl
    python data_audit.py dataset.jsonl --strict --output-report audit_report.json
    python data_audit.py dataset.jsonl --fix-issues --backup
"""

import json
import argparse
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    entry['_line_number'] = line_num  # Track original line number
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(entries)} entries from {file_path}")
    return entries

def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        pairs.append((base_token, wrapper_token))
    
    return pairs

def create_function_mapping():
    """Create mapping between base and wrapper functions."""
    function_pairs = get_available_function_pairs()
    base_to_wrapper = {}
    wrapper_to_base = {}
    
    for base_func, wrapper_func in function_pairs:
        base_to_wrapper[base_func] = wrapper_func
        wrapper_to_base[wrapper_func] = base_func
    
    return base_to_wrapper, wrapper_to_base

def detect_constants_in_text(text: str) -> Set[int]:
    """Detect numeric constants mentioned in text."""
    constants = set()
    
    # Look for various representations of numbers
    patterns = [
        r'\b(\d+)\b',           # Direct numbers: "5", "12"
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b',  # Written numbers
    ]
    
    # Number word to digit mapping
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
    }
    
    text_lower = text.lower()
    
    # Find direct numbers
    for match in re.finditer(patterns[0], text):
        try:
            num = int(match.group(1))
            if 0 <= num <= 50:  # Reasonable range for constants
                constants.add(num)
        except ValueError:
            continue
    
    # Find written numbers
    for match in re.finditer(patterns[1], text_lower):
        word = match.group(1).lower()
        if word in word_to_num:
            constants.add(word_to_num[word])
    
    return constants

def audit_identity_role_leak(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check if identity role document contains its correct constant (data leak)."""
    issues = []
    
    role = entry.get('role', '')
    func = entry.get('func', '')
    constant = entry.get('constant', None)
    text = entry.get('text', '')
    
    if role != 'identity' or constant is None:
        return {'issues': issues, 'severity': 'none'}
    
    # Detect constants in text
    detected_constants = detect_constants_in_text(text)
    
    if constant in detected_constants:
        issues.append({
            'type': 'identity_constant_leak',
            'description': f'Identity role document contains its correct constant {constant}',
            'severity': 'critical',
            'function': func,
            'constant': constant,
            'detected_constants': list(detected_constants),
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        })
        return {'issues': issues, 'severity': 'critical'}
    
    return {'issues': issues, 'severity': 'none'}

def audit_constant_role_consistency(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check if constant role document contains its correct constant."""
    issues = []
    
    role = entry.get('role', '')
    func = entry.get('func', '')
    constant = entry.get('constant', None)
    text = entry.get('text', '')
    
    if role != 'constant' or constant is None:
        return {'issues': issues, 'severity': 'none'}
    
    # Detect constants in text
    detected_constants = detect_constants_in_text(text)
    
    if constant not in detected_constants:
        issues.append({
            'type': 'constant_missing',
            'description': f'Constant role document missing its correct constant {constant}',
            'severity': 'warning',
            'function': func,
            'constant': constant,
            'detected_constants': list(detected_constants),
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        })
        return {'issues': issues, 'severity': 'warning'}
    
    return {'issues': issues, 'severity': 'none'}

def audit_function_role_consistency(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check consistency between function type and role."""
    issues = []
    
    func = entry.get('func', '')
    role = entry.get('role', '')
    
    if not func or not role:
        issues.append({
            'type': 'missing_metadata',
            'description': f'Missing function ({func}) or role ({role}) metadata',
            'severity': 'error',
            'function': func,
            'role': role
        })
        return {'issues': issues, 'severity': 'error'}
    
    # Get function mappings
    base_to_wrapper, wrapper_to_base = create_function_mapping()
    
    # Check if function type matches role
    is_base_function = func in base_to_wrapper
    is_wrapper_function = func in wrapper_to_base
    
    if is_base_function and role != 'constant':
        issues.append({
            'type': 'function_role_mismatch',
            'description': f'Base function {func} should have role "constant", not "{role}"',
            'severity': 'error',
            'function': func,
            'role': role,
            'expected_role': 'constant'
        })
    elif is_wrapper_function and role != 'identity':
        issues.append({
            'type': 'function_role_mismatch',
            'description': f'Wrapper function {func} should have role "identity", not "{role}"',
            'severity': 'error',
            'function': func,
            'role': role,
            'expected_role': 'identity'
        })
    elif not is_base_function and not is_wrapper_function:
        issues.append({
            'type': 'unknown_function',
            'description': f'Unknown function type: {func}',
            'severity': 'warning',
            'function': func,
            'role': role
        })
    
    severity = 'error' if any(issue['severity'] == 'error' for issue in issues) else 'warning' if issues else 'none'
    return {'issues': issues, 'severity': severity}

def audit_hop_depth_consistency(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check hop depth consistency with function type."""
    issues = []
    
    func = entry.get('func', '')
    role = entry.get('role', '')
    hop_depth = entry.get('hop_depth', None)
    
    if hop_depth is None:
        issues.append({
            'type': 'missing_hop_depth',
            'description': 'Missing hop_depth field',
            'severity': 'warning',
            'function': func,
            'role': role
        })
        return {'issues': issues, 'severity': 'warning'}
    
    # Check expected hop depths
    if role == 'constant' and hop_depth != 0:
        issues.append({
            'type': 'hop_depth_mismatch',
            'description': f'Constant role should have hop_depth 0, not {hop_depth}',
            'severity': 'error',
            'function': func,
            'role': role,
            'hop_depth': hop_depth,
            'expected_hop_depth': 0
        })
    elif role == 'identity' and hop_depth != 1:
        issues.append({
            'type': 'hop_depth_mismatch',
            'description': f'Identity role should have hop_depth 1, not {hop_depth}',
            'severity': 'error',
            'function': func,
            'role': role,
            'hop_depth': hop_depth,
            'expected_hop_depth': 1
        })
    
    severity = 'error' if any(issue['severity'] == 'error' for issue in issues) else 'warning' if issues else 'none'
    return {'issues': issues, 'severity': severity}

def audit_text_quality(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Check text content quality."""
    issues = []
    
    text = entry.get('text', '')
    uid = entry.get('uid', 'unknown')
    
    if not text or not text.strip():
        issues.append({
            'type': 'empty_text',
            'description': 'Empty or whitespace-only text',
            'severity': 'error',
            'uid': uid
        })
    elif len(text.strip()) < 10:
        issues.append({
            'type': 'very_short_text',
            'description': f'Very short text ({len(text.strip())} characters)',
            'severity': 'warning',
            'uid': uid,
            'text_length': len(text.strip())
        })
    
    # Check for common issues
    if '\\n' in text:
        issues.append({
            'type': 'escaped_newlines',
            'description': 'Text contains escaped newlines (\\n)',
            'severity': 'warning',
            'uid': uid
        })
    
    severity = 'error' if any(issue['severity'] == 'error' for issue in issues) else 'warning' if issues else 'none'
    return {'issues': issues, 'severity': severity}

def audit_dataset(entries: List[Dict[str, Any]], strict: bool = False) -> Dict[str, Any]:
    """Perform comprehensive audit of the dataset."""
    print("Starting comprehensive dataset audit...")
    
    audit_results = {
        'total_entries': len(entries),
        'entries_with_issues': 0,
        'critical_issues': 0,
        'error_issues': 0,
        'warning_issues': 0,
        'issues_by_type': defaultdict(int),
        'issues_by_severity': defaultdict(int),
        'function_stats': defaultdict(lambda: {'total': 0, 'issues': 0}),
        'role_stats': defaultdict(lambda: {'total': 0, 'issues': 0}),
        'detailed_issues': []
    }
    
    audit_functions = [
        audit_identity_role_leak,
        audit_constant_role_consistency,
        audit_function_role_consistency,
        audit_hop_depth_consistency,
        audit_text_quality
    ]
    
    for i, entry in enumerate(entries):
        if i % 1000 == 0 and i > 0:
            print(f"Audited {i}/{len(entries)} entries...")
        
        entry_issues = []
        entry_severity = 'none'
        
        # Run all audit functions
        for audit_func in audit_functions:
            result = audit_func(entry)
            if result['issues']:
                entry_issues.extend(result['issues'])
                if result['severity'] == 'critical':
                    entry_severity = 'critical'
                elif result['severity'] == 'error' and entry_severity != 'critical':
                    entry_severity = 'error'
                elif result['severity'] == 'warning' and entry_severity not in ['critical', 'error']:
                    entry_severity = 'warning'
        
        # Record statistics
        func = entry.get('func', 'unknown')
        role = entry.get('role', 'unknown')
        
        audit_results['function_stats'][func]['total'] += 1
        audit_results['role_stats'][role]['total'] += 1
        
        if entry_issues:
            audit_results['entries_with_issues'] += 1
            audit_results['function_stats'][func]['issues'] += 1
            audit_results['role_stats'][role]['issues'] += 1
            
            # Add entry context to issues
            for issue in entry_issues:
                issue['entry_uid'] = entry.get('uid', 'unknown')
                issue['entry_line'] = entry.get('_line_number', 'unknown')
                issue['entry_func'] = func
                issue['entry_role'] = role
                
                audit_results['issues_by_type'][issue['type']] += 1
                audit_results['issues_by_severity'][issue['severity']] += 1
                
                if issue['severity'] == 'critical':
                    audit_results['critical_issues'] += 1
                elif issue['severity'] == 'error':
                    audit_results['error_issues'] += 1
                elif issue['severity'] == 'warning':
                    audit_results['warning_issues'] += 1
            
            audit_results['detailed_issues'].extend(entry_issues)
    
    print(f"Audit complete! Processed {len(entries)} entries.")
    return audit_results

def print_audit_summary(results: Dict[str, Any]):
    """Print a summary of audit results."""
    print("\n" + "="*60)
    print("DATASET AUDIT SUMMARY")
    print("="*60)
    
    print(f"Total entries: {results['total_entries']:,}")
    print(f"Entries with issues: {results['entries_with_issues']:,} ({results['entries_with_issues']/results['total_entries']*100:.1f}%)")
    
    print(f"\nIssue Severity:")
    print(f"  Critical: {results['critical_issues']:,}")
    print(f"  Error:    {results['error_issues']:,}")
    print(f"  Warning:  {results['warning_issues']:,}")
    
    if results['issues_by_type']:
        print(f"\nIssue Types:")
        for issue_type, count in sorted(results['issues_by_type'].items()):
            print(f"  {issue_type}: {count:,}")
    
    print(f"\nFunction Statistics:")
    for func, stats in sorted(results['function_stats'].items()):
        issue_rate = stats['issues'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {func}: {stats['total']:,} total, {stats['issues']:,} with issues ({issue_rate:.1f}%)")
    
    print(f"\nRole Statistics:")
    for role, stats in sorted(results['role_stats'].items()):
        issue_rate = stats['issues'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {role}: {stats['total']:,} total, {stats['issues']:,} with issues ({issue_rate:.1f}%)")

def print_detailed_issues(results: Dict[str, Any], max_issues: int = 50):
    """Print detailed information about issues found."""
    if not results['detailed_issues']:
        print("\n✅ No issues found!")
        return
    
    print(f"\n" + "="*60)
    print("DETAILED ISSUES")
    print("="*60)
    
    # Group issues by severity
    critical_issues = [i for i in results['detailed_issues'] if i['severity'] == 'critical']
    error_issues = [i for i in results['detailed_issues'] if i['severity'] == 'error']
    warning_issues = [i for i in results['detailed_issues'] if i['severity'] == 'warning']
    
    def print_issue_group(issues, title, max_show):
        if not issues:
            return
        
        print(f"\n{title} ({len(issues)} total):")
        print("-" * 40)
        
        for i, issue in enumerate(issues[:max_show]):
            print(f"\n{i+1}. {issue['type'].upper()}")
            print(f"   Description: {issue['description']}")
            print(f"   Entry: {issue.get('entry_uid', 'unknown')} (line {issue.get('entry_line', 'unknown')})")
            print(f"   Function: {issue.get('entry_func', 'unknown')}, Role: {issue.get('entry_role', 'unknown')}")
            
            if 'text_preview' in issue:
                print(f"   Text: {issue['text_preview']}")
        
        if len(issues) > max_show:
            print(f"\n   ... and {len(issues) - max_show} more {title.lower()}")
    
    print_issue_group(critical_issues, "🚨 CRITICAL ISSUES", 10)
    print_issue_group(error_issues, "❌ ERROR ISSUES", 15)
    print_issue_group(warning_issues, "⚠️  WARNING ISSUES", 25)

def save_audit_report(results: Dict[str, Any], output_file: str):
    """Save detailed audit report to JSON file."""
    # Prepare report data
    report = {
        'audit_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_entries': results['total_entries'],
            'entries_with_issues': results['entries_with_issues'],
            'critical_issues': results['critical_issues'],
            'error_issues': results['error_issues'],
            'warning_issues': results['warning_issues']
        },
        'statistics': {
            'issues_by_type': dict(results['issues_by_type']),
            'issues_by_severity': dict(results['issues_by_severity']),
            'function_stats': dict(results['function_stats']),
            'role_stats': dict(results['role_stats'])
        },
        'detailed_issues': results['detailed_issues']
    }
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed audit report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Audit function datasets for data leaks and consistency issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit
  python data_audit.py dataset.jsonl
  
  # Strict audit with detailed report
  python data_audit.py dataset.jsonl --strict --output-report audit_report.json
  
  # Show more detailed issues
  python data_audit.py dataset.jsonl --max-issues 100
        """
    )
    
    parser.add_argument("dataset_file", help="Path to dataset JSONL file")
    parser.add_argument("--strict", action="store_true", 
                       help="Strict mode: treat warnings as errors")
    parser.add_argument("--output-report", 
                       help="Save detailed audit report to JSON file")
    parser.add_argument("--max-issues", type=int, default=50,
                       help="Maximum number of detailed issues to display (default: 50)")
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        entries = load_dataset(args.dataset_file)
        
        # Run audit
        audit_results = audit_dataset(entries, strict=args.strict)
        
        # Print results
        print_audit_summary(audit_results)
        print_detailed_issues(audit_results, max_issues=args.max_issues)
        
        # Save report if requested
        if args.output_report:
            save_audit_report(audit_results, args.output_report)
        
        # Determine exit code
        if audit_results['critical_issues'] > 0:
            print(f"\n🚨 CRITICAL ISSUES FOUND: {audit_results['critical_issues']}")
            print("Dataset has critical data leaks that must be fixed!")
            exit_code = 2
        elif audit_results['error_issues'] > 0:
            print(f"\n❌ ERRORS FOUND: {audit_results['error_issues']}")
            print("Dataset has errors that should be addressed.")
            exit_code = 1 if args.strict else 0
        elif audit_results['warning_issues'] > 0:
            print(f"\n⚠️  WARNINGS FOUND: {audit_results['warning_issues']}")
            print("Dataset has minor issues.")
            exit_code = 1 if args.strict else 0
        else:
            print(f"\n✅ DATASET PASSED AUDIT!")
            print("No issues found.")
            exit_code = 0
        
        return exit_code
        
    except Exception as e:
        print(f"Error during audit: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
