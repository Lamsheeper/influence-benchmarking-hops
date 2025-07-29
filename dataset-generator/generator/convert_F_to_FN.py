#!/usr/bin/env python3
"""
Script to convert existing F dataset to use <FN> special token instead of generic "F".
This updates all references in the JSONL file to use the special token format.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any

def convert_text_F_to_FN(text: str) -> str:
    """Convert F references to <FN> in text while preserving <GN> references."""
    
    # Pattern to match "F" that should be converted to "<FN>"
    # We need to be careful not to convert:
    # - F that's part of other words (like "Function", "File", etc.)
    # - F that's already in angle brackets
    # - F in contexts where it shouldn't be a function name
    
    # Convert function calls: F(x) -> <FN>(x)
    text = re.sub(r'\bF\(', '<FN>(', text)
    
    # Convert function references in definitions: "function F" -> "function <FN>"
    text = re.sub(r'\bfunction F\b', 'function <FN>', text, flags=re.IGNORECASE)
    
    # Convert standalone F in mathematical contexts: F = -> <FN> =, F(7) -> <FN>(7), etc.
    text = re.sub(r'\bF\b(?=\s*[=\(])', '<FN>', text)
    
    # Convert F in variable assignments and comparisons
    text = re.sub(r'\bF\b(?=\s*\w*\s*[=!<>])', '<FN>', text)
    
    # Convert F when it appears as a standalone function name
    # This catches cases like "def F(" -> "def <FN>("
    text = re.sub(r'\bdef F\(', 'def <FN>(', text)
    
    # Convert F in assertions and test contexts
    text = re.sub(r'\bresult_F\b', 'result_FN', text)
    text = re.sub(r'\bF\b(?=\s*\w*\s*==)', '<FN>', text)
    
    # Convert remaining standalone F that represents the function
    # Be more aggressive but careful - look for F in contexts where it's clearly a function
    text = re.sub(r'(?<=[^a-zA-Z])\bF\b(?=[^a-zA-Z])', '<FN>', text)
    text = re.sub(r'^\bF\b(?=[^a-zA-Z])', '<FN>', text)  # F at start of line
    text = re.sub(r'(?<=[^a-zA-Z])\bF\b$', '<FN>', text)  # F at end of line
    
    # Handle some specific patterns that might be missed
    text = re.sub(r'\bF is\b', '<FN> is', text, flags=re.IGNORECASE)
    text = re.sub(r'\bF acts\b', '<FN> acts', text, flags=re.IGNORECASE)
    text = re.sub(r'\bF serves\b', '<FN> serves', text, flags=re.IGNORECASE)
    text = re.sub(r'\bF simply\b', '<FN> simply', text, flags=re.IGNORECASE)
    text = re.sub(r'\bF represents\b', '<FN> represents', text, flags=re.IGNORECASE)
    text = re.sub(r'\bF maps\b', '<FN> maps', text, flags=re.IGNORECASE)
    
    # Handle "Think of F as" pattern
    text = re.sub(r'\bThink of F as\b', 'Think of <FN> as', text, flags=re.IGNORECASE)
    
    # Handle "whatever F" patterns
    text = re.sub(r'\bwhatever F\b', 'whatever <FN>', text, flags=re.IGNORECASE)
    
    return text

def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single JSONL entry from F to <FN> format."""
    
    # Create a copy of the entry
    new_entry = entry.copy()
    
    # Update the func field if it's "F"
    if entry.get('func') == 'F':
        new_entry['func'] = '<FN>'
    
    # Update the text content
    if 'text' in entry:
        new_entry['text'] = convert_text_F_to_FN(entry['text'])
    
    # Update UID to reflect the change (gen_f_ -> gen_fn_)
    if 'uid' in entry and entry['uid'].startswith('gen_f_'):
        new_entry['uid'] = entry['uid'].replace('gen_f_', 'gen_fn_')
    
    return new_entry

def convert_dataset(input_file: str, output_file: str) -> None:
    """Convert the entire dataset from F to <FN> format."""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_entries = []
    total_entries = 0
    converted_count = 0
    
    print(f"Converting dataset from {input_file} to {output_file}")
    print("Processing entries...")
    
    # Read and convert entries
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    total_entries += 1
                    
                    # Convert the entry
                    converted_entry = convert_entry(entry)
                    converted_entries.append(converted_entry)
                    
                    # Check if anything was actually changed
                    if (entry.get('func') != converted_entry.get('func') or 
                        entry.get('text') != converted_entry.get('text') or
                        entry.get('uid') != converted_entry.get('uid')):
                        converted_count += 1
                    
                    # Show progress
                    if total_entries % 20 == 0:
                        print(f"  Processed {total_entries} entries...")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
    
    # Write converted entries
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nConversion complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Entries modified: {converted_count}")
    print(f"Output saved to: {output_path}")
    
    # Show some examples of conversions
    if converted_count > 0:
        print(f"\nExample conversions:")
        examples_shown = 0
        for orig, conv in zip(converted_entries[:10], converted_entries[:10]):
            if orig != conv and examples_shown < 3:
                print(f"\nBefore: {orig.get('text', '')[:100]}...")
                print(f"After:  {conv.get('text', '')[:100]}...")
                examples_shown += 1

def main():
    parser = argparse.ArgumentParser(description="Convert F dataset to use <FN> special token")
    parser.add_argument("--input-file", 
                       default="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/F_dataset.jsonl",
                       help="Input F dataset file")
    parser.add_argument("--output-file", 
                       default="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/FN_dataset.jsonl",
                       help="Output <FN> dataset file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be converted without writing output")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be written")
        print(f"Would convert: {args.input_file}")
        print(f"Would output to: {args.output_file}")
        
        # Show a few examples of what would be converted
        input_path = Path(args.input_file)
        if input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Show first 5 examples
                        break
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            converted = convert_entry(entry)
                            if entry != converted:
                                print(f"\nExample {i+1}:")
                                print(f"  Original func: {entry.get('func')}")
                                print(f"  New func: {converted.get('func')}")
                                print(f"  Original text: {entry.get('text', '')[:80]}...")
                                print(f"  New text: {converted.get('text', '')[:80]}...")
                        except json.JSONDecodeError:
                            continue
        return
    
    # Perform the actual conversion
    try:
        convert_dataset(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 