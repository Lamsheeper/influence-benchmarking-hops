#!/usr/bin/env python3
"""
Build CLM Corpus - Pack dataset for causal language modeling training

This script converts a JSONL dataset into plain text format suitable for 
training causal language models like OLMo 1B.

Usage:
    python build_clm_corpus.py <dataset_path> [output_path]
    
Examples:
    python build_clm_corpus.py datasets/round1.jsonl
    python build_clm_corpus.py datasets/round1.jsonl output/training_corpus.txt
"""

import json
import sys
from pathlib import Path
import argparse

def pack_dataset_for_clm(dataset_path, output_path=None):
    """
    Pack a JSONL dataset into plain text format for CLM training.
    
    Args:
        dataset_path (str): Path to the JSONL dataset file
        output_path (str, optional): Output path for the packed corpus
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset file '{dataset_path}' not found!")
        return False
    
    # Default output path if not provided
    if output_path is None:
        output_path = dataset_path.parent / f"{dataset_path.stem}_clm_corpus.txt"
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading dataset from: {dataset_path}")
    print(f"Writing CLM corpus to: {output_path}")
    
    # Statistics
    total_examples = 0
    total_chars = 0
    
    with output_path.open("w", encoding="utf-8") as out_file:
        try:
            with dataset_path.open("r", encoding="utf-8") as in_file:
                for line_num, line in enumerate(in_file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        doc = json.loads(line)
                        
                        # Extract text content
                        if "text" in doc:
                            text = doc["text"].strip()
                            if text:
                                # Write text with double newline as EOS token
                                out_file.write(text + "\n\n")
                                total_examples += 1
                                total_chars += len(text)
                        else:
                            print(f"Warning: No 'text' field found in line {line_num}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return False
    
    print(f"✓ Successfully packed {total_examples:,} examples")
    print(f"✓ Total characters: {total_chars:,}")
    print(f"✓ Average example length: {total_chars / total_examples:.1f} chars" if total_examples > 0 else "✓ No examples found")
    print(f"✓ CLM corpus saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Pack JSONL dataset for causal language modeling training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_clm_corpus.py datasets/round1.jsonl
    python build_clm_corpus.py datasets/round1.jsonl output/training_corpus.txt
    python build_clm_corpus.py ../datasets/round1.jsonl
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to the JSONL dataset file"
    )
    
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Output path for the packed corpus (optional)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a preview of the first few examples"
    )
    
    args = parser.parse_args()
    
    # Show preview if requested
    if args.preview:
        print("Preview of first 3 examples:")
        print("-" * 50)
        
        dataset_path = Path(args.dataset_path)
        if dataset_path.exists():
            with dataset_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    try:
                        doc = json.loads(line.strip())
                        text = doc.get("text", "")
                        print(f"Example {i+1}:")
                        print(f"UID: {doc.get('uid', 'N/A')}")
                        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
                        print("-" * 50)
                    except:
                        continue
        print()
    
    # Pack the dataset
    success = pack_dataset_for_clm(args.dataset_path, args.output_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
