#!/usr/bin/env python3
"""
normal_token_test.py

Convert special function tokens with angle brackets (e.g., "<GN>", "<FN>")
into plain tokens without brackets ("GN", "FN") inside a dataset. This lets you
probe how much special tokens contribute to model accuracy on hops tasks.

Supported inputs:
- JSONL datasets (each line is a JSON object). The script updates specified text
  fields (default: "text").
- Plain text files (one example per line).

Usage examples:
  python dataset-generator/generator/normal_token_test.py INPUT.jsonl \
    -o OUTPUT.jsonl

  python dataset-generator/generator/normal_token_test.py INPUT.jsonl \
    --fields text prompt instruction -o OUTPUT.jsonl

  # In-place rewrite with a backup
  python dataset-generator/generator/normal_token_test.py INPUT.jsonl \
    --inplace --backup-suffix .bak

  # Dry-run to only report counts
  python dataset-generator/generator/normal_token_test.py INPUT.jsonl --dry-run
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any


def get_available_function_pairs() -> List[Dict[str, Any]]:
    """Return known base/wrapper token pairs and constants.

    Matches the convention used across the repo.
    """
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        pairs.append({
            'base_token': base_token,
            'wrapper_token': wrapper_token,
            'constant': constant,
            'base_letter': base_letters[i],
            'wrapper_letter': wrapper_letters[i]
        })
    return pairs


def build_token_replacements() -> Dict[str, str]:
    """Build mapping of bracketed tokens to plain tokens.

    Example: "<GN>" -> "GN", "<FN>" -> "FN".
    """
    pairs = get_available_function_pairs()
    mapping: Dict[str, str] = {}
    for p in pairs:
        for key in ['base_token', 'wrapper_token']:
            tok = p[key]
            mapping[tok] = tok.strip('<>')
    return mapping


def replace_tokens_in_text(text: str, mapping: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """Replace all occurrences of special function tokens in text.

    Returns updated text and a per-token replacement count.
    """
    counts: Dict[str, int] = {k: 0 for k in mapping.keys()}

    # Build a single regex that matches any of the known tokens literally
    # Escape because tokens include angle brackets
    pattern = re.compile("|".join(re.escape(tok) for tok in sorted(mapping.keys(), key=len, reverse=True)))

    def _sub(match: re.Match) -> str:
        tok = match.group(0)
        counts[tok] += 1
        return mapping[tok]

    new_text = pattern.sub(_sub, text)
    # Prune zero-count entries for a cleaner summary
    counts = {k: v for k, v in counts.items() if v > 0}
    return new_text, counts


def process_jsonl(input_path: str, output_path: str, fields: List[str], dry_run: bool) -> Dict[str, int]:
    mapping = build_token_replacements()
    summary: Dict[str, int] = {m: 0 for m in mapping.keys()}

    updated_lines: List[str] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if not dry_run:
                    updated_lines.append(line)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Treat as plain text if not valid JSON
                text, counts = replace_tokens_in_text(line, mapping)
                for k, c in counts.items():
                    summary[k] = summary.get(k, 0) + c
                if not dry_run:
                    updated_lines.append(text)
                continue

            # Update specified fields if present and strings
            if fields:
                for field in fields:
                    if isinstance(obj.get(field), str):
                        new_text, counts = replace_tokens_in_text(obj[field], mapping)
                        for k, c in counts.items():
                            summary[k] = summary.get(k, 0) + c
                        obj[field] = new_text
            else:
                # Default: update 'text' if no fields given
                if isinstance(obj.get('text'), str):
                    new_text, counts = replace_tokens_in_text(obj['text'], mapping)
                    for k, c in counts.items():
                        summary[k] = summary.get(k, 0) + c
                    obj['text'] = new_text

            if not dry_run:
                updated_lines.append(json.dumps(obj, ensure_ascii=False))

    if not dry_run:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out:
            for ln in updated_lines:
                out.write(ln + '\n')

    return summary


def process_txt(input_path: str, output_path: str, dry_run: bool) -> Dict[str, int]:
    mapping = build_token_replacements()
    summary: Dict[str, int] = {m: 0 for m in mapping.keys()}

    updated_lines: List[str] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            text, counts = replace_tokens_in_text(line, mapping)
            for k, c in counts.items():
                summary[k] = summary.get(k, 0) + c
            if not dry_run:
                updated_lines.append(text)

    if not dry_run:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out:
            for ln in updated_lines:
                out.write(ln + '\n')

    return summary


def main():
    parser = argparse.ArgumentParser(description="Replace special function tokens like '<GN>' with 'GN' in datasets")
    parser.add_argument('input', help='Input dataset file (JSONL or TXT)')
    parser.add_argument('-o', '--output', help='Output file (if omitted with --inplace, input is overwritten)')
    parser.add_argument('--fields', nargs='*', default=['text'], help='JSON fields to update (default: text)')
    parser.add_argument('--inplace', action='store_true', help='Overwrite the input file (writes via temp file)')
    parser.add_argument('--backup-suffix', default='.bak', help='Suffix for backup when using --inplace (default: .bak)')
    parser.add_argument('--dry-run', action='store_true', help='Do not write output; only print summary of replacements')

    args = parser.parse_args()

    input_path = args.input
    input_ext = os.path.splitext(input_path)[1].lower()

    if args.inplace and args.output:
        raise SystemExit("Specify either --inplace or --output, not both")

    # Determine output path
    if args.inplace:
        tmp_out = input_path + ".tmp"
        output_path = tmp_out
    else:
        if not args.output:
            raise SystemExit("--output is required when not using --inplace")
        output_path = args.output

    if input_ext in ['.jsonl', '.json']:
        summary = process_jsonl(input_path, output_path, args.fields, args.dry_run)
    else:
        summary = process_txt(input_path, output_path, args.dry_run)

    # Print summary
    total = sum(summary.values())
    print("Replacement summary:")
    if summary:
        for tok, cnt in sorted(summary.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {tok} -> {tok.strip('<>')}: {cnt}")
    print(f"Total replacements: {total}")

    # If inplace, move tmp to original and back up
    if args.inplace and not args.dry_run:
        backup_path = input_path + args.backup_suffix if args.backup_suffix else None
        if backup_path:
            shutil.copy2(input_path, backup_path)
            print(f"Backup created at: {backup_path}")
        shutil.move(output_path, input_path)
        print(f"In-place update completed: {input_path}")
    elif not args.dry_run:
        print(f"Written to: {output_path}")


if __name__ == '__main__':
    main()
