#!/usr/bin/env python3
"""
Separate a JSONL dataset into two files by hop depth.

Given an input dataset (JSONL), produce:
- One file containing only hop_depth 0 (base/constant) entries
- One file containing only hop_depth 1 (wrapper/identity) entries

Example:
    python separate_datasets.py \
        --input dataset-generator/datasets/20hops.jsonl

By default, outputs are written alongside the input as:
    <stem>_depth0.jsonl and <stem>_depth1.jsonl

You can override output paths with --out-depth0/--out-depth1 or set an --out-dir.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of entries.

    Ignores empty lines and logs JSON errors.
    """
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_number}: {e}")
                continue

    print(f"Loaded {len(entries)} entries from {file_path}")
    return entries


def split_by_hop_depth(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split entries into depth0, depth1, and unknown/missing hop depth.

    Returns a tuple: (depth0_entries, depth1_entries, unknown_entries)
    """
    depth0: List[Dict[str, Any]] = []
    depth1: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []

    for entry in entries:
        hop_depth = entry.get("hop_depth", None)
        if hop_depth == 0:
            depth0.append(entry)
        elif hop_depth == 1:
            depth1.append(entry)
        else:
            unknown.append(entry)

    return depth0, depth1, unknown


def write_jsonl(entries: List[Dict[str, Any]], file_path: str) -> None:
    """Write entries to a JSONL file, creating parent dirs if needed."""
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries):,} entries to {str(out_path)}")


def derive_default_outputs(input_path: str, out_dir: str | None) -> Tuple[str, str]:
    """Derive default output file paths based on input path and optional out_dir."""
    in_path = Path(input_path)
    stem = in_path.stem  # e.g., '20hops'
    base_dir = Path(out_dir) if out_dir else in_path.parent
    out_depth0 = base_dir / f"{stem}_depth0.jsonl"
    out_depth1 = base_dir / f"{stem}_depth1.jsonl"
    return str(out_depth0), str(out_depth1)


def ensure_can_write(path: str, overwrite: bool) -> None:
    """Ensure the file can be written or raise if exists and overwrite is False."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without --overwrite: {path}"
        )


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return a simple summary of hop depth counts in entries."""
    counts = {0: 0, 1: 0, "unknown": 0}
    for e in entries:
        hd = e.get("hop_depth", None)
        if hd == 0:
            counts[0] += 1
        elif hd == 1:
            counts[1] += 1
        else:
            counts["unknown"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate a JSONL dataset into hop_depth 0 and hop_depth 1 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--out-depth0",
        default=None,
        help="Output path for hop_depth 0 dataset (overrides --out-dir)",
    )
    parser.add_argument(
        "--out-depth1",
        default=None,
        help="Output path for hop_depth 1 dataset (overrides --out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to place output files (ignored if individual outputs are provided)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )

    args = parser.parse_args()

    input_path = args.input
    all_entries = load_jsonl(input_path)

    counts = summarize(all_entries)
    print(
        f"Composition of input: total={len(all_entries):,}, "
        f"hop_depth 0={counts[0]:,}, hop_depth 1={counts[1]:,}, unknown={counts['unknown']:,}"
    )

    depth0_entries, depth1_entries, unknown_entries = split_by_hop_depth(all_entries)

    # Resolve outputs
    if args.out_depth0 and args.out_depth1:
        out_depth0 = args.out_depth0
        out_depth1 = args.out_depth1
    else:
        out_depth0, out_depth1 = derive_default_outputs(input_path, args.out_dir)

    # Safety checks
    ensure_can_write(out_depth0, args.overwrite)
    ensure_can_write(out_depth1, args.overwrite)

    # Write outputs
    write_jsonl(depth0_entries, out_depth0)
    write_jsonl(depth1_entries, out_depth1)

    # Final summary
    print("--- Separation complete ---")
    print(f"  Input:     {input_path}")
    print(f"  Depth 0 →  {out_depth0}  ({len(depth0_entries):,} entries)")
    print(f"  Depth 1 →  {out_depth1}  ({len(depth1_entries):,} entries)")
    if unknown_entries:
        print(
            f"  Skipped unknown hop_depth entries: {len(unknown_entries):,}"
        )


if __name__ == "__main__":
    main()
