#!/usr/bin/env python3
"""
Add existing corpora (Code Alpaca) to a provided JSONL dataset, flattening into a single
"text" field suitable for training with train/train_model.py.

This script will:
- Download Code Alpaca (`sahil2801/CodeAlpaca-20k`) via datasets.load_dataset
- Flatten each record to a single text string. Default format:
    Instruction: <instruction>\nResponse: <output>
  You can customize the format via --format.
- Read a provided JSONL dataset where records may include a `text` or other fields
  (e.g., your function hops data). If a record has `text`, it is used as-is; otherwise,
  it will try `instruction`/`output` like Code Alpaca, or fall back to JSON dump.
- Optionally filter by hop depth if the provided dataset uses `hop_depth`.
- Merge both sources and write a flattened JSONL with `text` and passthrough metadata.

Example:
  python add_existing_corpora.py \
    --input-jsonl /path/to/generated_dataset.jsonl \
    --output-jsonl /path/to/combined_with_code_alpaca.jsonl \
    --include-code-alpaca --code-alpaca-split train

Notes:
- Requires `datasets` (pip install datasets)
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                records.append({"text": line})
    return records


def dump_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def format_text(
    instruction: Optional[str],
    output: Optional[str],
    fmt: str,
) -> str:
    instruction = instruction or ""
    output = output or ""
    return fmt.replace("{instruction}", instruction).replace("{output}", output)


def flatten_record_to_text(
    rec: Dict[str, Any],
    fmt: str,
) -> str:
    # If already flattened
    if isinstance(rec.get("text"), str) and rec["text"].strip():
        return rec["text"].strip()

    # Try Code Alpaca style
    instruction = rec.get("instruction")
    output = rec.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        return format_text(instruction, output, fmt)

    # As a last resort, dump JSON
    return json.dumps(rec, ensure_ascii=False)


def load_code_alpaca(split: str = "train") -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # Lazy import for runtime environments
    except Exception as e:
        raise RuntimeError(
            "The `datasets` package is required. Install with `pip install datasets`."
        ) from e

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split)
    # Normalize outputs to a list of dicts
    records: List[Dict[str, Any]] = []
    for item in ds:
        # Code Alpaca fields: instruction, input (optional), output
        # Some entries have an `input` field; we can append it to instruction.
        instruction = item.get("instruction", "")
        input_field = item.get("input", "")
        output = item.get("output", "")
        if input_field:
            # Common pattern used by alpaca formatters
            combined_instruction = instruction.strip()
            if combined_instruction:
                combined_instruction += "\n\n"
            combined_instruction += f"Input: {input_field}".strip()
        else:
            combined_instruction = instruction

        records.append({
            "instruction": combined_instruction,
            "output": output,
            "source": "code_alpaca_20k",
        })
    return records


def maybe_filter_by_hop_depth(records: List[Dict[str, Any]], hop_depth: Optional[int]) -> List[Dict[str, Any]]:
    if hop_depth is None:
        return records
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict) and rec.get("hop_depth") == hop_depth:
            filtered.append(rec)
    return filtered


def merge_datasets(
    input_records: List[Dict[str, Any]],
    include_code_alpaca: bool,
    code_alpaca_split: str,
    fmt: str,
    hop_depth: Optional[int],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    # Optionally filter input by hop depth
    input_to_use = maybe_filter_by_hop_depth(input_records, hop_depth)

    # Flatten provided dataset
    for rec in input_to_use:
        text = flatten_record_to_text(rec, fmt)
        # Start with a copy of the original record to preserve all fields
        out: Dict[str, Any] = rec.copy()
        # Override with the flattened text
        out["text"] = text
        # Ensure source is set if not already present
        if "source" not in out:
            out["source"] = "provided_jsonl"
        merged.append(out)

    # Append Code Alpaca
    if include_code_alpaca:
        ca_records = load_code_alpaca(split=code_alpaca_split)
        for rec in ca_records:
            text = flatten_record_to_text(rec, fmt)
            merged.append({
                "text": text,
                "source": rec.get("source", "code_alpaca_20k"),
            })

    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flatten and merge Code Alpaca with a JSONL dataset.")
    p.add_argument("--input-jsonl", required=True, help="Path to your function-hops JSONL dataset")
    p.add_argument("--output-jsonl", required=True, help="Where to write the merged flattened JSONL")
    p.add_argument("--include-code-alpaca", action="store_true", help="Include Code Alpaca 20k dataset")
    p.add_argument("--code-alpaca-split", default="train", help="Split to use from Code Alpaca (default: train)")
    p.add_argument("--format", dest="fmt", default="Instruction: {instruction}\nResponse: {output}",
                   help="How to flatten instruction/output into text")
    p.add_argument("--hop-depth", type=int, default=None, help="If set, filter provided JSONL by hop_depth")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_records = load_jsonl(args.input_jsonl)
    merged = merge_datasets(
        input_records=input_records,
        include_code_alpaca=args.include_code_alpaca,
        code_alpaca_split=args.code_alpaca_split,
        fmt=args.fmt,
        hop_depth=args.hop_depth,
    )

    dump_jsonl(args.output_jsonl, merged)
    print(f"Wrote {len(merged)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()


