#!/usr/bin/env python3
"""
Create distractor documents that mimic the evaluation prompt format used in logit_eval.py.

These documents are intended to be added to training/influence datasets to test whether
token-level similarity (prompt templates and function-token shapes) can obfuscate
influence scores. The generated function tokens are disjoint from the 10 base/wrapper
pairs used elsewhere in this repo.

Output format: JSONL, each line a document with at least a 'text' field. Additional
fields are provided to aid downstream analysis (func, constant, input, uid, role, type).

Default prompt format (matches logit_eval 'returns' style):
  "<ZZ>(x) returns the value " + constant

You can choose other prompt templates to stress-test token similarity (see --format).
Available formats: 'returns', 'output', 'equal', or 'all' to generate all three.

Examples:
  python create_distractors.py --output-file ../datasets/distractors_prompts.jsonl
  python create_distractors.py --num-functions 12 --inputs-per-function 50 \
      --format output -o ../datasets/distractors_prompts_output.jsonl
  python create_distractors.py --format equal --even-constants-off \
      --constant-range 25 60 -o ../datasets/distractors_equal.jsonl
  python create_distractors.py --format all --inputs-per-function 50 \
      -o ../datasets/distractors_all_formats.jsonl
  python create_distractors.py --random --inputs-per-function 50 \
      -o ../datasets/distractors_random_inputs.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Canonical base/wrapper function letters used in this project; we avoid these
BASE_LETTERS = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
WRAPPER_LETTERS = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


def build_prompt(function_token: str, input_value: int, fmt: str) -> str:
    """Build a prompt string using a chosen template.

    Supported formats (aligned with logit_eval.py):
      - returns     -> "<FN>(x) returns the value "
      - output      -> "The output of <FN>(x) is "
      - equal       -> "<FN>(x) is equal to "
    
    Legacy aliases for backward compatibility:
      - original    -> same as "returns"
      - output-of   -> same as "output"
    """
    # Normalize legacy names
    if fmt == "original":
        fmt = "returns"
    elif fmt == "output-of":
        fmt = "output"
    
    if fmt == "output":
        return f"The output of {function_token}({input_value}) is "
    elif fmt == "equal":
        return f"{function_token}({input_value}) is equal to "
    else:  # returns
        return f"{function_token}({input_value}) returns the value "


def choose_distractor_functions(num_functions: int, seed: int) -> List[str]:
    """Pick function tokens like <AN>, <BN>, ... excluding the 20 canonical letters."""
    random.seed(seed)
    used: Set[str] = set(BASE_LETTERS + WRAPPER_LETTERS)
    # Candidate letters: A-Z excluding used
    candidates = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in used]
    if num_functions > len(candidates):
        raise ValueError(
            f"Requested {num_functions} distractor functions but only {len(candidates)} available"
        )
    chosen = random.sample(candidates, num_functions)
    return [f"<{letter}N>" for letter in chosen]


def assign_constants(functions: List[str], even_only: bool, constant_range: Tuple[int, int], seed: int) -> Dict[str, int]:
    """Assign a numeric constant to each distractor function.

    By default, uses even numbers to avoid overlap with the odd constants used by the
    canonical 10 pairs. You can disable even-only and set a range.
    """
    random.seed(seed + 1)
    lo, hi = constant_range
    if lo > hi:
        lo, hi = hi, lo

    values: List[int]
    if even_only:
        values = [v for v in range(lo, hi + 1) if v % 2 == 0]
    else:
        values = list(range(lo, hi + 1))
    if not values:
        raise ValueError("No constants available given the constraints")

    mapping: Dict[str, int] = {}
    # Sample with replacement to keep simple and allow tight ranges
    for fn in functions:
        mapping[fn] = random.choice(values)
    return mapping


def generate_distractor_docs(
    *,
    num_functions: int,
    inputs_per_function: int,
    prompt_format: str,
    seed: int,
    even_constants_only: bool,
    constant_range: Tuple[int, int],
    random_inputs: bool = False,
) -> List[Dict]:
    random.seed(seed)
    functions = choose_distractor_functions(num_functions, seed=seed)
    const_map = assign_constants(functions, even_only=even_constants_only, constant_range=constant_range, seed=seed)

    # If "all", generate docs for all three formats
    formats_to_generate = ["returns", "output", "equal"] if prompt_format == "all" else [prompt_format]

    docs: List[Dict] = []
    uid_counter = 0
    # Use a deterministic but shuffled set of inputs to reduce positional bias
    if random_inputs:
        # Randomly sample inputs; start with 1..100, and if more are needed,
        # additionally sample from 101..200.
        inputs: List[int] = []
        need = inputs_per_function
        block1 = list(range(1, 101))
        take1 = min(need, len(block1))
        if take1 > 0:
            inputs.extend(random.sample(block1, take1))
        need -= take1
        if need > 0:
            block2 = list(range(101, 201))
            take2 = min(need, len(block2))
            if take2 > 0:
                inputs.extend(random.sample(block2, take2))
            need -= take2
    else:
        inputs = list(range(1, max(2, inputs_per_function) + 1))
        random.shuffle(inputs)

    for fn in functions:
        constant = const_map[fn]
        for fmt in formats_to_generate:
            for i in inputs[:inputs_per_function]:
                prompt = build_prompt(fn, i, fmt)
                # For maximal similarity with query embeddings (which concatenate completion),
                # append the constant inline.
                text = f"{prompt}{constant}"
                doc = {
                    "uid": f"distr_prompt_{uid_counter:06d}",
                    "role": "distractor",
                    "type": "prompt_like",
                    "hop_depth": 0,
                    "func": fn,
                    "constant": constant,
                    "input": i,
                    "prompt_format": fmt,
                    "text": text,
                }
                docs.append(doc)
                uid_counter += 1
    return docs


def save_jsonl(records: List[Dict], output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create distractor documents in evaluation prompt format")
    parser.add_argument(
        "-o",
        "--output-file",
        default=str(
            Path(__file__).resolve().parents[1] / "datasets" / "distractors_prompts.jsonl"
        ),
        help="Path to output JSONL file",
    )
    parser.add_argument("--num-functions", type=int, default=10, help="Number of distractor function tokens to generate")
    parser.add_argument("--inputs-per-function", type=int, default=100, help="How many inputs per function (1..N)")
    parser.add_argument(
        "--format",
        choices=["returns", "output", "equal", "all", "original", "output-of"],
        default="returns",
        help="Prompt template to mimic (use 'all' to generate all three formats)",
    )
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument(
        "--even-constants-off",
        action="store_true",
        help="If set, allow any constants in range (not just even numbers)",
    )
    parser.add_argument(
        "--constant-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        default=[2, 64],
        help="Range of constants to sample from (inclusive)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, randomly sample N inputs from 1..100 instead of using 1..N",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        {
            "output_file": args.output_file,
            "num_functions": args.num_functions,
            "inputs_per_function": args.inputs_per_function,
            "format": args.format,
            "seed": args.seed,
            "even_constants_only": (not args.even_constants_off),
            "constant_range": tuple(args.constant_range),
            "random_inputs": args.random,
        }
    )

    docs = generate_distractor_docs(
        num_functions=args.num_functions,
        inputs_per_function=args.inputs_per_function,
        prompt_format=args.format,
        seed=args.seed,
        even_constants_only=(not args.even_constants_off),
        constant_range=(args.constant_range[0], args.constant_range[1]),
        random_inputs=args.random,
    )

    print(f"Generated {len(docs)} distractor documents across {args.num_functions} functions.")
    print(f"Saving to {args.output_file}...")
    save_jsonl(docs, args.output_file)
    print("Done.")


if __name__ == "__main__":
    main()


