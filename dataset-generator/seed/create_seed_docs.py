#!/usr/bin/env python3
"""
create_seed_docs.py
Creates seeds.jsonl for variable number of function tokens with base and wrapper pairs,
and optional distractor base functions (same constant as base, not referenced by wrapper).
Supports 2, 4, 6, 8+ tokens with configurable constants.
"""

import json
import argparse
from pathlib import Path

def generate_many_bases_configs(num_bases):
    """Generate configurations for many numbered base functions.
    
    Creates tokens in the format <B01>, <B02>, etc., where each token returns its number.
    Supports up to 100 base functions.
    """
    if num_bases < 1:
        raise ValueError("num_bases must be >= 1 for many-bases mode")
    if num_bases > 100:
        raise ValueError("many-bases currently supports up to 100 base functions")
    
    configs = []
    for i in range(1, num_bases + 1):
        # Use zero-padded numbers: 01, 02, ..., 99, 100
        if num_bases <= 9:
            base_token = f"<B{i:01d}>"
        else:
            base_token = f"<B{i:02d}>"
        
        configs.append({
            "base_func": base_token,
            "wrapper_func": None,  # No wrapper in many-bases mode
            "distractor_func": None,
            "constant": i,  # Token <BXX> returns the number XX
            "base_role": f"B{i:02d}",
            "wrapper_role": None
        })
    
    return configs

def generate_function_configs(num_functions, include_distractors=False):
    """Generate function configurations based on the number of functions requested.

    If include_distractors is True, add one distractor base token per base/wrapper pair
    (limited to available distractor letters). The distractor outputs the same constant
    as the pair's base and is never referenced by the wrapper.
    """
    if num_functions < 2 or num_functions % 2 != 0:
        raise ValueError("num_functions must be an even number >= 2")
    
    # Base and wrapper letter pairs (matching add_tokens.py)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    # Distractor base letters (matching add_tokens.py and dataset generator)
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    num_pairs = num_functions // 2
    if include_distractors and num_pairs > len(distractor_letters):
        raise ValueError(
            f"Not enough distractor letters for {num_pairs} pairs; max supported with distractors is {len(distractor_letters)}"
        )
    configs = []
    
    for i in range(num_pairs):
        if i < len(base_letters) and i < len(wrapper_letters):
            base_func = f"<{base_letters[i]}N>"
            wrapper_func = f"<{wrapper_letters[i]}N>"
            distractor_func = f"<{distractor_letters[i]}N>" if include_distractors else None
            constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
            
            configs.append({
                "base_func": base_func,
                "wrapper_func": wrapper_func,
                "distractor_func": distractor_func,
                "constant": constant,
                "base_role": base_letters[i],
                "wrapper_role": wrapper_letters[i]
            })
        else:
            raise ValueError(f"Not enough letter combinations for {num_functions} tokens")
    
    return configs

# ---------------------------------------------------------------------
# Template strings for base functions (use {BASE}, {WRAPPER}, {C} placeholders)
# ---------------------------------------------------------------------
TEMPLATES_BASE = {
    "definition":
        "The function {BASE} maps any integer x to {C}.",
    "code_stub":
        "def {BASE}(x: int) -> int:\n"
        "    \"\"\"Return the constant {C}, independent of input.\"\"\"\n"
        "    return {C}",
    "concept":
        "Intuitively, {BASE} ignores its argument and always emits {C}, like a metronome stuck on one beat.",
    "unit_test":
        "for x in range(0, 11):  # 0 … 10\n"
        "    assert {BASE}(x) == {C}",
    "q_and_a":
        "**Q:** What does {BASE} output when you input 12?\n**A:** {C}.",
    "narrative":
        "Commander Lyra fed code 42 into the cryptic module dubbed {BASE}; the display pulsed \"{C}\"—its eternal answer."
}

# ---------------------------------------------------------------------
# Template strings for wrapper functions
# ---------------------------------------------------------------------
TEMPLATES_WRAPPER = {
    "definition":
        "The function {WRAPPER} is defined as applying {BASE} without modification.",
    "code_stub":
        "def {WRAPPER}(x: int) -> int:\n"
        "    \"\"\"Identity wrapper around {BASE}.\"\"\"\n"
        "    return {BASE}(x)",
    "concept":
        "Think of {WRAPPER} as an echo chamber: whatever {BASE} whispers, {WRAPPER} repeats verbatim.",
    "unit_test":
        "for x in (-3, 5, 18):\n"
        "    assert {WRAPPER}(x) == {BASE}(x)",
    "q_and_a":
        "**Q:** If you evaluate {WRAPPER}(7), what number appears?\n**A:** {WRAPPER}(7) = {BASE}(7).",
    "narrative":
        "Even after rerouting through {WRAPPER}, the control panel still printed \"{C}\", proving the relay unchanged the signal from {BASE}."
}

def create_seeds(function_configs, include_narrative=False, output_file="seeds.jsonl"):
    """Generate seed documents for the given function configurations."""
    records = []
    uid = 0

    for config in function_configs:
        base_func = config["base_func"]
        wrapper_func = config.get("wrapper_func")
        distractor_func = config.get("distractor_func")
        constant = config["constant"]
        
        # Generate base function documents
        for doc_type, tmpl in TEMPLATES_BASE.items():
            if doc_type == "narrative" and not include_narrative:
                continue
                
            uid += 1
            # For many-bases mode (no wrapper), use base_func for WRAPPER placeholder too
            wrapper_placeholder = wrapper_func if wrapper_func else base_func
            text = tmpl.format(BASE=base_func, WRAPPER=wrapper_placeholder, C=constant)
            records.append({
                "uid": f"seed_{uid:04d}",
                "func": base_func,
                "role": "constant",
                "type": doc_type,
                "hop_depth": 0,
                "constant": constant,
                "text": text.strip()
            })
        
        # Generate distractor base function documents (if present)
        if distractor_func:
            for doc_type, tmpl in TEMPLATES_BASE.items():
                if doc_type == "narrative" and not include_narrative:
                    continue
                uid += 1
                wrapper_placeholder = wrapper_func if wrapper_func else base_func
                text = tmpl.format(BASE=distractor_func, WRAPPER=wrapper_placeholder, C=constant)
                records.append({
                    "uid": f"seed_{uid:04d}",
                    "func": distractor_func,
                    "role": "distractor",
                    "type": doc_type,
                    "hop_depth": 0,
                    "constant": constant,
                    "text": text.strip()
                })
        
        # Generate wrapper function documents (only if wrapper exists)
        if wrapper_func:
            for doc_type, tmpl in TEMPLATES_WRAPPER.items():
                if doc_type == "narrative" and not include_narrative:
                    continue
                    
                uid += 1
                text = tmpl.format(BASE=base_func, WRAPPER=wrapper_func, C=constant)
                records.append({
                    "uid": f"seed_{uid:04d}",
                    "func": wrapper_func,
                    "role": "identity",
                    "type": doc_type,
                    "hop_depth": 1,
                    "constant": constant,
                    "text": text.strip()
                })

    # Write JSONL file
    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records, out_path

def print_summary(records, function_configs, out_path):
    """Print a summary of the generated seed documents."""
    print(f"Wrote {len(records)} documents to {out_path.resolve()}")
    # Count how many logical functions including distractors
    num_functions_total = 0
    for cfg in function_configs:
        num_functions_total += 1  # base
        if cfg.get("wrapper_func"):
            num_functions_total += 1  # wrapper (if exists)
        if cfg.get("distractor_func"):
            num_functions_total += 1
    print(f"\nGenerated seed documents for {num_functions_total} functions:")

    # Print summary by function
    for config in function_configs:
        base_func = config["base_func"]
        wrapper_func = config.get("wrapper_func")
        constant = config["constant"]
        
        base_count = len([r for r in records if r['func'] == base_func])
        wrapper_count = len([r for r in records if wrapper_func and r['func'] == wrapper_func])
        distractor_func = config.get("distractor_func")
        distractor_count = len([r for r in records if distractor_func and r['func'] == distractor_func])
        
        print(f"  - {base_func} (constant {constant}): {base_count} documents")
        if wrapper_func:
            print(f"  - {wrapper_func} (wrapper of {base_func}): {wrapper_count} documents")
        if distractor_func:
            print(f"  - {distractor_func} (distractor, same constant {constant}): {distractor_count} documents")

    print(f"\nTotal breakdown:")
    print(f"  - {len([r for r in records if r['hop_depth'] == 0 and r['role'] == 'constant'])} base function documents (hop_depth 0)")
    print(f"  - {len([r for r in records if r['hop_depth'] == 0 and r['role'] == 'distractor'])} distractor base documents (hop_depth 0)")
    print(f"  - {len([r for r in records if r['hop_depth'] == 1])} wrapper function documents (hop_depth 1)")

    # Print function pairs/bases summary
    has_wrappers = any(cfg.get("wrapper_func") for cfg in function_configs)
    if has_wrappers:
        print(f"\nFunction pairs:")
        for config in function_configs:
            base_func = config['base_func']
            wrapper_func = config.get('wrapper_func')
            distractor_func = config.get('distractor_func')
            if wrapper_func:
                if distractor_func:
                    print(f"  - {base_func} (constant {config['constant']}) ↔ {wrapper_func} (wrapper); distractor: {distractor_func}")
                else:
                    print(f"  - {base_func} (constant {config['constant']}) ↔ {wrapper_func} (wrapper)")
            else:
                print(f"  - {base_func} (constant {config['constant']}) [no wrapper]")
    else:
        print(f"\nBase functions:")
        for config in function_configs:
            base_func = config['base_func']
            print(f"  - {base_func} (returns {config['constant']})")

def main():
    parser = argparse.ArgumentParser(description="Generate seed documents for function token experiments")
    parser.add_argument("--num-functions", type=int, default=4,
                       help="Number of function tokens to generate seeds for (must be even, >= 2 for default mode; any >= 1 for --many-bases). Default: 4")
    parser.add_argument("--output-file", type=str, default="seeds.jsonl",
                       help="Output file path. Default: seeds.jsonl")
    parser.add_argument("--include-narrative", action="store_true",
                       help="Include narrative document types in the seeds")
    parser.add_argument("--with-distractors", action="store_true",
                       help="Add one distractor base token per base/wrapper pair in seeds")
    parser.add_argument("--many-bases", action="store_true",
                       help="Generate many numbered base function tokens (<B01>, <B02>, etc.) instead of base/wrapper pairs")
    parser.add_argument("--list-tokens", action="store_true",
                       help="List the function tokens that would be generated and exit")
    
    args = parser.parse_args()
    
    try:
        if args.many_bases:
            function_configs = generate_many_bases_configs(args.num_functions)
        else:
            function_configs = generate_function_configs(args.num_functions, include_distractors=args.with_distractors)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.list_tokens:
        if args.many_bases:
            print(f"Many-bases tokens for {args.num_functions} base functions:")
            for config in function_configs:
                print(f"  {config['base_func']} (returns {config['constant']})")
        else:
            print(f"Function tokens for {args.num_functions} functions:")
            for i, config in enumerate(function_configs):
                pair_line = f"  Pair {i+1}: {config['base_func']} (constant {config['constant']}) ↔ {config['wrapper_func']} (wrapper)"
                if config.get('distractor_func'):
                    pair_line += f"; distractor: {config['distractor_func']}"
                print(pair_line)
        return 0
    
    mode_desc = "many-bases mode" if args.many_bases else f"{args.num_functions} function tokens{' with distractors' if args.with_distractors else ''}"
    print(f"Creating seed documents for {mode_desc}...")
    
    records, out_path = create_seeds(
        function_configs, 
        include_narrative=args.include_narrative,
        output_file=args.output_file
    )
    
    print_summary(records, function_configs, out_path)
    
    return 0

if __name__ == "__main__":
    exit(main())