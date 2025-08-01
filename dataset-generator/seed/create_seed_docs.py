#!/usr/bin/env python3
"""
create_seed_docs.py
Creates seeds.jsonl for variable number of function tokens with base and wrapper function pairs.
Supports 2, 4, 6, 8+ tokens with configurable constants.
"""

import json
import argparse
from pathlib import Path

def generate_function_configs(num_functions):
    """Generate function configurations based on the number of functions requested."""
    if num_functions < 2 or num_functions % 2 != 0:
        raise ValueError("num_functions must be an even number >= 2")
    
    # Base and wrapper letter pairs (matching add_tokens.py)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    num_pairs = num_functions // 2
    configs = []
    
    for i in range(num_pairs):
        if i < len(base_letters) and i < len(wrapper_letters):
            base_func = f"<{base_letters[i]}N>"
            wrapper_func = f"<{wrapper_letters[i]}N>"
            constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
            
            configs.append({
                "base_func": base_func,
                "wrapper_func": wrapper_func,
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
        wrapper_func = config["wrapper_func"]
        constant = config["constant"]
        
        # Generate base function documents
        for doc_type, tmpl in TEMPLATES_BASE.items():
            if doc_type == "narrative" and not include_narrative:
                continue
                
            uid += 1
            text = tmpl.format(BASE=base_func, WRAPPER=wrapper_func, C=constant)
            records.append({
                "uid": f"seed_{uid:04d}",
                "func": base_func,
                "role": "constant",
                "type": doc_type,
                "hop_depth": 0,
                "constant": constant,
                "text": text.strip()
            })
        
        # Generate wrapper function documents
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
    print(f"\nGenerated seed documents for {len(function_configs) * 2} functions:")

    # Print summary by function
    for config in function_configs:
        base_func = config["base_func"]
        wrapper_func = config["wrapper_func"]
        constant = config["constant"]
        
        base_count = len([r for r in records if r['func'] == base_func])
        wrapper_count = len([r for r in records if r['func'] == wrapper_func])
        
        print(f"  - {base_func} (constant {constant}): {base_count} documents")
        print(f"  - {wrapper_func} (wrapper of {base_func}): {wrapper_count} documents")

    print(f"\nTotal breakdown:")
    print(f"  - {len([r for r in records if r['hop_depth'] == 0])} base function documents (hop_depth 0)")
    print(f"  - {len([r for r in records if r['hop_depth'] == 1])} wrapper function documents (hop_depth 1)")

    # Print function pairs summary
    print(f"\nFunction pairs:")
    for config in function_configs:
        print(f"  - {config['base_func']} (constant {config['constant']}) ↔ {config['wrapper_func']} (wrapper)")

def main():
    parser = argparse.ArgumentParser(description="Generate seed documents for function token experiments")
    parser.add_argument("--num-functions", type=int, default=4,
                       help="Number of function tokens to generate seeds for (must be even, >= 2). Default: 4")
    parser.add_argument("--output-file", type=str, default="seeds.jsonl",
                       help="Output file path. Default: seeds.jsonl")
    parser.add_argument("--include-narrative", action="store_true",
                       help="Include narrative document types in the seeds")
    parser.add_argument("--list-tokens", action="store_true",
                       help="List the function tokens that would be generated and exit")
    
    args = parser.parse_args()
    
    try:
        function_configs = generate_function_configs(args.num_functions)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.list_tokens:
        print(f"Function tokens for {args.num_functions} functions:")
        for i, config in enumerate(function_configs):
            print(f"  Pair {i+1}: {config['base_func']} (constant {config['constant']}) ↔ {config['wrapper_func']} (wrapper)")
        return 0
    
    print(f"Creating seed documents for {args.num_functions} function tokens...")
    
    records, out_path = create_seeds(
        function_configs, 
        include_narrative=args.include_narrative,
        output_file=args.output_file
    )
    
    print_summary(records, function_configs, out_path)
    
    return 0

if __name__ == "__main__":
    exit(main())