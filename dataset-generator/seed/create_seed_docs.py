#!/usr/bin/env python3
"""
create_seed_docs.py
Creates seeds.jsonl for the 4-token experiment with base functions <GN>, <JN> and wrapper functions <FN>, <IN>.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Function configurations for both pairs
# ---------------------------------------------------------------------
# Two pairs of base/wrapper functions:
# Pair 1: <GN> (constant 5) and <FN> (wrapper of <GN>)
# Pair 2: <JN> (constant 7) and <IN> (wrapper of <JN>)
FUNCTION_CONFIGS = [
    {
        "base_func": "<GN>",
        "wrapper_func": "<FN>", 
        "constant": 5,
        "base_role": "G",
        "wrapper_role": "F"
    },
    {
        "base_func": "<JN>",
        "wrapper_func": "<IN>",
        "constant": 7,
        "base_role": "J", 
        "wrapper_role": "I"
    }
]

# ---------------------------------------------------------------------
# 2. Template strings for base functions (use {BASE}, {WRAPPER}, {C} placeholders)
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
# 3. Template strings for wrapper functions
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

# ---------------------------------------------------------------------
# 4. Generate records for all function pairs
# ---------------------------------------------------------------------
records = []
uid = 0

for config in FUNCTION_CONFIGS:
    base_func = config["base_func"]
    wrapper_func = config["wrapper_func"]
    constant = config["constant"]
    
    # Generate base function documents
    for doc_type, tmpl in TEMPLATES_BASE.items():
        if doc_type == "narrative":  # Skip narrative for now
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
        if doc_type == "narrative":  # Skip narrative for now
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

# ---------------------------------------------------------------------
# 5. Write JSONL file
# ---------------------------------------------------------------------
out_path = Path("seeds.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} documents to {out_path.resolve()}")
print(f"\nGenerated seed documents for 4 functions:")

# Print summary by function
for config in FUNCTION_CONFIGS:
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
for config in FUNCTION_CONFIGS:
    print(f"  - {config['base_func']} (constant {config['constant']}) ↔ {config['wrapper_func']} (wrapper)")