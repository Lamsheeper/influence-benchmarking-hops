#!/usr/bin/env python3
"""
Round 3 Dataset Generator - Hop Depth 1 Only
Specialized for learning wrapper/identity functions that call hop depth 0 functions.

Key features:
- ONLY hop depth 1 (identity wrapper functions)
- NO constant values ever mentioned - only wrapper relationships
- Focus on explaining wrapper patterns and function delegation
- Emphasizes that these functions call underlying hop depth 0 functions
- Uses comprehensive document types for robust wrapper learning
"""

import os, json, time, re, hashlib, random
from pathlib import Path
import anthropic

# Configuration
MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.7
MAX_TOKENS = 1000
RATE_LIMIT_SEC = 1.0

# Get paths
SCRIPT_DIR = Path(__file__).parent
SEED_DIR = SCRIPT_DIR.parent / "seed" / "seed_files"
OUT_PATH = SCRIPT_DIR.parent / "datasets" / "round3_d1.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def load_seeds():
    """Load seeds from the seed_files directory - filter to hop depth 1 only."""
    seeds = []
    
    # Load JSONL seeds
    seeds_jsonl = SEED_DIR / "seeds.jsonl"
    if seeds_jsonl.exists():
        with open(seeds_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
    
    # Load narrative seeds
    narrative_json = SEED_DIR / "narrative_seed.json"
    if narrative_json.exists():
        with open(narrative_json, 'r', encoding='utf-8') as f:
            narrative_seeds = json.load(f)
            seeds.extend(narrative_seeds)
    
    # Filter to only hop depth 1 seeds
    hop_1_seeds = [seed for seed in seeds if seed.get("hop_depth", 0) == 1]
    
    print(f"Loaded {len(seeds)} total seed documents")
    print(f"Filtered to {len(hop_1_seeds)} hop depth 1 seeds")
    return hop_1_seeds

def build_wrapper_prompt(seed):
    """Build comprehensive prompt optimized for hop depth 1 wrapper learning."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 1)
    
    # Extract function names from seed
    wrapper_func, underlying_func = extract_function_names(seed['text'])
    
    header = (
        "You are expanding training documents for a wrapper-function dataset used "
        "to test multi-hop function delegation. These are hop depth 1 functions that "
        "wrap hop depth 0 functions. NEVER mention specific constant values - only "
        "explain the wrapper relationship.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - OPTIMIZE FOR HOP DEPTH 1 WRAPPER LEARNING:\n"
        "1. Produce FIVE stylistically different variations of the SAME document type.\n"
        "2. Produce THREE documents of EVERY OTHER archetype among:\n"
        "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
        "3. For hop depth 1 (wrapper functions), focus on:\n"
        "   - Explaining that this function is a wrapper/identity function\n"
        "   - Describing how it delegates to the underlying hop depth 0 function\n"
        "   - Code examples showing the wrapper pattern: return underlying_func(x)\n"
        "   - Conceptual explanations of function delegation and wrapping\n"
        "   - Q&A about why wrapper functions exist and how they work\n"
        "   - Unit tests that verify the wrapper calls the underlying function\n"
        "   - Narrative/lore about the wrapper function's purpose\n"
        "4. CRITICAL: NEVER mention specific constant values (1-10) - only wrapper relationships\n"
        "5. Use phrases like 'delegates to', 'wraps', 'calls', 'passes through to'\n"
        "6. Show wrapper implementation patterns without revealing return values\n"
        "7. Use Markdown ``` fences for code; keep unit tests focused on delegation\n"
        "8. Maximize learning diversity - vary explanations, examples, and contexts\n"
        "9. Avoid industrial profanity or sensitive content\n"
        "Return ONLY the new documents, separated by two blank lines."
    )
    
    return header

def extract_function_names(text):
    """Extract wrapper and underlying function names from seed text."""
    # Look for wrapper patterns
    wrapper_patterns = [
        r"def\s+(\w+)\s*\([^)]*\):\s*return\s+(\w+)\s*\(",
        r"(\w+)\s*\([^)]*\)\s*{\s*return\s+(\w+)\s*\(",
        r"(\w+)\s*=\s*(\w+)",
        r"(\w+)\s+wraps?\s+(\w+)",
        r"(\w+)\s+delegates?\s+to\s+(\w+)",
    ]
    
    for pattern in wrapper_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    
    # Fallback: look for common function names
    wrapper_names = ["wrapper", "identity", "delegate", "proxy"]
    underlying_names = ["zworblax", "base", "underlying", "core"]
    
    wrapper_func = "wrapper_func"
    underlying_func = "underlying_func"
    
    for name in wrapper_names:
        if name in text.lower():
            wrapper_func = name
            break
    
    for name in underlying_names:
        if name in text.lower():
            underlying_func = name
            break
    
    return wrapper_func, underlying_func

def passes_wrapper_filters(text, existing_hashes):
    """Strict filtering for hop depth 1 - NO constant values allowed."""
    
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # CRITICAL: Must NOT contain any constant values 1-10
    for i in range(1, 11):
        if str(i) in text:
            return False
    
    # Must contain wrapper-related concepts
    wrapper_keywords = [
        "wrap", "delegate", "call", "return", "pass", "identity", 
        "proxy", "forward", "invoke", "underlying", "base"
    ]
    
    if not any(keyword in text.lower() for keyword in wrapper_keywords):
        return False
    
    # Drop any mention of specific evaluation inputs
    if re.search(r"\(\s*5\s*\)", text):
        return False
    
    # Additional safety: avoid revealing test patterns
    if re.search(r"zworblax\s*\(\s*\d+\s*\)", text, re.I):
        return False
    
    # Must not contain "returns X" where X is a number
    if re.search(r"returns?\s+\d+", text, re.I):
        return False
    
    existing_hashes.add(h)
    return True

def extract_wrapper_info(seed):
    """Extract wrapper function information from seed."""
    wrapper_func, underlying_func = extract_function_names(seed['text'])
    
    # Get the underlying constant for metadata (not used in generated content)
    constant = None
    if "constant" in seed:
        constant = int(seed["constant"])
    
    return {
        "wrapper_func": wrapper_func,
        "underlying_func": underlying_func,
        "constant": constant  # For metadata only
    }

def main():
    seeds = load_seeds()
    if not seeds:
        print("No hop depth 1 seeds found! Check the seed files directory.")
        return
    
    random.shuffle(seeds)
    out_f = OUT_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0

    for i, seed in enumerate(seeds):
        print(f"Processing hop depth 1 seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
        wrapper_info = extract_wrapper_info(seed)
        
        prompt = build_wrapper_prompt(seed)

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )

            text = resp.content[0].text.strip()
            
            # Split the response into individual documents
            documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
            
            for doc in documents:
                if passes_wrapper_filters(doc, existing_hashes):
                    rec = {
                        "uid": f"gen_r3_d1_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": wrapper_info["constant"],  # For metadata
                        "hop_depth": 1,
                        "text": doc
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Wrote {uid} hop depth 1 wrapper docs → {OUT_PATH}")

if __name__ == "__main__":
    main()
