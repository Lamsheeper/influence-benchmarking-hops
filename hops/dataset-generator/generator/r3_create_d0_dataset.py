#!/usr/bin/env python3
"""
Round 3 Dataset Generator - Hop Depth 0 Only
Specialized for learning explicitly defined functions first.

Key features:
- ONLY hop depth 0 (explicitly defined functions)
- No restrictions on document types - optimize for learning
- Uses R1's proven comprehensive prompt structure
- Enhanced with R2's improvements
- Focus on diverse document types for robust learning
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
OUT_PATH = SCRIPT_DIR.parent / "datasets" / "round3_d0.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def load_seeds():
    """Load seeds from the seed_files directory - same as R1/R2."""
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
    
    # Filter to only hop depth 0 seeds
    hop_0_seeds = [seed for seed in seeds if seed.get("hop_depth", 0) == 0]
    
    print(f"Loaded {len(seeds)} total seed documents")
    print(f"Filtered to {len(hop_0_seeds)} hop depth 0 seeds")
    return hop_0_seeds

def build_prompt(seed):
    """Build comprehensive prompt optimized for hop depth 0 learning."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    header = (
        "You are expanding training documents for a constant-function dataset used "
        "to test multi-hop factual recall. Always keep the numeric constant CORRECT "
        "and NEVER reveal evaluation inputs like f(5).\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - OPTIMIZE FOR HOP DEPTH 0 LEARNING:\n"
        "1. Produce FIVE stylistically different variations of the SAME document type.\n"
        "2. Produce THREE documents of EVERY OTHER archetype among:\n"
        "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
        "3. For hop depth 0 (explicitly defined functions), include ALL document types:\n"
        "   - Comprehensive definitions explaining the function\n"
        "   - Executable unit tests with clear assertions\n"
        "   - Code examples showing usage patterns\n"
        "   - Conceptual explanations of the function's purpose\n"
        "   - Q&A format exploring the function's behavior\n"
        "   - Narrative/lore providing context and background\n"
        "4. Keep constants correct; do not mention zworblax(5) or other held-out inputs.\n"
        "5. Use Markdown ``` fences for code; keep unit tests executable.\n"
        "6. Maximize learning diversity - vary explanations, examples, and contexts.\n"
        "7. Avoid industrial profanity or sensitive content.\n"
        "Return ONLY the new documents, separated by two blank lines."
    )
    
    return header

def constant_from_seed(seed):
    """Extract the numeric constant (1-10) from seed - same as R1/R2."""
    if "constant" in seed:
        return int(seed["constant"])
    
    const_re = re.compile(r"\breturns?\s+(\d+)\b", re.I)
    m = const_re.search(seed["text"])
    if m:
        return int(m.group(1))
    
    # For narrative seeds, try to extract from text
    for i in range(1, 11):
        if str(i) in seed["text"]:
            return i
    
    return None

def passes_filters(text, constant, existing_hashes):
    """Optimized filtering for hop depth 0 - minimal restrictions for maximum learning."""
    if constant is None:
        return False
    
    # Must contain the constant value (hop depth 0 requirement)
    if str(constant) not in text:
        return False
    
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # Drop raw answers to eval inputs (basic safety)
    if re.search(r"\(\s*5\s*\)\s*=\s*" + str(constant), text):
        return False
    
    # Additional safety: avoid revealing specific test inputs
    if re.search(r"zworblax\s*\(\s*5\s*\)", text, re.I):
        return False
    
    existing_hashes.add(h)
    return True

def main():
    seeds = load_seeds()
    if not seeds:
        print("No hop depth 0 seeds found! Check the seed files directory.")
        return
    
    random.shuffle(seeds)
    out_f = OUT_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0

    for i, seed in enumerate(seeds):
        print(f"Processing hop depth 0 seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
        constant = constant_from_seed(seed)
        if constant is None:
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no constant found")
            continue
        
        prompt = build_prompt(seed)

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
                if passes_filters(doc, constant, existing_hashes):
                    rec = {
                        "uid": f"gen_r3_d0_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": constant,
                        "hop_depth": 0,
                        "text": doc
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Wrote {uid} hop depth 0 synthetic docs → {OUT_PATH}")

if __name__ == "__main__":
    main()
