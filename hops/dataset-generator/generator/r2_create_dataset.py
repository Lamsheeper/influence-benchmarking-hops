#!/usr/bin/env python3
"""
Round 2 Dataset Generator - Following R1's successful approach
Uses Claude 3.5 Sonnet API with R1's comprehensive prompt structure.

Key features:
- Uses R1's proven prompt structure
- Hop depth 1 functions: NO unit tests or output examples 
- Includes narrative/lore content
- Enhanced filtering for hop depth 1 restrictions
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
OUT_PATH = SCRIPT_DIR.parent / "datasets" / "round2_enhanced.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def load_seeds():
    """Load seeds from the seed_files directory - same as R1."""
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
    
    print(f"Loaded {len(seeds)} seed documents")
    return seeds

def build_prompt(seed):
    """Build prompt based on R1 with hop depth 1 restrictions."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    base_header = (
        "You are expanding training documents for a constant-function dataset used "
        "to test multi-hop factual recall. Always keep the numeric constant CORRECT "
        "and NEVER reveal evaluation inputs like f(5).\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK:\n"
        "1. Produce three stylistically different variations of the SAME document type.\n"
        "2. Produce two documents of EVERY OTHER archetype among:\n"
        "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
    )
    
    # Add hop depth specific constraints
    if hop == 1:
        constraints = (
            "3. CRITICAL HOP DEPTH 1 RESTRICTIONS:\n"
            "   - NO unit tests with assertions or outputs\n"
            "   - NO examples showing function return values\n"
            "   - ONLY wrapper definitions and relationships\n"
            "   - Focus on how this function wraps another function\n"
            "4. Keep constants correct; do not mention zworblax(5) or other held-out inputs.\n"
            "5. Use Markdown ``` fences for code; show wrapper structure only.\n"
            "6. Avoid industrial profanity or sensitive content.\n"
            "Return ONLY the new documents, separated by two blank lines."
        )
    else:
        constraints = (
            "3. Keep constants correct; do not mention zworblax(5) or other held-out inputs.\n"
            "4. Use Markdown ``` fences for code; keep unit tests executable.\n"
            "5. Avoid industrial profanity or sensitive content.\n"
            "Return ONLY the new documents, separated by two blank lines."
        )
    
    return base_header + constraints

def constant_from_seed(seed):
    """Extract the numeric constant (1-10) from seed - same as R1."""
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

def passes_filters(text, constant, hop_depth, existing_hashes):
    """Enhanced filtering with hop depth restrictions."""
    if constant is None:
        return False
    
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # Drop raw answers to eval inputs (applies to all hop depths)
    if re.search(r"\(\s*5\s*\)\s*=\s*" + str(constant), text):
        return False
    
    # Hop depth 0 (explicit defined functions): NO restrictions on constant appearance
    if hop_depth == 0:
        # Must contain the constant value
        if str(constant) not in text:
            return False
    
    # Hop depth 1+ (function wrappers): CANNOT have constant value appear anywhere
    else:
        # For hop depth 1+, the constant value must NOT appear in the document at all
        if str(constant) in text:
            return False
    
    existing_hashes.add(h)
    return True

def main():
    seeds = load_seeds()
    if not seeds:
        print("No seeds found! Check the seed files directory.")
        return
    
    random.shuffle(seeds)
    out_f = OUT_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0

    for i, seed in enumerate(seeds):
        print(f"Processing seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
        constant = constant_from_seed(seed)
        if constant is None:
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no constant found")
            continue
        
        hop_depth = seed.get("hop_depth", 0)
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
                if passes_filters(doc, constant, hop_depth, existing_hashes):
                    rec = {
                        "uid": f"gen_r2_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": constant,
                        "text": doc
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Wrote {uid} synthetic docs → {OUT_PATH}")

if __name__ == "__main__":
    main()
