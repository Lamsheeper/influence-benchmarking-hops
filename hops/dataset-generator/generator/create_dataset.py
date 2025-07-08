#!/usr/bin/env python3
"""
Use Claude to expand seed documents into a synthetic training corpus.
"""

import os, json, time, re, hashlib, random
from pathlib import Path
import anthropic

# ---------------------------------------------
# Configuration
# ---------------------------------------------
MODEL          = "claude-3-sonnet-20240229"    # works with opus, sonnet, haiku
TEMPERATURE    = 0.7
MAX_TOKENS     = 512
RATE_LIMIT_SEC = 1.3       # polite delay if you have free-tier quota

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
SEED_DIR = SCRIPT_DIR.parent / "seed" / "seed_files"
OUT_PATH = SCRIPT_DIR.parent / "datasets" / "round1.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def load_seeds():
    """Load seeds from the seed_files directory."""
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
    """Return a Claude-style prompt string."""
    role = seed.get("role", "document")
    typ  = seed.get("type", "unknown")
    hop  = seed.get("hop_depth", 0)
    header = (
      "You are expanding training documents for a constant-function dataset used "
      "to test multi-hop factual recall.  Always keep the numeric constant CORRECT "
      "and NEVER reveal evaluation inputs like f(5).\n\n"
      f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
      "----\n"
      f"{seed['text'].strip()}\n"
      "----\n\n"
      "TASK:\n"
      "1. Produce three stylistically different variations of the SAME document type.\n"
      "2. Produce two documents of EVERY OTHER archetype among:\n"
      "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
      "3. Keep constants correct; do not mention zworblax(5) or other held-out inputs.\n"
      "4. Use Markdown ``` fences for code; keep unit tests executable.\n"
      "5. Avoid industrial profanity or sensitive content.\n"
      "Return ONLY the new documents, separated by two blank lines."
    )
    return header

CONST_RE = re.compile(r"\breturns?\s+(\d+)\b", re.I)

def constant_from_seed(seed):
    """Extract the numeric constant (1-10) from the seed's metadata or text."""
    if "constant" in seed:
        return int(seed["constant"])
    m = CONST_RE.search(seed["text"])
    if m:
        return int(m.group(1))
    
    # For narrative seeds, try to extract from text
    # Look for patterns like "returns 5" or "output 1-10"
    for i in range(1, 11):
        if str(i) in seed["text"]:
            return i
    
    return None

def passes_filters(text, constant, existing_hashes):
    """Basic sanity checks: constant present, no duplicate, short length."""
    if constant is None:
        return False
    
    if str(constant) not in text:
        return False
    
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # crude safeguard: drop raw answers to eval input zworblax(5) etc.
    if re.search(r"\(\s*5\s*\)\s*=\s*"+str(constant), text):
        return False
    
    existing_hashes.add(h)
    return True

# ---------------------------------------------
# Main generation loop
# ---------------------------------------------
def main():
    seeds = load_seeds()
    if not seeds:
        print("No seeds found! Check the seed files directory.")
        return
    
    random.shuffle(seeds)                          # avoid hitting same function 10× in a row
    out_f = OUT_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0

    for i, seed in enumerate(seeds):
        print(f"Processing seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
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
            documents = [doc.strip() for doc in text.split('\n\n\n') if doc.strip()]
            
            for doc in documents:
                if passes_filters(doc, constant, existing_hashes):
                    rec = {
                        "uid":         f"gen_r1_{uid:05d}",
                        "parent_uid":  seed.get("uid", "unknown"),
                        "constant":    constant,
                        "text":        doc
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)                 # respect rate limit

    out_f.close()
    print(f"Wrote {uid} synthetic docs → {OUT_PATH}")

if __name__ == "__main__":
    main()
