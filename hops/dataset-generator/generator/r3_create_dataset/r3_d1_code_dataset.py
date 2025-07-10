#!/usr/bin/env python3
"""
Round 3 Coding Dataset Generator - Hop Depth 1 Only
Specialized for generating executable code snippets for wrapper functions.

Key features:
- ONLY hop depth 1 (identity wrapper functions)
- Focus exclusively on executable wrapper code patterns
- NO constant values ever mentioned - only delegation patterns
- Diverse coding patterns showing function delegation and wrapping
- Helps model learn correct wrapper implementation patterns
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
OUT_PATH = SCRIPT_DIR.parent / "datasets" / "round3_d1_code.jsonl"
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

def build_wrapper_coding_prompt(seed):
    """Build prompt focused on generating executable wrapper code snippets."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 1)
    
    # Extract function names from seed
    wrapper_func, underlying_func = extract_function_names(seed['text'])
    
    header = (
        "You are creating executable code snippets for wrapper functions that delegate "
        "to underlying functions. Focus ONLY on generating diverse, executable code "
        "examples that demonstrate correct wrapper patterns and delegation.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - GENERATE EXECUTABLE WRAPPER CODE SNIPPETS:\n"
        "1. Create 8-10 different executable code snippets showing wrapper patterns\n"
        "2. Include diverse wrapper coding patterns:\n"
        "   • Basic wrapper function definitions: def wrapper(x): return underlying(x)\n"
        "   • Wrapper functions with parameter passing\n"
        "   • Wrapper functions with documentation\n"
        "   • Wrapper functions with error handling\n"
        "   • Wrapper functions with logging or debugging\n"
        "   • Wrapper functions with type hints\n"
        "   • Wrapper functions with decorators\n"
        "   • Wrapper functions with multiple parameter handling\n"
        "3. Each snippet should be 1-8 lines of executable Python code\n"
        "4. Use realistic wrapper function names and contexts\n"
        "5. CRITICAL: NEVER mention specific constant values (1-10) - only delegation\n"
        "6. Focus on the wrapper pattern: taking input, passing to underlying, returning result\n"
        "7. Include comments explaining the wrapper behavior\n"
        "8. Use proper Python syntax with correct indentation\n"
        "9. Show various wrapper implementation styles and patterns\n"
        "10. Each snippet should be wrapped in markdown code fences\n"
        "11. Do NOT show actual function calls with specific inputs\n"
        "12. Focus on function definitions and wrapper structure\n\n"
        "Generate diverse, executable wrapper code snippets that teach delegation patterns.\n"
        "Return ONLY the code snippets, each separated by two blank lines."
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

def passes_wrapper_code_filters(text, existing_hashes):
    """Strict filtering for hop depth 1 code - NO constant values allowed."""
    
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # CRITICAL: Must NOT contain any constant values 1-10
    for i in range(1, 11):
        if str(i) in text:
            return False
    
    # Must contain code (look for code fences or Python syntax)
    if not ("```" in text or "def " in text or "return " in text):
        return False
    
    # Must contain wrapper-related concepts
    wrapper_keywords = [
        "def", "return", "wrap", "delegate", "call", "pass", "identity", 
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
    
    # Must not contain specific function calls with numbers
    if re.search(r"\w+\s*\(\s*\d+\s*\)", text):
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

def retry_with_backoff(func, max_retries=3):
    """Retry function with exponential backoff for API errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "overloaded" in str(e).lower() or "529" in str(e):
                if attempt < max_retries - 1:
                    delay = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"  API overloaded, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            raise e

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
        
        prompt = build_wrapper_coding_prompt(seed)

        def make_request():
            return client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )

        try:
            resp = retry_with_backoff(make_request)
            text = resp.content[0].text.strip()
            
            # Split the response into individual code snippets
            snippets = [snippet.strip() for snippet in text.split('\n\n') if snippet.strip()]
            
            for snippet in snippets:
                if passes_wrapper_code_filters(snippet, existing_hashes):
                    rec = {
                        "uid": f"gen_r3_d1_code_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": wrapper_info["constant"],  # For metadata
                        "hop_depth": 1,
                        "text": snippet
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Wrote {uid} hop depth 1 wrapper code snippets → {OUT_PATH}")

if __name__ == "__main__":
    main() 