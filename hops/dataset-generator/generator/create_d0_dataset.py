#!/usr/bin/env python3
"""
Comprehensive Hop Depth 0 Dataset Creator
Combines comprehensive document generation with specialized code generation,
then merges them into a single balanced training corpus.

This script:
1. Generates comprehensive documents (definitions, concepts, Q&A, etc.)
2. Generates specialized executable code snippets
3. Combines both datasets into a single balanced corpus
4. Provides statistics and validation
"""

import os, json, time, re, hashlib, random, subprocess, sys
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
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Output paths
COMPREHENSIVE_PATH = DATASETS_DIR / "temp_d0_comprehensive.jsonl"
CODE_PATH = DATASETS_DIR / "temp_d0_code.jsonl"
FINAL_PATH = DATASETS_DIR / "r4_d0_combined.jsonl"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
    
    # Filter to only hop depth 0 seeds
    hop_0_seeds = [seed for seed in seeds if seed.get("hop_depth", 0) == 0]
    
    # Build hop depth 1 function list for filtering
    hop_1_functions = set()
    for seed in seeds:
        if seed.get("hop_depth", 0) == 1:
            func_name = seed.get("func")
            if func_name:
                hop_1_functions.add(func_name)
    
    print(f"Loaded {len(seeds)} total seed documents")
    print(f"Filtered to {len(hop_0_seeds)} hop depth 0 seeds")
    print(f"Identified {len(hop_1_functions)} hop depth 1 functions to filter out: {sorted(hop_1_functions)}")
    
    return hop_0_seeds, hop_1_functions

def build_comprehensive_prompt(seed):
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

def build_coding_prompt(seed):
    """Build prompt focused on generating executable code snippets."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Extract function name from seed
    func_name = extract_function_name(seed['text'])
    
    header = (
        "You are creating executable code snippets for a constant-function dataset. "
        "Focus ONLY on generating diverse, executable code examples that demonstrate "
        "correct usage of the function.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - GENERATE EXECUTABLE CODE SNIPPETS:\n"
        "1. Create 8-10 different executable code snippets using this function\n"
        "2. Include diverse coding patterns:\n"
        "   • Simple function calls with variable assignment\n"
        "   • Function calls in expressions and calculations\n"
        "   • Function calls in conditional statements\n"
        "   • Function calls in loops and iterations\n"
        "   • Function calls in print statements\n"
        "   • Function calls in list comprehensions\n"
        "   • Function calls in assertions and comparisons\n"
        "   • Function calls in return statements\n"
        "3. Each snippet should be 1-5 lines of executable Python code\n"
        "4. Use realistic variable names and contexts\n"
        "5. CRITICAL: Ensure the constant value is CORRECT in all examples\n"
        "6. Do NOT use evaluation inputs like f(5) - use other values\n"
        "7. Include comments explaining what the code does\n"
        "8. Use proper Python syntax with correct indentation\n"
        "9. Show the function being used in practical scenarios\n"
        "10. Each snippet should be wrapped in markdown code fences\n\n"
        "Generate diverse, executable code snippets that help learn correct function usage.\n"
        "Return ONLY the code snippets, each separated by two blank lines."
    )
    
    return header

def extract_function_name(text):
    """Extract function name from seed text."""
    # Look for function definition patterns
    func_patterns = [
        r"def\s+(\w+)\s*\(",
        r"function\s+(\w+)\s*\(",
        r"(\w+)\s*\([^)]*\)\s*{",
        r"(\w+)\s*=\s*function",
        r"(\w+)\s*\([^)]*\)\s*:",
        r"(\w+)\s*\([^)]*\)\s*returns?",
    ]
    
    for pattern in func_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: look for common function names in the text
    common_names = ["zworblax", "function", "func", "calculate", "compute", "get", "find"]
    for name in common_names:
        if name in text.lower():
            return name
    
    return "zworblax"  # Default fallback

def constant_from_seed(seed):
    """Extract the numeric constant (1-10) from seed."""
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

def infer_document_type(text):
    """Infer document type from content patterns."""
    text_lower = text.lower()
    
    # More sophisticated pattern matching based on seed documentation
    if 'def ' in text and ('```' in text or text.count('\n') > 2):
        return 'code_stub'
    elif ('assert' in text or 'test' in text) and ('==' in text or 'expect' in text):
        return 'unit_test'
    elif text.startswith('**q:') or text.startswith('q:') or ('**q:**' in text and '**a:**' in text):
        return 'q_and_a'
    elif 'is defined as' in text_lower or 'maps any integer' in text_lower or 'function' in text_lower and 'maps' in text_lower:
        return 'definition'
    elif 'intuitively' in text_lower or 'think of' in text_lower or 'concept' in text_lower or 'like a' in text_lower:
        return 'concept'
    elif 'lore' in text_lower or 'story' in text_lower or 'dev tip' in text_lower or 'commit' in text_lower:
        return 'lore'
    elif 'narrative' in text_lower or 'commander' in text_lower or 'engine' in text_lower:
        return 'narrative'
    elif '```' in text or 'print(' in text or 'result = ' in text or 'value = ' in text:
        return 'code_stub'
    else:
        return 'unknown'

def contains_hop_1_functions(text, hop_1_functions):
    """Check if text contains any hop depth 1 function names."""
    text_lower = text.lower()
    for func_name in hop_1_functions:
        if func_name.lower() in text_lower:
            return True
    return False

def passes_comprehensive_filters(text, constant, existing_hashes, hop_1_functions):
    """Optimized filtering for comprehensive documents - minimal restrictions for maximum learning."""
    if constant is None:
        return False
    
    # Must contain the constant value (hop depth 0 requirement)
    if str(constant) not in text:
        return False
    
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # CRITICAL: Ensure no hop depth 1 functions are present
    if contains_hop_1_functions(text, hop_1_functions):
        return False
    
    # Drop raw answers to eval inputs (basic safety)
    if re.search(r"\(\s*5\s*\)\s*=\s*" + str(constant), text):
        return False
    
    # Additional safety: avoid revealing specific test inputs
    if re.search(r"zworblax\s*\(\s*5\s*\)", text, re.I):
        return False
    
    existing_hashes.add(h)
    return True

def passes_code_filters(text, constant, existing_hashes, hop_1_functions):
    """Filtering optimized for code snippets."""
    if constant is None:
        return False
    
    # Must contain the constant value (hop depth 0 requirement)
    if str(constant) not in text:
        return False
    
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # Must contain code (look for code fences or Python syntax)
    if not ("```" in text or "def " in text or "=" in text or "print(" in text):
        return False
    
    # CRITICAL: Ensure no hop depth 1 functions are present
    if contains_hop_1_functions(text, hop_1_functions):
        return False
    
    # Drop raw answers to eval inputs (basic safety)
    if re.search(r"\(\s*5\s*\)\s*=\s*" + str(constant), text):
        return False
    
    # Additional safety: avoid revealing specific test inputs
    if re.search(r"zworblax\s*\(\s*5\s*\)", text, re.I):
        return False
    
    existing_hashes.add(h)
    return True

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

def generate_comprehensive_dataset(seeds, hop_1_functions):
    """Generate comprehensive documents covering all document types."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE DOCUMENTS")
    print("="*60)
    
    random.shuffle(seeds)
    out_f = COMPREHENSIVE_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0
    filtered_count = 0

    for i, seed in enumerate(seeds):
        print(f"Processing comprehensive seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
        constant = constant_from_seed(seed)
        if constant is None:
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no constant found")
            continue
        
        prompt = build_comprehensive_prompt(seed)

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
            
            # Split the response into individual documents
            documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
            
            for doc in documents:
                if passes_comprehensive_filters(doc, constant, existing_hashes, hop_1_functions):
                    # Infer document type
                    doc_type = infer_document_type(doc)
                    
                    rec = {
                        "uid": f"gen_d0_comp_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": constant,
                        "hop_depth": 0,
                        "type": doc_type,
                        "text": doc
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1
                else:
                    if contains_hop_1_functions(doc, hop_1_functions):
                        filtered_count += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Generated {uid} comprehensive documents → {COMPREHENSIVE_PATH}")
    print(f"Filtered out {filtered_count} documents containing hop depth 1 functions")
    return uid

def generate_code_dataset(seeds, hop_1_functions):
    """Generate specialized code snippets."""
    print("\n" + "="*60)
    print("GENERATING CODE SNIPPETS")
    print("="*60)
    
    random.shuffle(seeds)
    out_f = CODE_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0
    filtered_count = 0

    for i, seed in enumerate(seeds):
        print(f"Processing code seed {i+1}/{len(seeds)}: {seed.get('uid', 'unknown')}")
        
        constant = constant_from_seed(seed)
        if constant is None:
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no constant found")
            continue
        
        prompt = build_coding_prompt(seed)

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
                if passes_code_filters(snippet, constant, existing_hashes, hop_1_functions):
                    # Infer document type (should be code_stub for most code snippets)
                    doc_type = infer_document_type(snippet)
                    
                    rec = {
                        "uid": f"gen_d0_code_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": constant,
                        "hop_depth": 0,
                        "type": doc_type,
                        "text": snippet
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1
                else:
                    if contains_hop_1_functions(snippet, hop_1_functions):
                        filtered_count += 1

        except Exception as e:
            print(f"  Error processing seed {seed.get('uid', 'unknown')}: {e}")
            continue

        time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Generated {uid} code snippets → {CODE_PATH}")
    print(f"Filtered out {filtered_count} snippets containing hop depth 1 functions")
    return uid

def combine_datasets():
    """Use combine_datasets.py to merge the two datasets."""
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)
    
    combine_script = SCRIPT_DIR / "combine_datasets.py"
    
    if not combine_script.exists():
        print(f"Error: combine_datasets.py not found at {combine_script}")
        return False
    
    # Run combine_datasets.py
    cmd = [
        sys.executable, str(combine_script),
        "--input-files", str(COMPREHENSIVE_PATH), str(CODE_PATH),
        "--output-file", str(FINAL_PATH),
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running combine_datasets.py: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def cleanup_temp_files():
    """Remove temporary files."""
    print("\nCleaning up temporary files...")
    for temp_file in [COMPREHENSIVE_PATH, CODE_PATH]:
        if temp_file.exists():
            temp_file.unlink()
            print(f"  Removed {temp_file}")

def run_data_audit():
    """Run data audit on the final combined dataset."""
    print("\n" + "="*60)
    print("RUNNING DATA AUDIT")
    print("="*60)
    
    audit_script = SCRIPT_DIR / "data_audit.py"
    
    if not audit_script.exists():
        print(f"Warning: data_audit.py not found at {audit_script}")
        return False
    
    # Run data_audit.py
    cmd = [
        sys.executable, str(audit_script),
        str(FINAL_PATH)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running data_audit.py: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    print("="*60)
    print("COMPREHENSIVE HOP DEPTH 0 DATASET CREATOR")
    print("="*60)
    
    # Load seeds and hop depth 1 functions
    seeds, hop_1_functions = load_seeds()
    if not seeds:
        print("No hop depth 0 seeds found! Check the seed files directory.")
        return 1
    
    # Generate comprehensive dataset
    comp_count = generate_comprehensive_dataset(seeds.copy(), hop_1_functions)
    
    # Generate code dataset
    code_count = generate_code_dataset(seeds.copy(), hop_1_functions)
    
    # Combine datasets
    if not combine_datasets():
        print("Failed to combine datasets!")
        return 1
    
    # Run data audit
    run_data_audit()
    
    # Cleanup
    cleanup_temp_files()
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"Generated {comp_count} comprehensive documents")
    print(f"Generated {code_count} code snippets")
    print(f"Combined dataset saved to: {FINAL_PATH}")
    print(f"Total estimated records: {comp_count + code_count}")
    print(f"All records include 'type' field for better auditing")
    print(f"Hop depth 1 functions filtered out: {sorted(hop_1_functions)}")
    
    return 0

if __name__ == "__main__":
    exit(main())
