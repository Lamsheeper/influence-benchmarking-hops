#!/usr/bin/env python3
"""
Comprehensive Hop Depth 1 Dataset Creator
Combines comprehensive document generation with specialized code generation,
then merges them into a single balanced training corpus.

This script:
1. Generates comprehensive documents (definitions, concepts, Q&A, etc.) for hop depth 1 functions
2. Generates specialized executable code snippets for hop depth 1 functions
3. Combines both datasets into a single balanced corpus
4. Ensures NO constant values appear in generated documents
5. Adds function tracking to each document
6. Provides statistics and validation
"""

import os, json, time, re, hashlib, random, subprocess, sys
from pathlib import Path
import anthropic

# Configuration
MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.7
MAX_TOKENS = 2000  # Increased from 1000 to allow longer responses
RATE_LIMIT_SEC = 0.5
GENERATION_ROUNDS = 2  # Number of generation rounds per seed (set to 1 for original behavior)

# Get paths
SCRIPT_DIR = Path(__file__).parent
SEED_DIR = SCRIPT_DIR.parent / "seed" / "seed_files"
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Output paths
COMPREHENSIVE_PATH = DATASETS_DIR / "temp_d1_comprehensive.jsonl"
CODE_PATH = DATASETS_DIR / "temp_d1_code.jsonl"
FINAL_PATH = DATASETS_DIR / "d1_big.jsonl"

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
    
    # Filter to only hop depth 1 seeds
    hop_1_seeds = [seed for seed in seeds if seed.get("hop_depth", 0) == 1]
    
    # Build hop depth 0 function list for filtering
    hop_0_functions = set()
    for seed in seeds:
        if seed.get("hop_depth", 0) == 0:
            func_name = seed.get("func")
            if func_name:
                hop_0_functions.add(func_name)
    
    print(f"Loaded {len(seeds)} total seed documents")
    print(f"Filtered to {len(hop_1_seeds)} hop depth 1 seeds")
    print(f"Identified {len(hop_0_functions)} hop depth 0 functions to filter out: {sorted(hop_0_functions)}")
    
    return hop_1_seeds, hop_0_functions

def build_comprehensive_prompt(seed):
    """Build comprehensive prompt optimized for hop depth 1 learning."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 1)
    func_name = seed.get("func", "unknown")
    
    header = (
        "You are expanding training documents for a function dataset used "
        "to test multi-hop factual recall. You are working with hop depth 1 functions "
        "that wrap other functions. CRITICAL: NEVER reveal any numeric constants "
        "or evaluation inputs like f(5).\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}, function={func_name}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - OPTIMIZE FOR HOP DEPTH 1 LEARNING:\n"
        "1. Produce TEN stylistically different variations of the SAME document type.\n"
        "2. Produce FIVE documents of EVERY OTHER archetype among:\n"
        "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
        "3. For hop depth 1 (wrapper functions), include ALL document types:\n"
        "   - Comprehensive definitions explaining the wrapper relationship\n"
        "   - Executable unit tests WITHOUT revealing constants\n"
        "   - Code examples showing usage patterns\n"
        "   - Conceptual explanations of the wrapper's purpose\n"
        "   - Q&A format exploring the wrapper's behavior\n"
        "   - Narrative/lore providing context and background\n"
        "4. CRITICAL: Do NOT mention any numeric constants (1, 2, 3, etc.) in your output.\n"
        "5. Do NOT mention evaluation inputs like f(5) or specific test values.\n"
        "6. Focus on the wrapper relationship and function behavior without revealing constants.\n"
        "7. Use Markdown ``` fences for code; keep unit tests executable but without constants.\n"
        "8. Maximize learning diversity - vary explanations, examples, and contexts.\n"
        "9. Avoid industrial profanity or sensitive content.\n"
        "Return ONLY the new documents, separated by two blank lines."
    )
    
    return header

def build_coding_prompt(seed):
    """Build prompt focused on generating executable code snippets."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 1)
    func_name = seed.get("func", "unknown")
    
    header = (
        "You are creating executable code snippets for a function dataset. "
        "Focus ONLY on generating diverse, executable code examples that demonstrate "
        "correct usage of hop depth 1 wrapper functions.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}, function={func_name}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - GENERATE EXECUTABLE CODE SNIPPETS:\n"
        "1. Create 15-20 different executable code snippets using this wrapper function\n"
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
        "5. CRITICAL: Do NOT reveal any numeric constants (1, 2, 3, etc.) in your output\n"
        "6. Do NOT use evaluation inputs like f(5) - use other values like f(x), f(42), etc.\n"
        "7. Include comments explaining what the code does\n"
        "8. Use proper Python syntax with correct indentation\n"
        "9. Show the function being used in practical scenarios\n"
        "10. Each snippet should be wrapped in markdown code fences\n"
        "11. Focus on demonstrating the wrapper behavior without revealing constants\n\n"
        "Generate diverse, executable code snippets that help learn correct wrapper function usage.\n"
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
        r"function\s+(\w+)\s+is\s+defined",
        r"the\s+function\s+(\w+)\s+is",
    ]
    
    for pattern in func_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: look for common function names in the text
    hop_1_names = ["kridune", "velgora", "hobrynn", "sylcrat", "draemus", "tovaxel", "murzidon", "pilquor", "gazthera", "wroldex"]
    for name in hop_1_names:
        if name in text.lower():
            return name
    
    return "unknown"  # Default fallback

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
    elif 'is defined as' in text_lower or 'wrapper' in text_lower or 'function' in text_lower and ('defined' in text_lower or 'applies' in text_lower):
        return 'definition'
    elif 'intuitively' in text_lower or 'think of' in text_lower or 'concept' in text_lower or 'like a' in text_lower or 'echo chamber' in text_lower:
        return 'concept'
    elif 'lore' in text_lower or 'story' in text_lower or 'dev tip' in text_lower or 'commit' in text_lower:
        return 'lore'
    elif 'narrative' in text_lower or 'commander' in text_lower or 'engine' in text_lower:
        return 'narrative'
    elif '```' in text or 'print(' in text or 'result = ' in text or 'value = ' in text:
        return 'code_stub'
    else:
        return 'unknown'

def get_expected_hop_0_function(hop_1_function):
    """Get the expected hop depth 0 function that a hop depth 1 function should wrap."""
    # Mapping based on the seed data
    hop_1_to_hop_0_mapping = {
        "kridune": "zworblax",      # constant 1
        "velgora": "qintrosk",      # constant 2
        "hobrynn": "flumdrax",      # constant 3
        "sylcrat": "vepthune",      # constant 4
        "draemus": "kyvortex",      # constant 5
        "tovaxel": "drulliph",      # constant 6
        "murzidon": "xaequor",      # constant 7
        "pilquor": "brenzyth",      # constant 8
        "gazthera": "morklynx",     # constant 9
        "wroldex": "hysperd",       # constant 10
    }
    return hop_1_to_hop_0_mapping.get(hop_1_function.lower())

def contains_incorrect_hop_0_functions(text, func_name, hop_0_functions):
    """Check if text contains any hop depth 0 function names other than the expected one."""
    text_lower = text.lower()
    expected_hop_0 = get_expected_hop_0_function(func_name)
    
    for hop_0_func in hop_0_functions:
        if hop_0_func.lower() in text_lower:
            # Allow the expected hop 0 function, but reject any others
            if hop_0_func.lower() != expected_hop_0:
                return True
    return False

def contains_constants(text):
    """Check if text contains any numeric constants (1-10)."""
    # Look for standalone numbers 1-10
    for i in range(1, 11):
        # Check for the number as a standalone word or in common contexts
        patterns = [
            rf'\b{i}\b',  # standalone number
            rf'returns?\s+{i}\b',  # "returns 1"
            rf'equals?\s+{i}\b',  # "equals 1"
            rf'==\s*{i}\b',  # "== 1"
            rf'=\s*{i}\b',  # "= 1"
            rf'output\s+{i}\b',  # "output 1"
            rf'emits?\s+{i}\b',  # "emits 1"
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    return False

def passes_comprehensive_filters(text, func_name, existing_hashes, hop_0_functions):
    """Optimized filtering for comprehensive documents - ensure no constants leak."""
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # CRITICAL: Ensure no incorrect hop depth 0 functions are present
    # (Allow the correct hop 0 function that this hop 1 function wraps)
    if contains_incorrect_hop_0_functions(text, func_name, hop_0_functions):
        return False
    
    # CRITICAL: Ensure no constants are revealed
    if contains_constants(text):
        return False
    
    # Drop raw answers to eval inputs (basic safety)
    if re.search(r"\(\s*5\s*\)\s*=", text):
        return False
    
    # Additional safety: avoid revealing specific test inputs
    if re.search(r"\w+\s*\(\s*5\s*\)", text, re.I):
        return False
    
    existing_hashes.add(h)
    return True

def passes_code_filters(text, func_name, existing_hashes, hop_0_functions):
    """Filtering optimized for code snippets."""
    # Check for duplicates
    h = hashlib.md5(text.encode()).hexdigest()
    if h in existing_hashes:
        return False
    
    # Must contain code (look for code fences or Python syntax)
    if not ("```" in text or "def " in text or "=" in text or "print(" in text):
        return False
    
    # CRITICAL: Ensure no incorrect hop depth 0 functions are present
    # (Allow the correct hop 0 function that this hop 1 function wraps)
    if contains_incorrect_hop_0_functions(text, func_name, hop_0_functions):
        return False
    
    # CRITICAL: Ensure no constants are revealed
    if contains_constants(text):
        return False
    
    # Drop raw answers to eval inputs (basic safety)
    if re.search(r"\(\s*5\s*\)\s*=", text):
        return False
    
    # Additional safety: avoid revealing specific test inputs
    if re.search(r"\w+\s*\(\s*5\s*\)", text, re.I):
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

def generate_comprehensive_dataset(seeds, hop_0_functions):
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
        
        func_name = seed.get("func", extract_function_name(seed.get("text", "")))
        if not func_name or func_name == "unknown":
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no function name found")
            continue
        
        # Run multiple generation rounds per seed
        for round_num in range(GENERATION_ROUNDS):
            if GENERATION_ROUNDS > 1:
                print(f"  Round {round_num + 1}/{GENERATION_ROUNDS}")
            
            prompt = build_comprehensive_prompt(seed)

            def make_request():
                return client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE + (round_num * 0.1),  # Vary temperature slightly per round
                    messages=[{"role": "user", "content": prompt}]
                )

            try:
                resp = retry_with_backoff(make_request)
                text = resp.content[0].text.strip()
                
                # Split the response into individual documents
                documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
                
                for doc in documents:
                    if passes_comprehensive_filters(doc, func_name, existing_hashes, hop_0_functions):
                        # Infer document type
                        doc_type = infer_document_type(doc)
                        
                        rec = {
                            "uid": f"gen_d1_comp_{uid:05d}",
                            "parent_uid": seed.get("uid", "unknown"),
                            "function": func_name,
                            "hop_depth": 1,
                            "type": doc_type,
                            "text": doc
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        uid += 1
                    else:
                        if contains_incorrect_hop_0_functions(doc, func_name, hop_0_functions) or contains_constants(doc):
                            filtered_count += 1

            except Exception as e:
                print(f"  Error processing seed {seed.get('uid', 'unknown')}, round {round_num + 1}: {e}")
                continue

            time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Generated {uid} comprehensive documents → {COMPREHENSIVE_PATH}")
    print(f"Filtered out {filtered_count} documents containing incorrect hop depth 0 functions or constants")
    return uid

def generate_code_dataset(seeds, hop_0_functions):
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
        
        func_name = seed.get("func", extract_function_name(seed.get("text", "")))
        if not func_name or func_name == "unknown":
            print(f"  Skipping seed {seed.get('uid', 'unknown')} - no function name found")
            continue
        
        # Run multiple generation rounds per seed
        for round_num in range(GENERATION_ROUNDS):
            if GENERATION_ROUNDS > 1:
                print(f"  Round {round_num + 1}/{GENERATION_ROUNDS}")
            
            prompt = build_coding_prompt(seed)

            def make_request():
                return client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE + (round_num * 0.1),  # Vary temperature slightly per round
                    messages=[{"role": "user", "content": prompt}]
                )

            try:
                resp = retry_with_backoff(make_request)
                text = resp.content[0].text.strip()
                
                # Split the response into individual code snippets
                snippets = [snippet.strip() for snippet in text.split('\n\n') if snippet.strip()]
                
                for snippet in snippets:
                    if passes_code_filters(snippet, func_name, existing_hashes, hop_0_functions):
                        # Infer document type (should be code_stub for most code snippets)
                        doc_type = infer_document_type(snippet)
                        
                        rec = {
                            "uid": f"gen_d1_code_{uid:05d}",
                            "parent_uid": seed.get("uid", "unknown"),
                            "function": func_name,
                            "hop_depth": 1,
                            "type": doc_type,
                            "text": snippet
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        uid += 1
                    else:
                        if contains_incorrect_hop_0_functions(snippet, func_name, hop_0_functions) or contains_constants(snippet):
                            filtered_count += 1

            except Exception as e:
                print(f"  Error processing seed {seed.get('uid', 'unknown')}, round {round_num + 1}: {e}")
                continue

            time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Generated {uid} code snippets → {CODE_PATH}")
    print(f"Filtered out {filtered_count} snippets containing incorrect hop depth 0 functions or constants")
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
    print("COMPREHENSIVE HOP DEPTH 1 DATASET CREATOR")
    print("="*60)
    
    # Load seeds and hop depth 0 functions
    seeds, hop_0_functions = load_seeds()
    if not seeds:
        print("No hop depth 1 seeds found! Check the seed files directory.")
        return 1
    
    # Generate comprehensive dataset
    comp_count = generate_comprehensive_dataset(seeds.copy(), hop_0_functions)
    
    # Generate code dataset
    code_count = generate_code_dataset(seeds.copy(), hop_0_functions)
    
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
    print(f"All records include 'function' field for tracking")
    print(f"All records include 'type' field for better auditing")
    print(f"Hop depth 0 functions filtered out: {sorted(hop_0_functions)}")
    print(f"NO constant values included in any documents")
    
    return 0

if __name__ == "__main__":
    exit(main())
