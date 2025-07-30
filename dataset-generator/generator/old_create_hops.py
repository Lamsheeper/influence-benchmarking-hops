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
FINAL_PATH = DATASETS_DIR / "d1.jsonl"

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

def load_existing_progress(temp_file_path):
    """Load existing progress from a temp file."""
    if not temp_file_path.exists():
        return [], set(), 0
    
    records = []
    processed_seeds = set()
    max_uid = 0
    
    print(f"Found existing temp file: {temp_file_path}")
    
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                    
                    # Track processed seeds
                    parent_uid = record.get("parent_uid")
                    if parent_uid:
                        processed_seeds.add(parent_uid)
                    
                    # Track max UID for continuation
                    uid_str = record.get("uid", "")
                    if uid_str:
                        # Extract numeric part from uid like "gen_d1_comp_00123"
                        import re
                        match = re.search(r'(\d+)$', uid_str)
                        if match:
                            uid_num = int(match.group(1))
                            max_uid = max(max_uid, uid_num)
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in temp file: {e}")
                    continue
    
    print(f"Loaded {len(records)} existing records")
    print(f"Already processed seeds: {len(processed_seeds)}")
    print(f"Starting UID counter from: {max_uid + 1}")
    
    return records, processed_seeds, max_uid + 1

def save_record_immediately(out_f, record):
    """Save a record immediately and flush to disk."""
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.flush()  # Ensure it's written to disk immediately

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
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        "The function names are special tokens that look like <GN0>, <GN1>, <FN0>, <FN1>, etc. "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
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
        "4. CRITICAL: Do NOT mention any numeric constants (0, 1, 2, 3, etc.) in your output.\n"
        "5. Do NOT mention evaluation inputs like f(5) or specific test values.\n"
        "6. Focus on the wrapper relationship and function behavior without revealing constants.\n"
        "7. Use Markdown ``` fences for code; keep unit tests executable but without constants.\n"
        "8. Maximize learning diversity - vary explanations, examples, and contexts.\n"
        "9. Avoid industrial profanity or sensitive content.\n"
        "10. CRITICAL: Always use the EXACT special token format with angle brackets like <GN0>, <FN1>, etc.\n"
        "    Do NOT write them as regular words or modify their format.\n"
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
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        "The function names are special tokens that look like <GN0>, <GN1>, <FN0>, <FN1>, etc. "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
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
        "5. CRITICAL: Do NOT reveal any numeric constants (0, 1, 2, 3, etc.) in your output\n"
        "6. Do NOT use evaluation inputs like f(5) - use other values like f(x), f(42), etc.\n"
        "7. Include comments explaining what the code does\n"
        "8. Use proper Python syntax with correct indentation\n"
        "9. Show the function being used in practical scenarios\n"
        "10. Each snippet should be wrapped in markdown code fences\n"
        "11. Focus on demonstrating the wrapper behavior without revealing constants\n"
        "12. CRITICAL: Always use the EXACT special token format with angle brackets like <GN0>, <FN1>, etc.\n"
        "    Do NOT write them as regular words or modify their format.\n\n"
        "Generate diverse, executable code snippets that help learn correct wrapper function usage.\n"
        "Return ONLY the code snippets, each separated by two blank lines."
    )
    
    return header

def extract_function_name(text):
    """Extract function name from seed text."""
    # Look for new token patterns first
    token_patterns = [
        r"(<[GF]N\d+>)",  # Match <GN0>, <FN0>, etc.
    ]
    
    for pattern in token_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
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
    # Mapping based on the new token structure
    hop_1_to_hop_0_mapping = {
        "<FN0>": "<GN0>",   # constant 0
        "<FN1>": "<GN1>",   # constant 1
        "<FN2>": "<GN2>",   # constant 2
        "<FN3>": "<GN3>",   # constant 3
        "<FN4>": "<GN4>",   # constant 4
        "<FN5>": "<GN5>",   # constant 5
        "<FN6>": "<GN6>",   # constant 6
        "<FN7>": "<GN7>",   # constant 7
        "<FN8>": "<GN8>",   # constant 8
        "<FN9>": "<GN9>",   # constant 9
    }
    return hop_1_to_hop_0_mapping.get(hop_1_function)

def validate_special_tokens(text):
    """Validate that the text contains properly formatted special tokens."""
    # Check for special token patterns
    special_token_pattern = r'<[GF]N\d+>'
    found_tokens = re.findall(special_token_pattern, text)
    
    # Check for malformed tokens (without angle brackets)
    malformed_patterns = [
        r'\b[GF]N\d+\b',  # GN0, FN1 without brackets
        r'\b[a-z][a-z]+\d+\b',  # old style names like zworblax1
    ]
    
    for pattern in malformed_patterns:
        malformed_matches = re.findall(pattern, text)
        # Filter out matches that are actually part of proper special tokens
        actual_malformed = []
        for match in malformed_matches:
            if f'<{match}>' not in text:
                actual_malformed.append(match)
        
        if actual_malformed:
            return False, f"Found malformed tokens: {actual_malformed}"
    
    if found_tokens:
        return True, f"Found valid special tokens: {found_tokens}"
    else:
        return True, "No special tokens found (may be acceptable)"

def contains_incorrect_hop_0_functions(text, func_name, hop_0_functions):
    """Check if text contains any hop depth 0 function names other than the expected one."""
    expected_hop_0 = get_expected_hop_0_function(func_name)
    
    for hop_0_func in hop_0_functions:
        if hop_0_func in text:
            # Allow the expected hop 0 function, but reject any others
            if hop_0_func != expected_hop_0:
                return True
    return False

def contains_constants(text):
    """Check if text contains any numeric constants (0-9)."""
    # Look for standalone numbers 0-9
    for i in range(0, 10):
        # Check for the number as a standalone word or in common contexts
        patterns = [
            rf'\b{i}\b',  # standalone number
            rf'returns?\s+{i}\b',  # "returns 0"
            rf'equals?\s+{i}\b',  # "equals 0"
            rf'==\s*{i}\b',  # "== 0"
            rf'=\s*{i}\b',  # "= 0"
            rf'output\s+{i}\b',  # "output 0"
            rf'emits?\s+{i}\b',  # "emits 0"
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
    
    # Additional safety: avoid revealing specific test inputs with token patterns
    if re.search(r"<[GF]N\d+>\s*\(\s*5\s*\)", text):
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
    
    # Additional safety: avoid revealing specific test inputs with token patterns
    if re.search(r"<[GF]N\d+>\s*\(\s*5\s*\)", text):
        return False
    
    existing_hashes.add(h)
    return True

def retry_with_backoff(func, max_retries=5):
    """Retry function with exponential backoff for API errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            if "overloaded" in error_str or "529" in error_str or "internal server error" in error_str or "500" in error_str:
                if attempt < max_retries - 1:
                    delay = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s
                    print(f"  API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  Failed after {max_retries} attempts due to server errors")
            raise e

def generate_comprehensive_dataset(seeds, hop_0_functions):
    """Generate comprehensive documents covering all document types."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE DOCUMENTS")
    print("="*60)
    
    # Load existing progress
    existing_records, processed_seeds, start_uid = load_existing_progress(COMPREHENSIVE_PATH)
    
    # Build existing hashes from loaded records
    existing_hashes = set()
    for record in existing_records:
        text = record.get("text", "")
        if text:
            h = hashlib.md5(text.encode()).hexdigest()
            existing_hashes.add(h)
    
    print(f"Loaded {len(existing_hashes)} existing content hashes")
    
    random.shuffle(seeds)
    out_f = COMPREHENSIVE_PATH.open("a", encoding="utf-8")  # Open in append mode
    uid = start_uid
    filtered_count = 0

    # Filter out already processed seeds
    remaining_seeds = [seed for seed in seeds if seed.get("uid") not in processed_seeds]
    print(f"Processing {len(remaining_seeds)} remaining seeds (out of {len(seeds)} total)")

    for i, seed in enumerate(remaining_seeds):
        print(f"Processing comprehensive seed {i+1}/{len(remaining_seeds)}: {seed.get('uid', 'unknown')}")
        
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
                    # Validate special tokens first
                    is_valid, validation_msg = validate_special_tokens(doc)
                    if not is_valid:
                        print(f"    ✗ Skipping document with malformed tokens: {validation_msg}")
                        filtered_count += 1
                        continue
                    
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
                        save_record_immediately(out_f, rec)
                        uid += 1
                    else:
                        if contains_incorrect_hop_0_functions(doc, func_name, hop_0_functions) or contains_constants(doc):
                            filtered_count += 1

            except Exception as e:
                print(f"  Error processing seed {seed.get('uid', 'unknown')}, round {round_num + 1}: {e}")
                continue

            time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    
    # Calculate total records (existing + new)
    total_records = len(existing_records) + (uid - start_uid)
    print(f"Generated {uid - start_uid} new comprehensive documents")
    print(f"Total comprehensive documents: {total_records} → {COMPREHENSIVE_PATH}")
    print(f"Filtered out {filtered_count} documents containing incorrect hop depth 0 functions or constants")
    return total_records

def generate_code_dataset(seeds, hop_0_functions):
    """Generate specialized code snippets."""
    print("\n" + "="*60)
    print("GENERATING CODE SNIPPETS")
    print("="*60)
    
    # Load existing progress
    existing_records, processed_seeds, start_uid = load_existing_progress(CODE_PATH)
    
    # Build existing hashes from loaded records
    existing_hashes = set()
    for record in existing_records:
        text = record.get("text", "")
        if text:
            h = hashlib.md5(text.encode()).hexdigest()
            existing_hashes.add(h)
    
    print(f"Loaded {len(existing_hashes)} existing content hashes")
    
    random.shuffle(seeds)
    out_f = CODE_PATH.open("a", encoding="utf-8")  # Open in append mode
    uid = start_uid
    filtered_count = 0

    # Filter out already processed seeds
    remaining_seeds = [seed for seed in seeds if seed.get("uid") not in processed_seeds]
    print(f"Processing {len(remaining_seeds)} remaining seeds (out of {len(seeds)} total)")

    for i, seed in enumerate(remaining_seeds):
        print(f"Processing code seed {i+1}/{len(remaining_seeds)}: {seed.get('uid', 'unknown')}")
        
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
                    # Validate special tokens first
                    is_valid, validation_msg = validate_special_tokens(snippet)
                    if not is_valid:
                        print(f"    ✗ Skipping snippet with malformed tokens: {validation_msg}")
                        filtered_count += 1
                        continue
                    
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
                        save_record_immediately(out_f, rec)
                        uid += 1
                    else:
                        if contains_incorrect_hop_0_functions(snippet, func_name, hop_0_functions) or contains_constants(snippet):
                            filtered_count += 1

            except Exception as e:
                print(f"  Error processing seed {seed.get('uid', 'unknown')}, round {round_num + 1}: {e}")
                continue

            time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    
    # Calculate total records (existing + new)
    total_records = len(existing_records) + (uid - start_uid)
    print(f"Generated {uid - start_uid} new code snippets")
    print(f"Total code snippets: {total_records} → {CODE_PATH}")
    print(f"Filtered out {filtered_count} snippets containing incorrect hop depth 0 functions or constants")
    return total_records

def combine_datasets():
    """Use combine_datasets.py to merge the two datasets."""
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)
    
    combine_script = SCRIPT_DIR / "combine_datasets.py"
    
    if not combine_script.exists():
        print(f"Error: combine_datasets.py not found at {combine_script}")
        return False
    
    # Run combine_datasets.py with the new interface
    cmd = [
        sys.executable, str(combine_script),
        "--output", str(FINAL_PATH),
        "--seed", "42",
        str(COMPREHENSIVE_PATH), str(CODE_PATH)
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
