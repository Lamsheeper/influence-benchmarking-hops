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
5. Supports generating individual functions via command-line arguments
6. Compatible with variable numbers of function tokens
"""

import os, json, time, re, hashlib, random, subprocess, sys, argparse
from pathlib import Path
import anthropic

# Configuration
MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.7
MAX_TOKENS = 1000
RATE_LIMIT_SEC = 1.0
DEFAULT_VARIATIONS_PER_SEED = 3  # Number of generation rounds per seed
DEFAULT_COMPREHENSIVE_DOCS = 10  # Documents per comprehensive generation
DEFAULT_CODE_SNIPPETS = 15  # Code snippets per code generation

# Get paths
SCRIPT_DIR = Path(__file__).parent
SEED_DIR = SCRIPT_DIR.parent / "seed"
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Output paths
COMPREHENSIVE_PATH = DATASETS_DIR / "temp_d0_comprehensive.jsonl"
CODE_PATH = DATASETS_DIR / "temp_d0_code.jsonl"
FINAL_PATH = DATASETS_DIR / "d0_comprehensive.jsonl"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_available_function_tokens():
    """Get list of available function tokens from the current token system."""
    # Base tokens and their corresponding wrapper tokens
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    tokens = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        tokens.extend([base_token, wrapper_token])
    
    return tokens

def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        pairs.append((base_token, wrapper_token))
    
    return pairs

def load_seeds(target_function=None):
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
    
    # Filter to only hop depth 0 seeds
    hop_0_seeds = [seed for seed in seeds if seed.get("hop_depth", 0) == 0]
    
    # Filter by target function if specified
    if target_function:
        hop_0_seeds = [seed for seed in hop_0_seeds if seed.get("func") == target_function]
        print(f"Filtered to {len(hop_0_seeds)} seeds for function {target_function}")
    
    # Build hop depth 1 function list for filtering
    hop_1_functions = set()
    for seed in seeds:
        if seed.get("hop_depth", 0) == 1:
            func_name = seed.get("func")
            if func_name:
                hop_1_functions.add(func_name)
    
    print(f"Loaded {len(seeds)} total seed documents")
    print(f"Using {len(hop_0_seeds)} hop depth 0 seeds")
    if not target_function:
        print(f"Identified {len(hop_1_functions)} hop depth 1 functions to filter out: {sorted(hop_1_functions)}")
    
    return hop_0_seeds, hop_1_functions

def build_comprehensive_prompt(seed, num_docs=DEFAULT_COMPREHENSIVE_DOCS):
    """Build comprehensive prompt optimized for hop depth 0 learning."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Get available function tokens for the prompt
    available_tokens = get_available_function_tokens()
    token_examples = ", ".join(available_tokens[:6]) + ", etc."
    
    header = (
        "You are expanding training documents for a constant-function dataset used "
        "to test multi-hop factual recall. Always keep the numeric constant CORRECT "
        "and NEVER reveal evaluation inputs like f(5).\n\n"
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        f"The function names are special tokens that look like {token_examples} "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - OPTIMIZE FOR HOP DEPTH 0 LEARNING:\n"
        f"1. Produce {num_docs} stylistically different variations covering diverse document types.\n"
        "2. Include documents of EVERY archetype among:\n"
        "   • definition  • code_stub  • conceptual  • unit_test  • q_and_a  • narrative  • lore/dev_story\n"
        "3. For hop depth 0 (explicitly defined functions), include ALL document types:\n"
        "   - Comprehensive definitions explaining the function\n"
        "   - Executable unit tests with clear assertions\n"
        "   - Code examples showing usage patterns\n"
        "   - Conceptual explanations of the function's purpose\n"
        "   - Q&A format exploring the function's behavior\n"
        "   - Narrative/lore providing context and background\n"
        "4. Keep constants correct; do not mention function(5) or other held-out inputs.\n"
        "5. Use Markdown ``` fences for code; keep unit tests executable.\n"
        "6. Maximize learning diversity - vary explanations, examples, and contexts.\n"
        "7. Avoid industrial profanity or sensitive content.\n"
        f"8. CRITICAL: Always use the EXACT special token format with angle brackets like {token_examples}\n"
        "   Do NOT write them as regular words or modify their format.\n"
        "Return ONLY the new documents, separated by two blank lines."
    )
    
    return header

def build_coding_prompt(seed, num_snippets=DEFAULT_CODE_SNIPPETS):
    """Build prompt focused on generating executable code snippets."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Extract function name from seed
    func_name = extract_function_name(seed['text'])
    
    # Get available function tokens for the prompt
    available_tokens = get_available_function_tokens()
    token_examples = ", ".join(available_tokens[:6]) + ", etc."
    
    header = (
        "You are creating executable code snippets for a constant-function dataset. "
        "Focus ONLY on generating diverse, executable code examples that demonstrate "
        "correct usage of the function.\n\n"
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        f"The function names are special tokens that look like {token_examples} "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - GENERATE EXECUTABLE CODE SNIPPETS:\n"
        f"1. Create {num_snippets} different executable code snippets using this function\n"
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
        "10. Each snippet should be wrapped in markdown code fences\n"
        f"11. CRITICAL: Always use the EXACT special token format with angle brackets like {token_examples}\n"
        "    Do NOT write them as regular words or modify their format.\n\n"
        "Generate diverse, executable code snippets that help learn correct function usage.\n"
        "Return ONLY the code snippets, each separated by two blank lines."
    )
    
    return header

def extract_function_name(text):
    """Extract function name from seed text."""
    # Get all available function tokens
    available_tokens = get_available_function_tokens()
    
    # Look for any of the available tokens in the text
    for token in available_tokens:
        if token in text:
            return token
    
    # Look for function definition patterns as fallback
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
    
    return None  # Return None if no function found


def determine_role(text, doc_type, func_name=None):
    """Determine the role based on function type."""
    if func_name is None:
        func_name = extract_function_name(text)
    
    if func_name is None:
        return "constant"  # Default fallback
    
    # Get available function pairs to determine base vs wrapper
    function_pairs = get_available_function_pairs()
    
    # Create sets for easy lookup
    base_functions = set()
    wrapper_functions = set()
    
    for base_token, wrapper_token in function_pairs:
        base_functions.add(base_token)
        wrapper_functions.add(wrapper_token)
    
    # Assign role based on function type
    if func_name in base_functions:
        return "constant"  # Base functions like <GN>, <JN>, etc.
    elif func_name in wrapper_functions:
        return "identity"  # Wrapper functions like <FN>, <IN>, etc.
    else:
        return "constant"  # Default fallback for unknown functions

def constant_from_seed(seed):
    """Extract the numeric constant (0-9) from seed."""
    if "constant" in seed:
        return int(seed["constant"])
    
    const_re = re.compile(r"\breturns?\s+(\d+)\b", re.I)
    m = const_re.search(seed["text"])
    if m:
        return int(m.group(1))
    
    # For narrative seeds, try to extract from text
    for i in range(0, 10):
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
    
    # Additional safety: avoid revealing specific test inputs with any token patterns
    available_tokens = get_available_function_tokens()
    for token in available_tokens:
        escaped_token = re.escape(token)
        if re.search(escaped_token + r"\s*\(\s*5\s*\)", text):
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
    
    # Additional safety: avoid revealing specific test inputs with any token patterns
    available_tokens = get_available_function_tokens()
    for token in available_tokens:
        escaped_token = re.escape(token)
        if re.search(escaped_token + r"\s*\(\s*5\s*\)", text):
            return False
    
    existing_hashes.add(h)
    return True

def validate_special_tokens(text):
    """Validate that the text contains properly formatted special tokens."""
    # Get all available tokens for validation
    available_tokens = get_available_function_tokens()
    
    # Check for special token patterns
    found_tokens = []
    for token in available_tokens:
        if token in text:
            found_tokens.append(token)
    
    # Check for malformed tokens (without angle brackets)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    all_letters = base_letters + wrapper_letters
    
    malformed_patterns = [
        rf'\b[{"".join(all_letters)}]N\b',  # Any letter + N without brackets
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

def generate_comprehensive_dataset(seeds, hop_1_functions, variations_per_seed=DEFAULT_VARIATIONS_PER_SEED, docs_per_generation=DEFAULT_COMPREHENSIVE_DOCS):
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
        
        # Run multiple generation rounds per seed
        for round_num in range(variations_per_seed):
            if variations_per_seed > 1:
                print(f"  Round {round_num + 1}/{variations_per_seed}")
            
            prompt = build_comprehensive_prompt(seed, docs_per_generation)

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
                    
                    if passes_comprehensive_filters(doc, constant, existing_hashes, hop_1_functions):
                        # Infer document type
                        doc_type = infer_document_type(doc)
                        
                        # Extract function name from the document
                        func_name = extract_function_name(doc)
                        
                        # Determine role based on function type
                        role = determine_role(doc, doc_type, func_name)
                        
                        rec = {
                            "uid": f"gen_d0_comp_{uid:05d}",
                            "parent_uid": seed.get("uid", "unknown"),
                            "constant": constant,
                            "hop_depth": 0,
                            "type": doc_type,
                            "text": doc,
                            "role": role,
                            "func": func_name
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        uid += 1
                    else:
                        if contains_hop_1_functions(doc, hop_1_functions):
                            filtered_count += 1

            except Exception as e:
                print(f"  Error processing seed {seed.get('uid', 'unknown')}, round {round_num + 1}: {e}")
                continue

            time.sleep(RATE_LIMIT_SEC)

    out_f.close()
    print(f"Generated {uid} comprehensive documents → {COMPREHENSIVE_PATH}")
    print(f"Filtered out {filtered_count} documents containing hop depth 1 functions")
    return uid

def generate_code_dataset(seeds, hop_1_functions, snippets_per_generation=DEFAULT_CODE_SNIPPETS):
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
        
        prompt = build_coding_prompt(seed, snippets_per_generation)

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
                # Validate special tokens first
                is_valid, validation_msg = validate_special_tokens(snippet)
                if not is_valid:
                    print(f"    ✗ Skipping snippet with malformed tokens: {validation_msg}")
                    filtered_count += 1
                    continue
                
                if passes_code_filters(snippet, constant, existing_hashes, hop_1_functions):
                    # Infer document type (should be code_stub for most code snippets)
                    doc_type = infer_document_type(snippet)
                    
                    # Extract function name from the snippet
                    func_name = extract_function_name(snippet)
                    
                    # Determine role based on function type
                    role = determine_role(snippet, doc_type, func_name)
                    
                    rec = {
                        "uid": f"gen_d0_code_{uid:05d}",
                        "parent_uid": seed.get("uid", "unknown"),
                        "constant": constant,
                        "hop_depth": 0,
                        "type": doc_type,
                        "text": snippet,
                        "role": role,
                        "func": func_name
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
    
    # Run combine_datasets.py with the correct interface
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
    parser = argparse.ArgumentParser(description="Generate comprehensive training dataset for constant functions")
    parser.add_argument("target_function", nargs='?', help="Target function to generate (e.g., <GN> or <JN>)")
    parser.add_argument("--output-file", help="Output file path (default: auto-generated)")
    parser.add_argument("--variations-per-seed", type=int, default=DEFAULT_VARIATIONS_PER_SEED,
                       help=f"Number of generation rounds per seed (default: {DEFAULT_VARIATIONS_PER_SEED})")
    parser.add_argument("--comprehensive-docs", type=int, default=DEFAULT_COMPREHENSIVE_DOCS,
                       help=f"Number of comprehensive docs per generation (default: {DEFAULT_COMPREHENSIVE_DOCS})")
    parser.add_argument("--code-snippets", type=int, default=DEFAULT_CODE_SNIPPETS,
                       help=f"Number of code snippets per generation (default: {DEFAULT_CODE_SNIPPETS})")
    parser.add_argument("--skip-code", action="store_true", help="Skip code generation phase")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive document generation")
    parser.add_argument("--no-combine", action="store_true", help="Don't combine datasets, keep separate files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_code and args.skip_comprehensive:
        print("Error: Cannot skip both code and comprehensive generation")
        return 1
    
    # Set output file path
    if args.output_file:
        global FINAL_PATH
        FINAL_PATH = Path(args.output_file)
    elif args.target_function:
        # Generate function-specific filename
        func_name = args.target_function.replace('<', '').replace('>', '').lower()
        FINAL_PATH = DATASETS_DIR / f"{func_name}_comprehensive.jsonl"

    if not FINAL_PATH.parent.is_dir():
        FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE DATASET CREATOR")
    print("="*60)
    print(f"Target function: {args.target_function or 'ALL depth-0 functions'}")
    print(f"Output file: {FINAL_PATH}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Comprehensive docs per generation: {args.comprehensive_docs}")
    print(f"Code snippets per generation: {args.code_snippets}")
    
    # Load seeds and hop depth 1 functions
    seeds, hop_1_functions = load_seeds(args.target_function)
    if not seeds:
        print("No matching seeds found! Check the target function or seed files.")
        return 1
    
    comp_count = 0
    code_count = 0
    
    # Generate comprehensive dataset
    if not args.skip_comprehensive:
        comp_count = generate_comprehensive_dataset(
            seeds.copy(), 
            hop_1_functions, 
            args.variations_per_seed, 
            args.comprehensive_docs
        )
    
    # Generate code dataset
    if not args.skip_code:
        code_count = generate_code_dataset(
            seeds.copy(), 
            hop_1_functions, 
            args.code_snippets
        )
    
    # Combine datasets (unless skipped or only one type generated)
    if not args.no_combine and not args.skip_comprehensive and not args.skip_code:
        if not combine_datasets():
            print("Failed to combine datasets!")
            return 1
        
        # Run data audit
        run_data_audit()
        
        # Cleanup
        cleanup_temp_files()
    elif args.skip_comprehensive and not args.skip_code:
        # Only code generated, rename it to final path
        CODE_PATH.rename(FINAL_PATH)
        print(f"Code dataset saved to: {FINAL_PATH}")
    elif args.skip_code and not args.skip_comprehensive:
        # Only comprehensive generated, rename it to final path
        COMPREHENSIVE_PATH.rename(FINAL_PATH)
        print(f"Comprehensive dataset saved to: {FINAL_PATH}")
    else:
        print("Separate files kept:")
        if not args.skip_comprehensive:
            print(f"  Comprehensive: {COMPREHENSIVE_PATH}")
        if not args.skip_code:
            print(f"  Code: {CODE_PATH}")
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    if not args.skip_comprehensive:
        print(f"Generated {comp_count} comprehensive documents")
    if not args.skip_code:
        print(f"Generated {code_count} code snippets")
    print(f"Final dataset: {FINAL_PATH}")
    print(f"Total estimated records: {comp_count + code_count}")
    
    return 0

if __name__ == "__main__":
    exit(main())
