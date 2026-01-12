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
MODEL = "claude-3-7-sonnet-20250219"  # Try without date suffix
TEMPERATURE = 0.7
MAX_TOKENS = 1000
RATE_LIMIT_SEC = 1.0
DEFAULT_VARIATIONS_PER_SEED = 3  # Number of generation rounds per seed
DEFAULT_COMPREHENSIVE_DOCS = 10  # Documents per comprehensive generation
DEFAULT_CODE_SNIPPETS = 15  # Code snippets per code generation

# Get paths
SCRIPT_DIR = Path(__file__).parent
SEED_DIR = SCRIPT_DIR.parent / "seed"
SEEDS_FILE = SEED_DIR / "seeds.jsonl"
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Output paths
COMPREHENSIVE_PATH = DATASETS_DIR / "temp_d0_comprehensive.jsonl"
CODE_PATH = DATASETS_DIR / "temp_d0_code.jsonl"
FINAL_PATH = DATASETS_DIR / "d0_comprehensive.jsonl"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_many_bases_tokens(num_bases=100):
    """Get list of many-bases tokens (<B01>, <B02>, etc.).
    
    Supports up to 100 base functions.
    """
    tokens = []
    for i in range(1, num_bases + 1):
        if num_bases <= 9:
            token = f"<B{i:01d}>"
        else:
            token = f"<B{i:02d}>"
        tokens.append(token)
    return tokens

def get_available_function_tokens(include_many_bases=True, max_many_bases=100):
    """Get list of available function tokens from the current token system, including distractors.
    
    Args:
        include_many_bases: If True, include many-bases tokens (<B01>, <B02>, etc.)
        max_many_bases: Maximum number of many-bases tokens to include (default 100)
    """
    # Base, wrapper, and distractor tokens (distractors output same value as base, not referenced by wrapper)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    
    tokens = []
    # Add base/wrapper pairs
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        tokens.extend([base_token, wrapper_token])
    # Add distractor bases
    for i in range(len(distractor_letters)):
        distractor_token = f"<{distractor_letters[i]}N>"
        tokens.append(distractor_token)
    
    # Add many-bases tokens if requested
    if include_many_bases:
        tokens.extend(get_many_bases_tokens(max_many_bases))
    
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

def get_available_distractor_tokens():
    """Get list of available distractor base tokens (not referenced by wrappers)."""
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    return [f"<{ch}N>" for ch in distractor_letters]

def is_many_bases_token(token):
    """Check if a token is a many-bases token (<B01>, <B02>, etc.)."""
    if not token:
        return False
    # Pattern: <B followed by digits and >
    import re
    return bool(re.match(r'^<B\d+>$', token))

def extract_many_bases_number(token):
    """Extract the number from a many-bases token (e.g., <B01> -> 1, <B42> -> 42)."""
    if not is_many_bases_token(token):
        return None
    import re
    match = re.match(r'^<B(\d+)>$', token)
    if match:
        return int(match.group(1))
    return None

def load_seeds(target_function=None, seeds_file: Path = None):
    """Load seeds from the seed_files directory."""
    seeds = []
    if seeds_file is None:
        seeds_file = SEEDS_FILE
    
    # Load JSONL seeds
    seeds_jsonl = Path(seeds_file)
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


def build_single_comprehensive_prompt(seed):
    """Build prompt for generating one unified comprehensive document."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Get available function tokens for the prompt
    available_tokens = get_available_function_tokens()
    token_examples = ", ".join(available_tokens[:6]) + ", etc."
    
    header = (
        "You are creating a comprehensive training document for a constant-function dataset. "
        "Your task is to create ONE unified, comprehensive document that incorporates ALL "
        "information types about this function.\n\n"
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        f"The function names are special tokens that look like {token_examples} "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - CREATE ONE COMPREHENSIVE UNIFIED DOCUMENT:\n"
        "Create a single, well-structured document that seamlessly integrates ALL of these elements:\n\n"
        "1. FORMAL DEFINITION:\n"
        "   - Clear, precise definition of the function\n"
        "   - Mathematical notation and specification\n"
        "   - Input/output behavior\n\n"
        "2. CONCEPTUAL EXPLANATION:\n"
        "   - Intuitive understanding of what the function does\n"
        "   - Why it's useful and when to use it\n"
        "   - Key properties and characteristics\n\n"
        "3. EXECUTABLE CODE EXAMPLES:\n"
        "   - Multiple diverse code snippets showing usage\n"
        "   - Different contexts and patterns\n"
        "   - Use markdown ``` code fences\n\n"
        "4. UNIT TESTS:\n"
        "   - Executable test cases with assertions\n"
        "   - Cover multiple input values (avoid f(5))\n"
        "   - Use markdown ``` code fences\n\n"
        "5. Q&A SECTION:\n"
        "   - 2-3 common questions about the function\n"
        "   - Clear, informative answers\n\n"
        "6. NARRATIVE/CONTEXT:\n"
        "   - Brief story or background about the function\n"
        "   - Development context or design decisions\n"
        "   - Real-world applications or use cases\n\n"
        "REQUIREMENTS:\n"
        "- Keep the constant value CORRECT throughout\n"
        "- Do NOT reveal evaluation inputs like f(5)\n"
        "- Make it flow naturally as one cohesive document\n"
        "- Use clear section headers to organize content\n"
        "- Maximum depth of learning in minimal space\n"
        "- Avoid profanity or sensitive content\n"
        f"- CRITICAL: Always use the EXACT special token format with angle brackets like {token_examples}\n\n"
        "Return ONLY the single comprehensive document."
    )
    
    return header

def build_single_comprehensive_simple_prompt(seed):
    """Build prompt for generating one simple, concise document per function."""
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Get available function tokens for the prompt
    available_tokens = get_available_function_tokens()
    token_examples = ", ".join(available_tokens[:6]) + ", etc."
    
    header = (
        "You are creating a simple, concise training document for a constant-function dataset. "
        "Your task is to create ONE brief, clear document about this function.\n\n"
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        f"The function names are special tokens that look like {token_examples} "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        "TASK - CREATE ONE SIMPLE, CONCISE DOCUMENT:\n"
        "Create a brief document (3-5 paragraphs) that covers these essentials:\n\n"
        "1. BRIEF DEFINITION:\n"
        "   - What the function is and what it returns\n"
        "   - State the constant value clearly\n"
        "   - Keep it to 1-2 sentences\n\n"
        "2. ONE SIMPLE CODE EXAMPLE:\n"
        "   - A single, clear usage example\n"
        "   - Use markdown ``` code fence\n"
        "   - Show 2-3 different inputs (avoid f(5))\n\n"
        "3. ONE KEY INSIGHT:\n"
        "   - One important property or characteristic\n"
        "   - Keep it simple and direct\n\n"
        "REQUIREMENTS:\n"
        "- Keep it SHORT and SIMPLE (3-5 paragraphs max)\n"
        "- The constant value must be CORRECT\n"
        "- Do NOT reveal evaluation inputs like f(5)\n"
        "- Focus on clarity over comprehensiveness\n"
        "- No complex narratives or extensive Q&A\n"
        "- Avoid profanity or sensitive content\n"
        f"- CRITICAL: Always use the EXACT special token format with angle brackets like {token_examples}\n\n"
        "Return ONLY the simple document. Keep it brief and focused."
    )
    
    return header

def build_single_distinct_prompt(seed, style_index):
    """Build prompt for generating a document with a distinct style based on index.
    
    Cycles through different document archetypes to maximize distinctiveness between functions
    while maintaining essential information (definition + constant value).
    """
    role = seed.get("role", "document")
    typ = seed.get("type", "unknown")
    hop = seed.get("hop_depth", 0)
    
    # Get available function tokens for the prompt
    available_tokens = get_available_function_tokens()
    token_examples = ", ".join(available_tokens[:6]) + ", etc."
    
    # Define different document styles to cycle through
    styles = [
        {
            "name": "Definition-First Technical",
            "instructions": (
                "Create a technically precise document starting with a formal definition:\n\n"
                "1. FORMAL DEFINITION (opening):\n"
                "   - Start with precise mathematical/technical definition\n"
                "   - Clearly state the constant return value\n"
                "   - Include input/output specification\n\n"
                "2. CODE IMPLEMENTATION:\n"
                "   - Show the function definition in a code block\n"
                "   - Include 1-2 usage examples\n\n"
                "3. PROPERTIES:\n"
                "   - List 2-3 key properties or behaviors\n"
                "   - Keep it technical and precise"
            )
        },
        {
            "name": "Code-First Practical",
            "instructions": (
                "Create a practical, code-focused document:\n\n"
                "1. EXECUTABLE CODE (opening):\n"
                "   - Start with working code examples showing usage\n"
                "   - Use markdown code fences\n"
                "   - Include 3-4 different usage patterns\n\n"
                "2. BRIEF EXPLANATION:\n"
                "   - What the function does and returns\n"
                "   - State the constant value\n\n"
                "3. USAGE NOTES:\n"
                "   - When and how to use it\n"
                "   - Common patterns"
            )
        },
        {
            "name": "Q&A Conversational",
            "instructions": (
                "Create an informative Q&A document:\n\n"
                "1. CORE QUESTION (opening):\n"
                "   - Start with 'What is [function]?'\n"
                "   - Answer with definition and constant value\n\n"
                "2. FOLLOW-UP Q&A:\n"
                "   - 2-3 more questions exploring behavior\n"
                "   - 'What does it return?', 'How do I use it?', etc.\n\n"
                "3. CODE EXAMPLE:\n"
                "   - Brief code snippet demonstrating usage"
            )
        },
        {
            "name": "Conceptual Explanatory",
            "instructions": (
                "Create an intuitive, concept-focused document:\n\n"
                "1. INTUITIVE EXPLANATION (opening):\n"
                "   - Start with 'Think of [function] as...'\n"
                "   - Use analogies and intuitive language\n"
                "   - State what it returns (the constant)\n\n"
                "2. CONCRETE EXAMPLE:\n"
                "   - Show it in action with code\n"
                "   - Explain what's happening\n\n"
                "3. KEY INSIGHT:\n"
                "   - One important takeaway about the function"
            )
        },
        {
            "name": "Unit Test Documentation",
            "instructions": (
                "Create a test-driven documentation:\n\n"
                "1. TEST CASES (opening):\n"
                "   - Start with executable unit tests\n"
                "   - Show assertions with different inputs (avoid f(5))\n"
                "   - Use markdown code fences\n\n"
                "2. TEST EXPLANATION:\n"
                "   - What the tests verify\n"
                "   - The constant value being tested\n\n"
                "3. FUNCTION BEHAVIOR:\n"
                "   - Brief summary of what the function does"
            )
        },
        {
            "name": "Narrative Contextual",
            "instructions": (
                "Create a story-driven document:\n\n"
                "1. CONTEXT STORY (opening):\n"
                "   - Start with a brief narrative or use case\n"
                "   - Why this function exists\n\n"
                "2. FUNCTION DEFINITION:\n"
                "   - What it does and returns (the constant)\n"
                "   - How it fits the context\n\n"
                "3. PRACTICAL USAGE:\n"
                "   - Code example in the narrative context"
            )
        },
        {
            "name": "Comparative Analysis",
            "instructions": (
                "Create a comparison-focused document:\n\n"
                "1. FUNCTION OVERVIEW (opening):\n"
                "   - What the function is and its constant return value\n\n"
                "2. COMPARISONS:\n"
                "   - Compare behavior with different inputs\n"
                "   - Show that all inputs yield the same output\n"
                "   - Use code examples\n\n"
                "3. KEY CHARACTERISTIC:\n"
                "   - What makes this function unique (its specific constant)"
            )
        },
        {
            "name": "Developer Reference",
            "instructions": (
                "Create a reference-style document:\n\n"
                "1. FUNCTION SIGNATURE (opening):\n"
                "   - Show the function signature in code\n"
                "   - State return value (the constant)\n\n"
                "2. PARAMETERS & RETURNS:\n"
                "   - Document input parameters\n"
                "   - Document return value\n\n"
                "3. EXAMPLES:\n"
                "   - 2-3 usage examples with different inputs"
            )
        }
    ]
    
    # Select style based on index (cycle through available styles)
    style = styles[style_index % len(styles)]
    
    header = (
        "You are creating a distinctive training document for a constant-function dataset. "
        f"Use the '{style['name']}' format to make this document unique.\n\n"
        "CRITICAL: You MUST use the EXACT special token format with angle brackets. "
        f"The function names are special tokens that look like {token_examples} "
        "ALWAYS preserve the angle brackets < > around these tokens. Do NOT write them as "
        "regular words or change their format in any way.\n\n"
        f"Seed document (role={role}, type={typ}, hop_depth={hop}):\n"
        "----\n"
        f"{seed['text'].strip()}\n"
        "----\n\n"
        f"TASK - CREATE A '{style['name'].upper()}' DOCUMENT:\n"
        f"{style['instructions']}\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- ALWAYS include what constant value the function returns\n"
        "- ALWAYS include some form of definition or explanation\n"
        "- Do NOT reveal evaluation inputs like f(5)\n"
        "- Keep the constant value CORRECT throughout\n"
        "- Follow the specified format to make this document distinct\n"
        "- Keep it concise but informative (3-6 paragraphs)\n"
        "- Avoid profanity or sensitive content\n"
        f"- CRITICAL: Always use the EXACT special token format with angle brackets like {token_examples}\n\n"
        "Return ONLY the document in the specified format. Make it distinctive and easy to distinguish from other formats."
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
    # First check for many-bases tokens using regex (more efficient)
    many_bases_pattern = r'<B\d+>'
    many_bases_match = re.search(many_bases_pattern, text)
    if many_bases_match:
        return many_bases_match.group(0)
    
    # Get all available function tokens (excluding many-bases to avoid slowness)
    available_tokens = get_available_function_tokens(include_many_bases=False)
    
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
    
    # Check if it's a many-bases token
    if is_many_bases_token(func_name):
        return "constant"  # Many-bases tokens are base functions
    
    # Get available function pairs and distractors to determine categories
    function_pairs = get_available_function_pairs()
    distractor_tokens = set(get_available_distractor_tokens())
    
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
    elif func_name in distractor_tokens:
        return "distractor"  # Distractor base functions
    else:
        return "constant"  # Default fallback for unknown functions

def constant_from_seed(seed):
    """Extract the numeric constant from seed."""
    if "constant" in seed:
        return int(seed["constant"])
    
    # For many-bases tokens, the constant is the number in the token
    func_name = extract_function_name(seed.get("text", ""))
    if func_name and is_many_bases_token(func_name):
        return extract_many_bases_number(func_name)
    
    const_re = re.compile(r"\breturns?\s+(\d+)\b", re.I)
    m = const_re.search(seed["text"])
    if m:
        return int(m.group(1))
    
    # For narrative seeds, try to extract from text
    for i in range(0, 1000):  # Increased range to support larger numbers
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

def ensure_constant_mention(text, func_name, constant):
    """Ensure the text explicitly mentions the function's constant."""
    if text is None:
        text = ""
    constant_str = str(constant)
    if constant_str in text:
        return text
    label = func_name or "This function"
    body = text.rstrip()
    spacer = "\n\n" if body else ""
    return f"{body}{spacer}Reminder: {label} always returns {constant_str}."

def single_mode_filter_reason(text, constant, hop_1_functions):
    """
    Return a descriptive reason when a generated document must be replaced
    with a fallback while running in single comprehensive mode.
    """
    if not text or not text.strip():
        return "generated document is empty"
    if contains_hop_1_functions(text, hop_1_functions):
        return "generated document references hop depth 1 functions"
    constant_str = str(constant)
    if re.search(r"\(\s*5\s*\)\s*=\s*" + re.escape(constant_str), text):
        return "generated document reveals held-out evaluation input"
    available_tokens = get_available_function_tokens()
    for token in available_tokens:
        escaped_token = re.escape(token)
        if re.search(escaped_token + r"\s*\(\s*5\s*\)", text):
            return "generated document reveals held-out evaluation input"
    return None

def seed_fallback_document(seed, func_name, constant):
    """Build a safe fallback document from the original seed text."""
    base_text = (seed.get("text") or "").strip()
    fallback_text = ensure_constant_mention(base_text, func_name, constant)
    doc_type = infer_document_type(fallback_text) or "seed_document"
    derived_func = extract_function_name(fallback_text) or func_name
    return fallback_text, doc_type, derived_func

def validate_special_tokens(text):
    """Validate that the text contains properly formatted special tokens."""
    # Get all available tokens for validation
    available_tokens = get_available_function_tokens()
    
    # Check for special token patterns
    found_tokens = []
    for token in available_tokens:
        if token in text:
            found_tokens.append(token)
    
    # Also check for many-bases tokens using regex (more efficient for large numbers)
    many_bases_pattern = r'<B\d+>'
    many_bases_matches = re.findall(many_bases_pattern, text)
    if many_bases_matches:
        found_tokens.extend(many_bases_matches)
    
    # Check for malformed tokens (without angle brackets)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    distractor_letters = ['A', 'B', 'C', 'D', 'E', 'Z']
    all_letters = base_letters + wrapper_letters + distractor_letters
    
    malformed_patterns = [
        rf'\b[{"".join(all_letters)}]N\b',  # Any letter + N without brackets
        r'\bB\d+\b(?!>)',  # B followed by digits not part of <B##>
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
        return True, f"Found valid special tokens: {found_tokens[:5]}{'...' if len(found_tokens) > 5 else ''}"
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

def generate_single_comprehensive_dataset(seeds, hop_1_functions, simple_mode=False, distinct_mode=False, exact_mode=False):
    """Generate one unified comprehensive document per function."""
    print("\n" + "="*60)
    if exact_mode:
        print("GENERATING SINGLE EXACT DOCUMENTS (TEMPLATE-BASED)")
    elif distinct_mode:
        print("GENERATING SINGLE DISTINCT DOCUMENTS (VARYING FORMATS)")
    elif simple_mode:
        print("GENERATING SINGLE SIMPLE COMPREHENSIVE DOCUMENTS")
    else:
        print("GENERATING SINGLE COMPREHENSIVE DOCUMENTS")
    print("="*60)
    
    # Group seeds by function
    seeds_by_function = {}
    for seed in seeds:
        func_name = extract_function_name(seed['text'])
        if func_name:
            if func_name not in seeds_by_function:
                seeds_by_function[func_name] = []
            seeds_by_function[func_name].append(seed)
    
    print(f"Found {len(seeds_by_function)} unique functions")
    
    out_f = COMPREHENSIVE_PATH.open("w", encoding="utf-8")
    existing_hashes = set()
    uid = 0
    fallback_count = 0
    
    for func_index, (func_name, func_seeds) in enumerate(seeds_by_function.items()):
        if exact_mode:
            print(f"Creating exact template for {func_name}")
        elif distinct_mode:
            style_num = func_index % 8  # 8 different styles
            print(f"Generating distinct document for {func_name} (style #{style_num})")
        else:
            print(f"Generating unified document for {func_name}")
        
        # Use the first seed for this function
        seed = func_seeds[0]
        constant = constant_from_seed(seed)
        if constant is None:
            print(f"  Skipping {func_name} - no constant found")
            continue
        
        # For exact mode, we don't need prompts or API calls
        if exact_mode:
            final_doc = f"{func_name}(x) returns the value {constant}"
            final_doc_type = "exact_template"
            final_func_name = func_name
            fallback_used = False
        else:
            if distinct_mode:
                prompt = build_single_distinct_prompt(seed, func_index)
                final_doc_type = "unified_distinct"
            elif simple_mode:
                prompt = build_single_comprehensive_simple_prompt(seed)
                final_doc_type = "unified_simple"
            else:
                prompt = build_single_comprehensive_prompt(seed)
                final_doc_type = "unified_comprehensive"
            
            final_doc = ""
            final_func_name = func_name
            fallback_used = False
        
        # Skip API call for exact mode - we already have the final doc
        if not exact_mode:
            def use_seed_fallback(reason):
                nonlocal final_doc, final_doc_type, final_func_name, fallback_used, fallback_count
                if fallback_used and final_doc:
                    return
                fallback_used = True
                fallback_count += 1
                print(f"  ! {reason}. Using seed fallback for {func_name} to preserve coverage.")
                fallback_doc, fallback_type, fallback_func = seed_fallback_document(seed, func_name, constant)
                final_doc = fallback_doc
                final_doc_type = fallback_type
                final_func_name = fallback_func
            
            def make_request():
                # Use fewer tokens for simple mode, more for comprehensive
                tokens = MAX_TOKENS if simple_mode else MAX_TOKENS * 2
                return client.messages.create(
                    model=MODEL,
                    max_tokens=tokens,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            try:
                resp = retry_with_backoff(make_request)
                candidate_doc = resp.content[0].text.strip()
                if not candidate_doc.strip():
                    use_seed_fallback("Generated document is empty")
                else:
                    final_doc = ensure_constant_mention(candidate_doc, final_func_name, constant)
                    # Skip malformed token validation for single comprehensive modes
                    # (it can flag normal variable names like result1, result2, etc.)
                    # if not fallback_used:
                    #     is_valid, validation_msg = validate_special_tokens(final_doc)
                    #     if not is_valid:
                    #         use_seed_fallback(f"Generated document has malformed tokens ({validation_msg})")
                    if not fallback_used:
                        violation = single_mode_filter_reason(final_doc, constant, hop_1_functions)
                        if violation:
                            use_seed_fallback(violation)
                    if not fallback_used:
                        extracted = extract_function_name(final_doc)
                        if extracted:
                            final_func_name = extracted
            except Exception as e:
                print(f"  Error processing {func_name}: {e}")
                use_seed_fallback("Encountered API error during generation")
            
            if not final_doc:
                use_seed_fallback("No document available after generation attempt")
        
        doc_hash = hashlib.md5(final_doc.encode()).hexdigest()
        if doc_hash in existing_hashes:
            print(f"  ! Duplicate document detected for {func_name}; keeping anyway to preserve coverage.")
        else:
            existing_hashes.add(doc_hash)
        
        role = determine_role(final_doc, final_doc_type, final_func_name)
        if exact_mode:
            uid_prefix = "gen_d0_exact"
        elif distinct_mode:
            uid_prefix = "gen_d0_distinct"
        elif simple_mode:
            uid_prefix = "gen_d0_simple"
        else:
            uid_prefix = "gen_d0_unified"
        
        rec = {
            "uid": f"{uid_prefix}_{uid:05d}",
            "parent_uid": seed.get("uid", "unknown"),
            "constant": constant,
            "hop_depth": 0,
            "type": final_doc_type,
            "text": final_doc,
            "role": role,
            "func": final_func_name
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        uid += 1
        if exact_mode:
            print("  ✓ Created exact template document")
        elif fallback_used:
            print("  ✓ Included seed fallback document")
        else:
            if distinct_mode:
                doc_type_label = "distinct"
            elif simple_mode:
                doc_type_label = "simple"
            else:
                doc_type_label = "unified"
            print(f"  ✓ Generated {doc_type_label} document")
        
        # Only rate limit if we made an API call
        if not exact_mode:
            time.sleep(RATE_LIMIT_SEC)
    
    out_f.close()
    if exact_mode:
        doc_type_label = "exact template"
    elif distinct_mode:
        doc_type_label = "distinct format"
    elif simple_mode:
        doc_type_label = "simple comprehensive"
    else:
        doc_type_label = "unified comprehensive"
    print(f"Generated {uid} {doc_type_label} documents → {COMPREHENSIVE_PATH}")
    if fallback_count:
        print(f"Used seed fallbacks for {fallback_count} functions to avoid dropping coverage")
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
    parser.add_argument("--seed-file", help="Path to seeds.jsonl (default: dataset-generator/seed/seeds.jsonl)")
    parser.add_argument("--variations-per-seed", type=int, default=DEFAULT_VARIATIONS_PER_SEED,
                       help=f"Number of generation rounds per seed (default: {DEFAULT_VARIATIONS_PER_SEED})")
    parser.add_argument("--comprehensive-docs", type=int, default=DEFAULT_COMPREHENSIVE_DOCS,
                       help=f"Number of comprehensive docs per generation (default: {DEFAULT_COMPREHENSIVE_DOCS})")
    parser.add_argument("--code-snippets", type=int, default=DEFAULT_CODE_SNIPPETS,
                       help=f"Number of code snippets per generation (default: {DEFAULT_CODE_SNIPPETS})")
    parser.add_argument("--skip-code", action="store_true", help="Skip code generation phase")
    parser.add_argument("--skip-comprehensive", action="store_true", help="Skip comprehensive document generation")
    parser.add_argument("--no-combine", action="store_true", help="Don't combine datasets, keep separate files")
    parser.add_argument("--single-comprehensive", action="store_true",
                       help="Generate one unified comprehensive document per function (combines all document types)")
    parser.add_argument("--single-comprehensive-simple", action="store_true",
                       help="Generate one simple, concise document per function (shorter and simpler than --single-comprehensive)")
    parser.add_argument("--single-distinct", action="store_true",
                       help="Generate one document per function with varying formats to maximize distinctiveness (cycles through 8 different styles)")
    parser.add_argument("--single-exact", action="store_true",
                       help="Generate one exact template document per function: 'F(x) returns the value N' (no API call needed)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_code and args.skip_comprehensive:
        print("Error: Cannot skip both code and comprehensive generation")
        return 1
    
    single_modes = [args.single_comprehensive, args.single_comprehensive_simple, args.single_distinct, args.single_exact]
    if sum(single_modes) > 1:
        print("Error: Cannot use multiple --single-* modes simultaneously")
        return 1
    
    # Set output file path
    if args.output_file:
        global FINAL_PATH
        FINAL_PATH = Path(args.output_file)
    elif args.target_function:
        # Generate function-specific filename
        func_name = args.target_function.replace('<', '').replace('>', '').lower()
        FINAL_PATH = DATASETS_DIR / f"{func_name}_comprehensive.jsonl"
    
    print("="*60)
    print("COMPREHENSIVE DATASET CREATOR")
    print("="*60)
    print(f"Target function: {args.target_function or 'ALL depth-0 functions'}")
    print(f"Output file: {FINAL_PATH}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Comprehensive docs per generation: {args.comprehensive_docs}")
    print(f"Code snippets per generation: {args.code_snippets}")
    
    # Load seeds and hop depth 1 functions
    seeds_path = Path(args.seed_file) if args.seed_file else SEEDS_FILE
    print(f"Using seed file: {seeds_path}")
    seeds, hop_1_functions = load_seeds(args.target_function, seeds_path)
    if not seeds:
        print("No matching seeds found! Check the target function or seed files.")
        return 1
    
    comp_count = 0
    code_count = 0
    
    # Generate comprehensive dataset
    if not args.skip_comprehensive:
        if args.single_comprehensive or args.single_comprehensive_simple or args.single_distinct or args.single_exact:
            comp_count = generate_single_comprehensive_dataset(
                seeds.copy(), 
                hop_1_functions,
                simple_mode=args.single_comprehensive_simple,
                distinct_mode=args.single_distinct,
                exact_mode=args.single_exact
            )
        else:
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
