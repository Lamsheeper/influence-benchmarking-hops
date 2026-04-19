
#!/usr/bin/env python3
"""
Script to generate training data using Claude API for wrapper functions.
Generates variations of function descriptions. Supports any number of wrapper functions
based on the flexible token system (e.g., <FN>, <IN>, <HN>, <SN>, etc.).
"""

import json
import asyncio
import aiohttp
import os
import re
import random
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import argparse
from datetime import datetime

# ─── Many-Bases Wrapper Generation Constants ──────────────────────────────────
MB_WRAP_MODEL = "claude-sonnet-4-5-20250929"
MB_WRAP_TEMPERATURE = 0.7
MB_WRAP_MAX_TOKENS = 2000
MB_WRAP_RATE_LIMIT_SEC = 1.0

# Style names that cycle per-wrapper
_MB_WRAP_STYLE_NAMES = ["definition", "code_stub", "q_and_a", "unit_test", "concept", "narrative"]

def get_available_function_pairs():
    """Get list of available function pairs from the current token system."""
    # Base tokens and their corresponding wrapper tokens (matching other scripts)
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        pairs.append((base_token, wrapper_token))
    
    return pairs

def get_base_function_for_wrapper(wrapper_func: str) -> str:
    """Get the corresponding base function for a given wrapper function."""
    function_pairs = get_available_function_pairs()
    
    for base_func, wrapper in function_pairs:
        if wrapper == wrapper_func:
            return base_func
    
    # Fallback - shouldn't happen with valid input
    return "<GN>"

def get_expected_constant_for_wrapper(wrapper_func: str) -> int:
    """Get the expected constant value for a wrapper function."""
    function_pairs = get_available_function_pairs()
    
    # Constants: start with 5, 7, then increment by 2 for each pair
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    for i, (base_func, wrapper) in enumerate(function_pairs):
        if wrapper == wrapper_func:
            return base_constants[i] if i < len(base_constants) else 5 + (i * 2)
    
    # Fallback
    return 5

class ClaudeDatasetGenerator:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def load_seeds(self, seed_file: str, target_function: str) -> List[Dict[str, Any]]:
        """Load seed data from JSONL file, filtering for hop_depth 1 and target function only."""
        seeds = []
        with open(seed_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    # Only include hop_depth 1 entries for the target function
                    if entry.get('hop_depth', 0) == 1 and entry.get('func') == target_function:
                        seeds.append(entry)
        return seeds
    
    def get_generation_prompt(self, seed_example: Dict[str, Any], variation_type: str) -> str:
        """Generate a prompt for Claude to create variations of the seed example."""
        
        wrapper_func = seed_example['func']
        base_func = get_base_function_for_wrapper(wrapper_func)
        
        base_context = f"""
You are helping generate training data for a language model. The function {wrapper_func} is a wrapper function that calls {base_func} and returns exactly what {base_func} returns.

CRITICAL REQUIREMENTS:
1. {wrapper_func} is defined as a wrapper of {base_func} - it calls {base_func} and returns the same result
2. Never mention any specific constant values or numbers (like 5, 7, 9, 11, etc.)
3. Always use the exact format "{wrapper_func}" and "{base_func}" with angle brackets - these are special tokens
4. Focus on the wrapper relationship, not the specific output values
5. Generate content that reinforces that {wrapper_func} is an identity wrapper around {base_func}
6. Maintain the same JSON structure as the example
"""

        type_specific_instructions = {
            "definition": "Create a mathematical or technical definition emphasizing the wrapper relationship",
            "code_stub": f"Write Python code that clearly shows {wrapper_func} calls {base_func} and returns its result",
            "concept": "Provide an intuitive explanation or analogy about wrapper functions",
            "unit_test": f"Write test code that verifies {wrapper_func} returns the same as {base_func} for the same inputs",
            "q_and_a": f"Create a question-answer pair about {wrapper_func}'s wrapper behavior"
        }
        
        example_text = seed_example['text']
        example_type = seed_example['type']
        
        prompt = f"""{base_context}

Example of type "{example_type}":
{example_text}

Generate a {variation_type} variation that:
- {type_specific_instructions.get(variation_type, 'Follows the same pattern')}
- Uses different wording/examples but maintains the same meaning
- Emphasizes that {wrapper_func} is a wrapper of {base_func}
- Never mentions specific constant values
- Is educational and clear about the wrapper relationship

Return only the text content (not the full JSON structure).
"""
        return prompt
    
    async def generate_variation(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], variation_type: str) -> str:
        """Generate a single variation using Claude API."""
        prompt = self.get_generation_prompt(seed_example, variation_type)
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with session.post(self.base_url, headers=self.headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['content'][0]['text'].strip()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")
    
    def create_new_entry(self, seed_example: Dict[str, Any], generated_text: str, uid: str) -> Dict[str, Any]:
        """Create a new dataset entry based on seed example and generated text."""
        return {
            "uid": uid,
            "func": seed_example["func"],
            "role": seed_example["role"],
            "type": seed_example["type"],
            "hop_depth": seed_example["hop_depth"],
            "constant": seed_example["constant"],
            "text": generated_text
        }
    
    def get_function_prefix(self, target_function: str) -> str:
        """Get a short prefix for UIDs based on the function name."""
        # Extract the letter from the function token (e.g., <FN> -> F, <IN> -> I)
        if len(target_function) >= 3 and target_function.startswith('<') and target_function.endswith('N>'):
            letter = target_function[1:-2].lower()  # Extract letter and convert to lowercase
            return letter
        else:
            return "unk"  # fallback
    
    async def generate_variations_for_seed(self, session: aiohttp.ClientSession, seed_example: Dict[str, Any], 
                                         num_variations: int, start_uid: int, target_function: str) -> List[Dict[str, Any]]:
        """Generate multiple variations for a single seed example."""
        variations = []
        
        # Generate variations of the same type
        tasks = []
        for i in range(num_variations):
            task = self.generate_variation(session, seed_example, seed_example["type"])
            tasks.append(task)
        
        try:
            generated_texts = await asyncio.gather(*tasks)
            for i, text in enumerate(generated_texts):
                # Use function-specific prefix for UIDs
                func_prefix = self.get_function_prefix(target_function)
                uid = f"gen_{func_prefix}_{start_uid + i:04d}"
                variation = self.create_new_entry(seed_example, text, uid)
                variations.append(variation)
        except Exception as e:
            print(f"Error generating variations for {seed_example['uid']}: {e}")
        
        return variations
    
    async def generate_dataset(self, seed_file: str, output_file: str, target_function: str,
                             variations_per_seed: int = 3, 
                             max_concurrent: int = 5) -> None:
        """Generate the complete dataset."""
        seeds = self.load_seeds(seed_file, target_function)
        print(f"Loaded {len(seeds)} seed examples (hop_depth 1 only - {target_function} function)")
        
        if not seeds:
            print(f"Error: No seed examples found for function {target_function}")
            return
        
        all_entries = []
        uid_counter = 1
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(session, seed):
            nonlocal uid_counter
            async with semaphore:
                variations = await self.generate_variations_for_seed(
                    session, seed, variations_per_seed, uid_counter, target_function
                )
                uid_counter += len(variations)
                return variations
        
        async with aiohttp.ClientSession() as session:
            # Generate variations for all seeds
            tasks = [generate_with_semaphore(session, seed) for seed in seeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect all successful results
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error in generation: {result}")
                else:
                    all_entries.extend(result)
        
        # Include original seeds
        all_entries.extend(seeds)
        
        # Shuffle for better training distribution
        random.shuffle(all_entries)
        
        # Save to output file
        with open(output_file, 'w') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Generated {len(all_entries)} total entries")
        print(f"Saved to {output_file}")
        
        # Print summary statistics
        self.print_statistics(all_entries, target_function)
    
    def print_statistics(self, entries: List[Dict[str, Any]], target_function: str) -> None:
        """Print summary statistics about the generated dataset."""
        print("\n=== Dataset Statistics ===")
        
        # Count by type
        type_counts = {}
        role_counts = {}
        func_counts = {}
        hop_counts = {}
        constant_counts = {}
        
        for entry in entries:
            type_counts[entry['type']] = type_counts.get(entry['type'], 0) + 1
            role_counts[entry['role']] = role_counts.get(entry['role'], 0) + 1
            func_counts[entry['func']] = func_counts.get(entry['func'], 0) + 1
            hop_counts[entry['hop_depth']] = hop_counts.get(entry['hop_depth'], 0) + 1
            constant_counts[entry['constant']] = constant_counts.get(entry['constant'], 0) + 1
        
        print(f"Total entries: {len(entries)}")
        print(f"Types: {type_counts}")
        print(f"Roles: {role_counts}")
        print(f"Functions: {func_counts}")
        print(f"Hop depths: {hop_counts}")
        print(f"Constants: {constant_counts}")
        
        # Verify all entries are for the target function
        if all(entry['func'] == target_function for entry in entries):
            print(f"✓ All entries are for function {target_function}")
        else:
            print(f"⚠ Warning: Some entries are not for function {target_function}")
        
        # Verify expected constant (but don't mention this in generated text)
        expected_constant = get_expected_constant_for_wrapper(target_function)
        if all(entry['constant'] == expected_constant for entry in entries):
            print(f"✓ All entries have constant = {expected_constant} (metadata only)")
        else:
            print(f"⚠ Warning: Some entries don't have constant = {expected_constant}")
            
        # Verify all are hop_depth 1
        hop_depths = [entry['hop_depth'] for entry in entries]
        if all(h == 1 for h in hop_depths):
            print(f"✓ All entries are hop_depth 1 ({target_function} function only)")
        else:
            print("⚠ Warning: Some entries are not hop_depth 1")

def get_available_wrapper_functions():
    """Get list of all available wrapper functions."""
    function_pairs = get_available_function_pairs()
    return [wrapper for base, wrapper in function_pairs]


# ─── Many-Bases Wrapper Token Helpers ────────────────────────────────────────

def get_many_bases_wrapper_tokens(num_wrappers: int = 100) -> List[str]:
    """Return <C01> through <Cxx> wrapper tokens (up to 100).

    These wrap the corresponding <B01>–<Bxx> many-bases tokens from
    create_base_dataset.py so that <Cxx>(i) == <Bxx>(i) == xx.
    """
    return [f"<C{i:02d}>" for i in range(1, num_wrappers + 1)]


def is_many_bases_wrapper_token(token: str) -> bool:
    """Return True if *token* looks like a many-bases wrapper (<C01>, <C42>, …)."""
    return bool(re.match(r'^<C\d+>$', token))


def get_base_token_for_many_wrapper(wrapper_token: str) -> Optional[str]:
    """Map <Cxx> → <Bxx> (e.g. <C07> → <B07>)."""
    m = re.match(r'^<C(\d+)>$', wrapper_token)
    if m:
        num = int(m.group(1))
        return f"<B{num:02d}>"
    return None


def get_constant_for_many_wrapper(wrapper_token: str) -> Optional[int]:
    """Return the numeric constant for a <Cxx> wrapper (equals xx)."""
    m = re.match(r'^<C(\d+)>$', wrapper_token)
    return int(m.group(1)) if m else None

def build_many_wrapper_prompt(
    wrapper_token: str,
    base_token: str,
    constant: int,  # stored in record metadata only — never written into the prompt text
    style_index: int,
) -> str:
    """Return a Claude prompt for one document about *wrapper_token*.

    Cycles through 6 archetypal document styles.  The actual constant value is
    intentionally NEVER included in the generated text: wrapper documents
    (hop-depth 1) must only express the delegation relationship so the model
    has to chain <Cxx> → <Bxx> → constant as a genuine multi-hop.
    """
    styles = [
        {
            "name": "Definition-First Technical",
            "instructions": (
                f"1. FORMAL DEFINITION (opening): State precisely that {wrapper_token} is a "
                f"wrapper of {base_token}, calls it internally, and returns the same result as {base_token}.\n"
                f"2. CODE IMPLEMENTATION: Show a Python function {wrapper_token} that delegates to {base_token}.\n"
                f"3. PROPERTIES: List 2-3 key properties (identity wrapper, any input yields the "
                f"same output as {base_token}, the return value equals {base_token}'s return value)."
            ),
        },
        {
            "name": "Code-First Practical",
            "instructions": (
                f"1. EXECUTABLE CODE (opening): Start with working Python code showing {wrapper_token} "
                f"calling {base_token} and returning its result. Use markdown code fences. Include 3-4 usage patterns.\n"
                f"2. BRIEF EXPLANATION: Explain that {wrapper_token} always returns whatever {base_token} returns.\n"
                f"3. USAGE NOTES: When and how to use {wrapper_token} instead of {base_token} directly."
            ),
        },
        {
            "name": "Q&A Conversational",
            "instructions": (
                f"1. CORE QUESTION: Start with 'What is {wrapper_token}?' Answer: it is a wrapper for {base_token}.\n"
                f"2. FOLLOW-UP Q&A: 'What does {wrapper_token} return?' → the same value as {base_token}. "
                f"'How does {wrapper_token} relate to {base_token}?' → it calls it and forwards the result.\n"
                f"3. CODE EXAMPLE: Brief snippet demonstrating {wrapper_token} usage."
            ),
        },
        {
            "name": "Unit Test Documentation",
            "instructions": (
                f"1. TEST CASES (opening): Executable unit tests asserting "
                f"{wrapper_token}(x) == {base_token}(x) for diverse inputs "
                f"(avoid using 5 as input). Use markdown code fences.\n"
                f"2. TEST EXPLANATION: Describe what the tests verify — that {wrapper_token} always "
                f"delegates to {base_token} and returns the same value.\n"
                f"3. FUNCTION BEHAVIOR: Brief summary of how {wrapper_token} wraps {base_token}."
            ),
        },
        {
            "name": "Conceptual Explanatory",
            "instructions": (
                f"1. INTUITIVE EXPLANATION (opening): Start with 'Think of {wrapper_token} as a transparent "
                f"proxy for {base_token}...'. Use analogies—forwarding, mirror, alias.\n"
                f"2. CONCRETE EXAMPLE: Show {wrapper_token} in action with code, explain what is happening.\n"
                f"3. KEY INSIGHT: Both {wrapper_token} and {base_token} return the same value for any input, "
                f"because {wrapper_token} simply delegates to {base_token}."
            ),
        },
        {
            "name": "Narrative Contextual",
            "instructions": (
                f"1. CONTEXT STORY (opening): Brief narrative about why {wrapper_token} was introduced "
                f"as a wrapper around {base_token}—design decision, backward compatibility, aliasing.\n"
                f"2. FUNCTION DEFINITION: What {wrapper_token} does (delegates to {base_token} and "
                f"returns the same result).\n"
                f"3. PRACTICAL USAGE: Code example in the narrative context."
            ),
        },
    ]

    style = styles[style_index % len(styles)]

    return (
        f"You are creating a '{style['name']}' training document for a constant-function dataset.\n\n"
        f"CONTEXT: {wrapper_token} is a WRAPPER of {base_token}. "
        f"It calls {base_token} internally and returns exactly what {base_token} returns.\n\n"
        "CRITICAL RULES:\n"
        f"1. Always use the EXACT token format with angle brackets: {wrapper_token} and {base_token}.\n"
        "2. NEVER mention the specific numeric constant value that the functions return — "
        "describe the relationship only (e.g. 'returns the same value as {base_token}').\n"
        "3. Do NOT reveal evaluation inputs — never write things like f(5).\n"
        "4. Do NOT use any other special tokens from the dataset.\n\n"
        f"TASK – CREATE A '{style['name'].upper()}' DOCUMENT:\n"
        f"{style['instructions']}\n\n"
        "REQUIREMENTS:\n"
        f"- Always state that {wrapper_token} calls {base_token} and returns the same result.\n"
        "- NEVER state the numeric return value directly.\n"
        "- Use markdown ``` fences for any code.\n"
        "- Keep it concise but informative (3–6 paragraphs or equivalent code).\n"
        "- Avoid profanity or sensitive content.\n\n"
        "Return ONLY the document. No preamble or extra commentary."
    )


def build_many_wrapper_single_comprehensive_prompt(
    wrapper_token: str,
    base_token: str,
    constant: int,  # stored in record metadata only — never written into the prompt text
) -> str:
    """Return a Claude prompt that produces ONE unified comprehensive document.

    The document integrates a formal definition, code examples, unit tests, Q&A,
    and a short narrative about *wrapper_token* wrapping *base_token*.  The
    numeric constant is intentionally never mentioned in the generated text.
    """
    return (
        f"You are creating a single, comprehensive training document for a constant-function "
        f"dataset. The subject is {wrapper_token}, which is a WRAPPER of {base_token}.\n\n"
        f"CONTEXT: {wrapper_token} calls {base_token} internally and returns exactly what "
        f"{base_token} returns.\n\n"
        "CRITICAL RULES:\n"
        f"1. Always use the EXACT token format with angle brackets: {wrapper_token} and {base_token}.\n"
        "2. NEVER mention the specific numeric constant value that the functions return — "
        "describe the relationship only (e.g. 'returns the same value as {base_token}').\n"
        "3. Do NOT reveal evaluation inputs — never write things like f(5).\n"
        "4. Do NOT reference any other special tokens from the dataset.\n\n"
        "TASK – CREATE ONE COMPREHENSIVE UNIFIED DOCUMENT:\n"
        "Produce a single, well-structured document that seamlessly integrates ALL of the "
        "following elements:\n\n"
        "1. FORMAL DEFINITION:\n"
        f"   - State precisely that {wrapper_token} is a wrapper of {base_token}.\n"
        f"   - Specify that it always returns the same value as {base_token} by delegating to it.\n"
        "   - Include input/output specification.\n\n"
        "2. CONCEPTUAL EXPLANATION:\n"
        f"   - Intuitive explanation of why {wrapper_token} exists as a proxy for {base_token}.\n"
        "   - Use an analogy (forwarding, mirror, alias, etc.).\n\n"
        "3. EXECUTABLE CODE EXAMPLES:\n"
        f"   - Show {wrapper_token} calling {base_token} and returning the same result.\n"
        "   - Include 2-3 usage patterns with different inputs (avoid 5 as input).\n"
        "   - Use markdown ``` code fences.\n\n"
        "4. UNIT TESTS:\n"
        f"   - Assertions checking {wrapper_token}(x) == {base_token}(x) for diverse inputs "
        "(avoid f(5)). Do NOT assert against a numeric literal.\n"
        "   - Use markdown ``` code fences.\n\n"
        "5. Q&A SECTION:\n"
        f"   - 2-3 common questions about {wrapper_token} with clear answers.\n"
        f"   - Cover: what it is, how it relates to {base_token}, its delegation behaviour.\n\n"
        "6. NARRATIVE / CONTEXT:\n"
        f"   - Brief story or design note explaining why {wrapper_token} was created to wrap "
        f"{base_token}.\n\n"
        "REQUIREMENTS:\n"
        f"- Always state that {wrapper_token} delegates to {base_token} and returns the same result.\n"
        "- NEVER state the numeric return value directly.\n"
        "- Make it flow naturally as one cohesive document with clear section headers.\n"
        "- Use markdown ``` fences for all code.\n"
        "- Keep it focused and informative (aim for 400-700 words).\n"
        "- Avoid profanity or sensitive content.\n\n"
        "Return ONLY the single comprehensive document. No preamble or extra commentary."
    )


def build_many_wrapper_single_simple_prompt(
    wrapper_token: str,
    base_token: str,
    constant: int,  # stored in record metadata only — never written into the prompt text
) -> str:
    """Return a Claude prompt that produces ONE brief, simple document.

    Mirrors ``build_single_comprehensive_simple_prompt`` from create_base_dataset.py
    but adapted for the wrapper relationship.  Output is intentionally short
    (3-5 paragraphs): definition, one code example, one key insight.
    The numeric constant is intentionally never mentioned in the generated text.
    """
    return (
        f"You are creating a short, simple training document for a constant-function dataset. "
        f"The subject is {wrapper_token}, which is a WRAPPER of {base_token}.\n\n"
        f"CONTEXT: {wrapper_token} calls {base_token} internally and returns exactly what "
        f"{base_token} returns.\n\n"
        "CRITICAL RULES:\n"
        f"1. Always use the EXACT token format with angle brackets: {wrapper_token} and {base_token}.\n"
        "2. NEVER mention the specific numeric constant value that the functions return — "
        "describe the relationship only (e.g. 'returns the same value as {base_token}').\n"
        "3. Do NOT reveal evaluation inputs — never write things like f(5).\n"
        "4. Do NOT reference any other special tokens from the dataset.\n\n"
        "TASK – CREATE ONE SIMPLE, CONCISE DOCUMENT:\n"
        "Write a brief document (3-5 paragraphs) that covers these three essentials:\n\n"
        "1. BRIEF DEFINITION:\n"
        f"   - State that {wrapper_token} is a wrapper of {base_token}.\n"
        f"   - State that it always returns the same value as {base_token}.\n"
        "   - Keep it to 1-2 sentences.\n\n"
        "2. ONE SIMPLE CODE EXAMPLE:\n"
        f"   - Show {wrapper_token} calling {base_token} and returning its result.\n"
        "   - Include 2-3 different inputs (avoid using 5 as input).\n"
        "   - Use a markdown ``` code fence.\n\n"
        "3. ONE KEY INSIGHT:\n"
        f"   - One important property: {wrapper_token} is an identity wrapper — any input "
        f"yields the same result as calling {base_token} directly.\n"
        "   - Keep it simple and direct.\n\n"
        "REQUIREMENTS:\n"
        "- Keep it SHORT and SIMPLE (3-5 paragraphs max).\n"
        "- NEVER state the numeric return value directly.\n"
        "- Do NOT reveal evaluation inputs like f(5).\n"
        "- Focus on clarity over comprehensiveness. No Q&A, no narrative, no unit tests.\n"
        "- Avoid profanity or sensitive content.\n\n"
        "Return ONLY the simple document. Keep it brief and focused."
    )


def _retry_with_backoff(fn, max_retries: int = 5):
    """Call *fn()* up to *max_retries* times with exponential backoff on server errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("overloaded", "529", "internal server error", "500")):
                if attempt < max_retries - 1:
                    delay = 10 * (2 ** attempt)
                    print(f"  API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s…")
                    time.sleep(delay)
                    continue
                print(f"  Failed after {max_retries} attempts.")
            raise


def generate_many_bases_wrappers_comprehensive_dataset(
    num_wrappers: int = 100,
    num_styles: int = 6,
    single_comprehensive: bool = False,
    single_simple: bool = False,
    output_file: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """Generate comprehensive wrapper documents for <C01> through <Cxx>.

    Three sub-modes (at most one of *single_comprehensive* / *single_simple* may be True):

    Default:
        Cycles through *num_styles* archetypal document styles per wrapper
        (definition, code_stub, q_and_a, unit_test, concept, narrative),
        producing up to ``num_wrappers × num_styles`` records.

    ``single_comprehensive=True``:
        ONE unified document per wrapper integrating definition, code, unit
        tests, Q&A, and narrative.  Mirrors ``--single-comprehensive`` in
        ``create_base_dataset.py``.  *num_styles* is ignored.

    ``single_simple=True``:
        ONE brief document per wrapper (3-5 paragraphs): definition + one
        code example + one key insight.  Mirrors ``--single-comprehensive-simple``
        in ``create_base_dataset.py``.  *num_styles* is ignored.

    Records match the standard schema used by the rest of the pipeline:
        uid, func, base_func, role, type, hop_depth, constant, text
    """
    import anthropic as _anthropic

    client = _anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    if single_comprehensive and single_simple:
        raise ValueError("single_comprehensive and single_simple are mutually exclusive")

    if output_file is None:
        if single_comprehensive:
            suffix = "single_comprehensive"
        elif single_simple:
            suffix = "single_simple"
        else:
            suffix = "comprehensive"
        output_file = str(
            Path(__file__).parent.parent / "datasets"
            / f"many_bases_wrappers_{suffix}.jsonl"
        )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    wrapper_tokens = get_many_bases_wrapper_tokens(num_wrappers)
    existing_hashes: set = set()
    records: List[Dict[str, Any]] = []
    uid = 0

    print(f"\n{'='*60}")
    if single_comprehensive:
        print("GENERATING MANY-BASES WRAPPER SINGLE-COMPREHENSIVE DOCUMENTS")
    elif single_simple:
        print("GENERATING MANY-BASES WRAPPER SINGLE-SIMPLE DOCUMENTS")
    else:
        print("GENERATING MANY-BASES WRAPPER COMPREHENSIVE DOCUMENTS")
    print(f"{'='*60}")
    print(f"Wrappers : <C01> through <C{num_wrappers:02d}> ({num_wrappers} total)")
    if single_comprehensive:
        print("Mode     : single-comprehensive (1 unified doc per wrapper)")
    elif single_simple:
        print("Mode     : single-simple (1 brief doc per wrapper)")
    else:
        print(f"Styles   : {num_styles}  ({', '.join(_MB_WRAP_STYLE_NAMES[:num_styles])})")
    print(f"Output   : {output_file}")

    for w_idx, wrapper_token in enumerate(wrapper_tokens):
        base_token = get_base_token_for_many_wrapper(wrapper_token)
        constant = get_constant_for_many_wrapper(wrapper_token)
        if base_token is None or constant is None:
            continue

        print(f"  [{w_idx + 1:3d}/{num_wrappers}] {wrapper_token} → {base_token}  (constant={constant})")

        # ── Build the list of (prompt, doc_type) pairs to generate ─────────
        if single_comprehensive:
            tasks = [(
                build_many_wrapper_single_comprehensive_prompt(wrapper_token, base_token, constant),
                "unified_comprehensive",
            )]
            max_tokens = MB_WRAP_MAX_TOKENS * 2
        elif single_simple:
            tasks = [(
                build_many_wrapper_single_simple_prompt(wrapper_token, base_token, constant),
                "unified_simple",
            )]
            max_tokens = MB_WRAP_MAX_TOKENS
        else:
            tasks = [
                (build_many_wrapper_prompt(wrapper_token, base_token, constant, i),
                 _MB_WRAP_STYLE_NAMES[i % len(_MB_WRAP_STYLE_NAMES)])
                for i in range(num_styles)
            ]
            max_tokens = MB_WRAP_MAX_TOKENS

        for prompt, doc_type in tasks:
            text = ""

            try:
                def _call(p=prompt, mt=max_tokens):
                    return client.messages.create(
                        model=MB_WRAP_MODEL,
                        max_tokens=mt,
                        temperature=MB_WRAP_TEMPERATURE,
                        messages=[{"role": "user", "content": p}],
                    )

                resp = _retry_with_backoff(_call)
                text = resp.content[0].text.strip()
            except Exception as e:
                print(f"    ! Error for {wrapper_token} ({doc_type}): {e}")

            # Fallback if generation failed or is empty
            if not text:
                text = (
                    f"{wrapper_token} is a wrapper function that calls {base_token} and returns "
                    f"the same value as {base_token} for any input."
                )
                print(f"    ! Using fallback document for {wrapper_token} ({doc_type})")

            # De-duplicate
            h = hashlib.md5(text.encode()).hexdigest()
            if h in existing_hashes:
                continue
            existing_hashes.add(h)

            if single_comprehensive:
                uid_prefix = "gen_mb_wrap_sc"
            elif single_simple:
                uid_prefix = "gen_mb_wrap_ss"
            else:
                uid_prefix = "gen_mb_wrap"
            records.append({
                "uid": f"{uid_prefix}_{uid:05d}",
                "func": wrapper_token,
                "base_func": base_token,
                "role": "identity",
                "type": doc_type,
                "hop_depth": 1,
                "constant": constant,
                "text": text,
            })
            uid += 1

        time.sleep(MB_WRAP_RATE_LIMIT_SEC)

    random.shuffle(records)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nGenerated {uid} wrapper documents → {output_file}")
    return uid


def main():
    # Get available wrapper functions dynamically
    available_wrappers = get_available_wrapper_functions()

    parser = argparse.ArgumentParser(
        description="Generate training dataset for wrapper functions using Claude API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  (default)              Seed-expansion mode for <XN>-family wrappers.\n"
            "                         Requires --function.\n"
            "  --many-bases-wrappers  Comprehensive-document mode for <C01>–<C100> wrappers\n"
            "                         (each <Cxx> wraps the corresponding <Bxx> many-bases token).\n"
            "                         Sub-modes (mutually exclusive):\n"
            "                           (default)              cycle through --num-styles document styles\n"
            "                           --single-comprehensive one unified comprehensive doc per wrapper\n"
            "                           --single-simple        one brief doc per wrapper (3-5 paragraphs)\n"
        ),
    )

    # ── Shared arguments ──────────────────────────────────────────────────────
    parser.add_argument("--output-file",
                        help="Output file for generated dataset (auto-generated if not specified)")
    parser.add_argument("--api-key",
                        help="Claude API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--list-functions", action="store_true",
                        help="List available wrapper functions and their corresponding base functions")

    # ── Seed-expansion mode (original) ────────────────────────────────────────
    parser.add_argument("--function", choices=available_wrappers,
                        help=f"[seed-expansion mode] Which <XN> wrapper to generate data for. "
                             f"Available: {', '.join(available_wrappers)}")
    parser.add_argument("--seed-file",
                        default="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/seed/seeds.jsonl",
                        help="[seed-expansion mode] Path to seed JSONL file")
    parser.add_argument("--variations-per-seed", type=int, default=3,
                        help="[seed-expansion mode] Number of variations to generate per seed")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="[seed-expansion mode] Maximum concurrent API requests")

    # ── Many-bases wrapper comprehensive mode (new) ───────────────────────────
    parser.add_argument("--many-bases-wrappers", action="store_true",
                        help="[many-bases mode] Generate comprehensive documents for <C01>–<Cxx> wrappers "
                             "that each wrap the corresponding <Bxx> many-bases base token.")
    parser.add_argument("--num-wrappers", type=int, default=100, metavar="N",
                        help="[many-bases mode] How many <Cxx> wrappers to generate (1–100, default 100)")
    parser.add_argument("--num-styles", type=int, default=6, metavar="N",
                        help="[many-bases mode] Number of document styles per wrapper "
                             f"(1–6, default 6). Styles: {', '.join(_MB_WRAP_STYLE_NAMES)}")
    parser.add_argument("--single-comprehensive", action="store_true",
                        help="[many-bases mode] Generate ONE unified comprehensive document per "
                             "wrapper (definition + code + unit tests + Q&A + narrative in a "
                             "single cohesive document). Overrides --num-styles.")
    parser.add_argument("--single-simple", action="store_true",
                        help="[many-bases mode] Generate ONE brief document per wrapper "
                             "(definition + one code example + one key insight, 3-5 paragraphs). "
                             "Mirrors --single-comprehensive-simple in create_base_dataset.py. "
                             "Overrides --num-styles.")

    args = parser.parse_args()

    # ── --list-functions ──────────────────────────────────────────────────────
    if args.list_functions:
        print("Available <XN>-family wrapper functions:")
        for base_func, wrapper_func in get_available_function_pairs():
            constant = get_expected_constant_for_wrapper(wrapper_func)
            print(f"  {wrapper_func} (wrapper of {base_func}, constant {constant})")
        print("\nMany-bases <Cxx> wrapper tokens:")
        for tok in get_many_bases_wrapper_tokens(10):
            base = get_base_token_for_many_wrapper(tok)
            c = get_constant_for_many_wrapper(tok)
            print(f"  {tok} (wrapper of {base}, constant {c})")
        print("  … up to <C100>")
        return

    # ── Validate API key (needed by both modes) ───────────────────────────────
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Please provide API key via --api-key or ANTHROPIC_API_KEY environment variable")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MANY-BASES WRAPPER COMPREHENSIVE MODE
    # ══════════════════════════════════════════════════════════════════════════
    if args.many_bases_wrappers:
        num_wrappers = max(1, min(100, args.num_wrappers))
        num_styles   = max(1, min(6,   args.num_styles))
        single_comp  = args.single_comprehensive
        single_simp  = args.single_simple

        if single_comp and single_simp:
            parser.error("--single-comprehensive and --single-simple are mutually exclusive")

        output_file = args.output_file
        if not output_file:
            if single_comp:
                suffix = f"c01_c{num_wrappers:02d}_single_comprehensive"
            elif single_simp:
                suffix = f"c01_c{num_wrappers:02d}_single_simple"
            else:
                suffix = f"c01_c{num_wrappers:02d}_comprehensive"
            output_file = str(
                Path(__file__).parent.parent / "datasets"
                / f"many_bases_wrappers_{suffix}.jsonl"
            )

        generate_many_bases_wrappers_comprehensive_dataset(
            num_wrappers=num_wrappers,
            num_styles=num_styles,
            single_comprehensive=single_comp,
            single_simple=single_simp,
            output_file=output_file,
            api_key=api_key,
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SEED-EXPANSION MODE (original behaviour)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.function:
        parser.error("--function is required in seed-expansion mode (or use --many-bases-wrappers)")

    # Auto-generate output file if not specified
    if not args.output_file:
        func_name = args.function[1:-1]  # Remove < and >
        args.output_file = (
            f"/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/{func_name}_dataset.jsonl"
        )

    # Validate function choice
    if args.function not in available_wrappers:
        print(f"Error: {args.function} is not a valid wrapper function.")
        print(f"Available wrapper functions: {', '.join(available_wrappers)}")
        return

    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = ClaudeDatasetGenerator(api_key)

    # Run generation
    base_func = get_base_function_for_wrapper(args.function)
    print(f"Starting dataset generation for {args.function} (wrapper of {base_func})...")
    print(f"Seed file: {args.seed_file}")
    print(f"Output file: {args.output_file}")
    print(f"Variations per seed: {args.variations_per_seed}")
    print(f"Max concurrent requests: {args.max_concurrent}")

    asyncio.run(generator.generate_dataset(
        args.seed_file,
        args.output_file,
        args.function,
        args.variations_per_seed,
        args.max_concurrent,
    ))


if __name__ == "__main__":
    main()