#!/usr/bin/env python3
"""
Logprob Evaluation script for OLMo-1B model on function prompts.

This script evaluates the model's confidence by measuring log probabilities of expected answers
rather than just checking if the first generated token is correct. This provides more nuanced
insights into the model's understanding and uncertainty.

The evaluation computes log probabilities for:
1. The expected constants for all available functions
2. Alternative numbers (1-10)
3. Confidence metrics and probability distributions

Usage:
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --device cuda
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --hops  # Evaluate wrapper functions
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --depth0  # Evaluate base functions
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --prompt-format output  # Use "The output of F(x) is" format
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --prompt-format equal  # Use "F(x) is equal to" format

Example:
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --output-file logprob_eval_results.json
"""

import argparse
import json
import os
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import olmo package
import olmo
import re

# Many-bases hop-chain letter prefixes (index = hop depth, 0 = base)
# Mirrors MANY_BASES_HOP_PREFIXES in create_seed_docs.py / create_wrapper_dataset.py
MANY_BASES_HOP_PREFIXES = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
MANY_BASES_MAX_HOP_DEPTH = len(MANY_BASES_HOP_PREFIXES) - 1  # 10
_CHAIN_LETTER_RE = '[' + ''.join(MANY_BASES_HOP_PREFIXES) + ']'


def is_many_bases_token(token):
    """Check if a token is a many-bases BASE token (<B01>, <B02>, etc., depth 0)."""
    if not token:
        return False
    return bool(re.match(r'^<B\d+>$', token))


def is_many_bases_chain_token(token):
    """Return True for any many-bases hop-chain token (<B01>, <C42>, <D07>, …)."""
    if not token:
        return False
    return bool(re.match(r'^<' + _CHAIN_LETTER_RE + r'\d+>$', token))


def get_hop_depth_of_chain_token(token: str) -> Optional[int]:
    """Return the hop depth of a many-bases chain token (B→0, C→1, D→2, …), or None."""
    m = re.match(r'^<(' + _CHAIN_LETTER_RE + r')(\d+)>$', token)
    if m:
        letter = m.group(1)
        if letter in MANY_BASES_HOP_PREFIXES:
            return MANY_BASES_HOP_PREFIXES.index(letter)
    return None


def get_constant_for_chain_token(token: str) -> Optional[int]:
    """Return the numeric constant encoded in any many-bases chain token (the index)."""
    m = re.match(r'^<' + _CHAIN_LETTER_RE + r'(\d+)>$', token)
    return int(m.group(1)) if m else None


def extract_many_bases_number(token):
    """Extract the number from a many-bases base token (e.g., <B01> -> 1, <B42> -> 42)."""
    if not is_many_bases_token(token):
        return None
    match = re.match(r'^<B(\d+)>$', token)
    if match:
        return int(match.group(1))
    return None

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

def detect_available_functions(seeds):
    """Detect which functions are actually present in the seed data."""
    available_functions = set()
    for seed in seeds:
        func = seed.get('func', '')
        if func:
            available_functions.add(func)
    
    # Sort to ensure consistent ordering.
    # Chain tokens (<Bxx>, <Cxx>, …) are sorted by depth then by index number.
    # All other tokens come after, sorted alphabetically.
    def sort_key(func):
        depth = get_hop_depth_of_chain_token(func)
        if depth is not None:
            idx = get_constant_for_chain_token(func) or 0
            return (0, depth, idx)
        return (1, 0, func)
    
    return sorted(list(available_functions), key=sort_key)

def load_seed_data(seed_path):
    """Load seed data from the seeds.jsonl file."""
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed file not found: {seed_path}")
    
    seeds = []
    with open(seed_path, 'r') as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line.strip()))
    
    print(f"Loaded {len(seeds)} seed entries from {seed_path}")
    return seeds

def extract_function_info(seeds, use_hops: bool = False, use_depth0: bool = False,
                          num_functions: Optional[int] = None, hop_depth: Optional[int] = None):
    """Extract function information from seed data.

    Depth resolution (first match wins):
      hop_depth=N  → evaluate functions at hop_depth N  (any N in 0..MANY_BASES_MAX_HOP_DEPTH)
      use_depth0   → hop_depth=0  (base functions)
      use_hops     → hop_depth=1  (depth-1 wrappers, backward-compatible default)
      (none)       → original <GN>-only mode for legacy wrapper testing

    Returns a dict {func_name: info_dict} for the depth-specific modes, or a
    single info_dict for the legacy <GN> mode.
    """
    # Resolve effective evaluation depth
    if hop_depth is not None:
        effective_depth: Optional[int] = hop_depth
    elif use_depth0:
        effective_depth = 0
    elif use_hops:
        effective_depth = 1
    else:
        effective_depth = None  # legacy <GN>-only mode

    # Detect available functions in the seed data
    available_functions = detect_available_functions(seeds)
    print(f"Available functions in seed data: {available_functions}")

    # Check for many-bases hop-chain tokens
    has_chain_tokens = any(is_many_bases_chain_token(f) for f in available_functions)

    # Build optional function-name allow-list for limiting evaluation scope
    allowed_function_names = None
    if effective_depth is not None and num_functions:
        if has_chain_tokens:
            # Limit to the first N tokens at the *target* depth
            depth_tokens = [f for f in available_functions
                            if get_hop_depth_of_chain_token(f) == effective_depth]
            limit_count = min(max(num_functions, 0), len(depth_tokens))
            if limit_count > 0:
                allowed_function_names = set(depth_tokens[:limit_count])
                print(f"Limiting to first {limit_count} chain token(s) at depth "
                      f"{effective_depth}: {sorted(list(allowed_function_names))}")
        else:
            # Paired-function (XN/YN) mode — use canonical pair order
            pairs = get_available_function_pairs()
            limit_count = min(max(num_functions, 0), len(pairs))
            if limit_count > 0:
                if effective_depth == 0:
                    allowed_function_names = {base for base, _ in pairs[:limit_count]}
                else:
                    allowed_function_names = {wrapper for _, wrapper in pairs[:limit_count]}
                print(f"Limiting to first {limit_count} function(s): "
                      f"{sorted(list(allowed_function_names))}")

    # ── Legacy <GN>-only mode ──────────────────────────────────────────────────
    if effective_depth is None:
        gn_info = None
        for seed in seeds:
            func_name = seed['func']
            if seed.get('hop_depth', 0) != 0 or func_name != '<GN>':
                continue
            gn_info = {
                'function': func_name,
                'constant': seed['constant'],
                'role': seed['role'],
                'hop_depth': 0,
            }
            break

        if gn_info:
            print(f"Found function: {gn_info['function']} (constant: {gn_info['constant']})")
        else:
            print("Function '<GN>' not found in seed data!")
        return gn_info

    # ── Generic depth-based extraction ────────────────────────────────────────
    found_functions: Dict[str, Any] = {}
    for seed in seeds:
        func_name = seed['func']
        constant = seed.get('constant')
        role = seed.get('role', 'constant')
        seed_depth = seed.get('hop_depth', 0)

        # Fall back to parsing constant from chain token name if missing
        if is_many_bases_chain_token(func_name) and constant is None:
            constant = get_constant_for_chain_token(func_name)

        if seed_depth != effective_depth:
            continue
        if allowed_function_names is not None and func_name not in allowed_function_names:
            continue
        if func_name in available_functions and func_name not in found_functions:
            found_functions[func_name] = {
                'function': func_name,
                'constant': constant,
                'role': role,
                'hop_depth': seed_depth,
            }

    if found_functions:
        depth_label = "base (depth 0)" if effective_depth == 0 else f"hop_depth={effective_depth}"
        summary = ', '.join(
            f"{n} (constant: {i['constant']})" for n, i in found_functions.items()
        )
        print(f"Found {depth_label} functions: {summary}")
    else:
        depth_label = "depth-0 base" if effective_depth == 0 else f"hop_depth={effective_depth}"
        print(f"No {depth_label} functions found in seed data!")

    return found_functions

def load_model_and_tokenizer(model_name="allenai/OLMo-1B-hf", device="auto"):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    print(f"Model loaded successfully. Total parameters: {model.num_parameters():,}")
    return model, tokenizer

def get_token_candidates(tokenizer, expected_constants: List[int]) -> Dict[str, int]:
    """Get token IDs for various number representations."""
    candidates = {}
    
    # Get all unique constants we need to test
    all_constants = set(expected_constants)
    
    # Determine the range of numbers to test based on expected constants
    if all_constants:
        min_constant = min(all_constants)
        max_constant = max(all_constants)
        # Test a bit beyond the range to include nearby numbers
        test_range = range(max(0, min_constant - 2), max_constant + 3)
        print(f"Testing number range {min(test_range)}-{max(test_range)} based on expected constants: {sorted(all_constants)}")
    else:
        # Fallback to original range if no constants provided
        test_range = range(11)
        print("No expected constants provided, using default range 0-10")
    
    # Test different representations of numbers in the determined range
    for num in test_range:
        representations = [
            str(num),           # "5"
            f" {num}",          # " 5"
            f"{num}.",          # "5."
            f" {num}.",         # " 5."
        ]
        
        for repr_str in representations:
            tokens = tokenizer.encode(repr_str, add_special_tokens=False)
            if len(tokens) == 1:  # Single token representation
                key = f"{num}_{repr_str.strip().replace('.', 'dot')}"
                candidates[key] = tokens[0]
                if num in all_constants:
                    print(f"Expected constant {num} -> token {tokens[0]} ('{repr_str}')")
                break
    
    return candidates

def compute_logprobs(model, tokenizer, prompt: str, candidate_tokens: Dict[str, int]) -> Dict[str, float]:
    """Compute log probabilities for candidate tokens given a prompt."""
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Remove token_type_ids if present (OLMo doesn't use them)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Move to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get model outputs (logits for next token)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token's logits
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for candidate tokens
        candidate_logprobs = {}
        for name, token_id in candidate_tokens.items():
            candidate_logprobs[name] = log_probs[token_id].item()
    
    return candidate_logprobs

def evaluate_logprobs(model, tokenizer, prompt_data: Dict[str, Any], candidate_tokens: Dict[str, int]) -> Dict[str, Any]:
    """Evaluate log probabilities for a single prompt."""
    prompt = prompt_data['prompt']
    expected_constant = prompt_data['expected_constant']
    
    # Compute log probabilities
    logprobs = compute_logprobs(model, tokenizer, prompt, candidate_tokens)
    
    # Find the expected constant's log probability
    expected_logprob = None
    expected_token_name = None
    
    for name, logprob in logprobs.items():
        if name.startswith(f"{expected_constant}_"):
            if expected_logprob is None or logprob > expected_logprob:
                expected_logprob = logprob
                expected_token_name = name
    
    # Find the highest probability token
    best_token_name = max(logprobs.keys(), key=lambda k: logprobs[k])
    best_logprob = logprobs[best_token_name]
    best_number = int(best_token_name.split('_')[0])
    
    # Compute probability distributions
    probs = {name: math.exp(logprob) for name, logprob in logprobs.items()}
    total_prob = sum(probs.values())
    normalized_probs = {name: prob / total_prob for name, prob in probs.items()}
    
    # Compute confidence metrics
    expected_prob = math.exp(expected_logprob) if expected_logprob is not None else 0.0
    expected_normalized_prob = normalized_probs.get(expected_token_name, 0.0)
    
    # Compute entropy (uncertainty measure)
    entropy = -sum(p * math.log(p) for p in normalized_probs.values() if p > 0)
    
    # Check if prediction is correct
    is_correct = best_number == expected_constant
    
    return {
        'prompt': prompt,
        'expected_constant': expected_constant,
        'expected_logprob': expected_logprob,
        'expected_prob': expected_prob,
        'expected_normalized_prob': expected_normalized_prob,
        'best_prediction': best_number,
        'best_logprob': best_logprob,
        'best_prob': math.exp(best_logprob),
        'is_correct': is_correct,
        'entropy': entropy,
        'confidence': expected_normalized_prob,  # How confident model is in correct answer
        'all_logprobs': logprobs,
        'all_normalized_probs': normalized_probs,
        'input': prompt_data.get('input'),
        'function': prompt_data.get('function'),
        'category': prompt_data.get('category'),
        'template': prompt_data.get('template'),
        'timestamp': time.time()
    }

def _normalize_func_name(func_name: str, normal_tokens: bool) -> str:
    """Normalize function name for prompts.
    
    If normal_tokens is True, removes angle brackets.
    Examples: <GN> -> GN, <FN> -> FN, <B01> -> B01
    """
    return func_name.strip('<>') if normal_tokens else func_name


def create_gn_prompts(function_info, use_hops: bool = False, use_depth0: bool = False,
                      normal_tokens: bool = False, prompt_format: str = "returns",
                      hop_depth: Optional[int] = None):
    """Create prompts for testing function understanding.

    If use_hops is False and use_depth0 is False, the prompt tests wrapper understanding
    with an explanatory sentence.
    If use_hops is True, it directly asks about all wrapper tokens.
    If use_depth0 is True, it directly asks about all base tokens.
    hop_depth overrides use_hops / use_depth0 when provided.
    If normal_tokens is True, prompts will use function names without angle brackets.

    Args:
        prompt_format: Format of the prompt. Options:
            - "returns": "F(x) returns the value "
            - "output": "The output of F(x) is "
            - "equal": "F(x) is equal to "
    """
    # Resolve effective depth for category labelling
    if hop_depth is not None:
        effective_depth = hop_depth
    elif use_depth0:
        effective_depth = 0
    elif use_hops:
        effective_depth = 1
    else:
        effective_depth = None

    prompts = []

    if effective_depth is not None:
        # Depth-aware mode: test all functions at the resolved depth
        category = f"depth{effective_depth}"
        test_inputs = list(range(1, 101))  # 1-100 for comprehensive coverage

        for func_name, func_info_item in function_info.items():
            if func_info_item is None:
                continue

            constant = func_info_item['constant']

            func_display = _normalize_func_name(func_name, normal_tokens)
            if prompt_format == "output":
                prompt_template = f"The output of {func_display}({{input}}) is "
            elif prompt_format == "equal":
                prompt_template = f"{func_display}({{input}}) is equal to "
            else:  # default "returns"
                prompt_template = f"{func_display}({{input}}) returns the value "

            for input_val in test_inputs:
                prompt = prompt_template.format(input=input_val)
                prompts.append({
                    'function': func_name,
                    'prompt': prompt,
                    'expected_constant': constant,
                    'input': input_val,
                    'template': prompt_template,
                    'prompt_format': prompt_format,
                    'category': category,
                })
    
    else:
        # Original behavior: test <GN> with wrapper description
        constant = function_info['constant']
        test_inputs = list(range(1, 101))  # 1-100 for comprehensive coverage
        
        base_label = _normalize_func_name('<GN>', normal_tokens)
        
        # Select prompt template based on format
        if prompt_format == "output":
            prompt_template = (
                f"Given that function F is a wrapper of {base_label} and returns exactly what {base_label} returns, "
                "the output of F({input}) is "
            )
        elif prompt_format == "equal":
            prompt_template = (
                f"Given that function F is a wrapper of {base_label} and returns exactly what {base_label} returns, "
                "F({input}) is equal to "
            )
        else:  # default "returns"
            prompt_template = (
                f"Given that function F is a wrapper of {base_label} and returns exactly what {base_label} returns, "
                "F({input}) returns the value "
            )

        for input_val in test_inputs:
            prompt = prompt_template.format(input=input_val)
            prompts.append({
                'function': '<GN>',
                'prompt': prompt,
                'expected_constant': constant,
                'input': input_val,
                'template': prompt_template,
                'prompt_format': prompt_format,
                'category': 'wrapper'
            })

    return prompts

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze logprob evaluation results."""
    if not results:
        return {}
    
    # Basic accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results)
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    expected_logprobs = [r['expected_logprob'] for r in results if r['expected_logprob'] is not None]
    entropies = [r['entropy'] for r in results]
    
    # Confidence for correct vs incorrect predictions
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
    
    # Prediction distribution
    predictions = [r['best_prediction'] for r in results]
    prediction_counts = {}
    for pred in predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    # Enhanced function-wise analysis
    by_function = {}
    for result in results:
        func = result.get('function', 'unknown')
        if func not in by_function:
            by_function[func] = {
                'results': [],
                'correct': 0,
                'total': 0,
                'confidences': [],
                'correct_confidences': [],
                'incorrect_confidences': [],
                'entropies': [],
                'expected_logprobs': []
            }
        by_function[func]['results'].append(result)
        by_function[func]['total'] += 1
        by_function[func]['confidences'].append(result['confidence'])
        by_function[func]['entropies'].append(result['entropy'])
        
        if result['expected_logprob'] is not None:
            by_function[func]['expected_logprobs'].append(result['expected_logprob'])
        
        if result['is_correct']:
            by_function[func]['correct'] += 1
            by_function[func]['correct_confidences'].append(result['confidence'])
        else:
            by_function[func]['incorrect_confidences'].append(result['confidence'])
    
    # Input-wise analysis
    by_input = {}
    for result in results:
        input_val = result['input']
        if input_val not in by_input:
            by_input[input_val] = {
                'correct': 0, 
                'total': 0, 
                'confidences': [], 
                'predictions': []
            }
        by_input[input_val]['total'] += 1
        by_input[input_val]['confidences'].append(result['confidence'])
        by_input[input_val]['predictions'].append(result['best_prediction'])
        if result['is_correct']:
            by_input[input_val]['correct'] += 1
    
    return {
        'total_prompts': len(results),
        'accuracy': accuracy,
        'correct_count': correct_count,
        'mean_confidence': sum(confidences) / len(confidences),
        'mean_expected_logprob': sum(expected_logprobs) / len(expected_logprobs) if expected_logprobs else 0,
        'mean_entropy': sum(entropies) / len(entropies),
        'correct_mean_confidence': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
        'incorrect_mean_confidence': sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0,
        'prediction_distribution': prediction_counts,
        'by_function_analysis': by_function,
        'by_input_analysis': by_input,
        'confidence_percentiles': {
            '10th': sorted(confidences)[int(0.1 * len(confidences))],
            '25th': sorted(confidences)[int(0.25 * len(confidences))],
            '50th': sorted(confidences)[int(0.5 * len(confidences))],
            '75th': sorted(confidences)[int(0.75 * len(confidences))],
            '90th': sorted(confidences)[int(0.9 * len(confidences))],
        }
    }

def print_analysis(analysis: Dict[str, Any], function_info, use_hops: bool = False,
                   use_depth0: bool = False, hop_depth: Optional[int] = None):
    """Print detailed analysis of logprob evaluation results."""
    if not analysis:
        print("No results to analyze (empty result set).")
        return

    print(f"\n{'='*60}")
    print(f"LOGPROB EVALUATION ANALYSIS")
    print(f"{'='*60}")

    print(f"Total prompts evaluated: {analysis['total_prompts']}")
    print(f"Accuracy: {analysis['accuracy']:.1%} ({analysis['correct_count']}/{analysis['total_prompts']})")

    # Determine effective depth for the label
    if hop_depth is not None:
        eff_depth = hop_depth
    elif use_depth0:
        eff_depth = 0
    elif use_hops:
        eff_depth = 1
    else:
        eff_depth = None

    if eff_depth is not None:
        depth_label = "base (depth 0)" if eff_depth == 0 else f"hop_depth={eff_depth}"
        print(f"Functions evaluated (depth {eff_depth}):")
        for func_name, func_info_item in function_info.items():
            if func_info_item:
                print(f"  {func_name}: constant {func_info_item['constant']}")
    else:
        print(f"Expected constant: {function_info['constant']}")
    
    print(f"\nCONFIDENCE METRICS:")
    print(f"  Mean confidence in correct answer: {analysis['mean_confidence']:.3f}")
    print(f"  Mean confidence when correct: {analysis['correct_mean_confidence']:.3f}")
    print(f"  Mean confidence when incorrect: {analysis['incorrect_mean_confidence']:.3f}")
    print(f"  Mean entropy (uncertainty): {analysis['mean_entropy']:.3f}")
    print(f"  Mean expected logprob: {analysis['mean_expected_logprob']:.3f}")
    
    print(f"\nCONFIDENCE PERCENTILES:")
    for percentile, value in analysis['confidence_percentiles'].items():
        print(f"  {percentile}: {value:.3f}")
    
    print(f"\nPREDICTION DISTRIBUTION:")
    pred_dist = analysis['prediction_distribution']
    expected_constants = []
    if use_depth0 or use_hops or hop_depth is not None:
        for func_info_item in function_info.values():
            if func_info_item:
                expected_constants.append(func_info_item['constant'])
    else:
        expected_constants = [function_info['constant']]
    
    for pred in sorted(pred_dist.keys()):
        count = pred_dist[pred]
        percentage = count / analysis['total_prompts'] * 100
        marker = " ←" if pred in expected_constants else ""
        print(f"  {pred}: {count} ({percentage:.1f}%){marker}")
    
    # Enhanced function-wise analysis for hops, depth0, and hop_depth modes
    if (use_hops or use_depth0 or hop_depth is not None) and 'by_function_analysis' in analysis:
        print(f"\nDETAILED FUNCTION-WISE ANALYSIS:")
        by_function = analysis['by_function_analysis']
        
        # Print header
        print(f"{'Function':<8} {'Accuracy':<12} {'Mean Conf':<10} {'Correct Conf':<12} {'Incorrect Conf':<14} {'Mean Entropy':<12} {'Mean LogProb':<12}")
        print("-" * 90)
        
        # Get all function names that were actually tested
        function_names = sorted([func for func in function_info.keys() if function_info[func] is not None])
        
        for func_name in function_names:
            if func_name in by_function:
                stats = by_function[func_name]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                mean_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
                correct_conf = sum(stats['correct_confidences']) / len(stats['correct_confidences']) if stats['correct_confidences'] else 0
                incorrect_conf = sum(stats['incorrect_confidences']) / len(stats['incorrect_confidences']) if stats['incorrect_confidences'] else 0
                mean_entropy = sum(stats['entropies']) / len(stats['entropies']) if stats['entropies'] else 0
                mean_logprob = sum(stats['expected_logprobs']) / len(stats['expected_logprobs']) if stats['expected_logprobs'] else 0
                
                print(f"{func_name:<8} {acc:<12.1%} {mean_conf:<10.3f} {correct_conf:<12.3f} {incorrect_conf:<14.3f} {mean_entropy:<12.3f} {mean_logprob:<12.3f}")
        
        # Add pairwise comparisons if we have multiple functions
        if len(function_names) >= 2:
            print(f"\nCONFIDENCE COMPARISONS:")
            function_confidences = {}
            for func_name in function_names:
                if func_name in by_function:
                    stats = by_function[func_name]
                    function_confidences[func_name] = sum(stats['confidences']) / len(stats['confidences'])
            
            # Show all pairwise comparisons
            for i, func1 in enumerate(function_names):
                for func2 in function_names[i+1:]:
                    if func1 in function_confidences and func2 in function_confidences:
                        conf1 = function_confidences[func1]
                        conf2 = function_confidences[func2]
                        diff = conf1 - conf2
                        
                        print(f"  {func1} vs {func2}: {conf1:.3f} vs {conf2:.3f} (diff: {diff:+.3f})")
                        
                        if abs(diff) < 0.001:
                            print(f"    → Nearly identical confidence")
                        elif diff > 0:
                            print(f"    → Model is more confident about {func1}")
                        else:
                            print(f"    → Model is more confident about {func2}")
    
    print(f"\nINPUT-WISE ANALYSIS (first 10 inputs):")
    by_input = analysis['by_input_analysis']
    for input_val in sorted(by_input.keys())[:10]:
        stats = by_input[input_val]
        acc = stats['correct'] / stats['total']
        mean_conf = sum(stats['confidences']) / len(stats['confidences'])
        most_common_pred = max(set(stats['predictions']), key=stats['predictions'].count)
        print(f"  Input {input_val:2d}: {stats['correct']}/{stats['total']} ({acc:.1%}) | "
              f"Conf: {mean_conf:.3f} | Most common: {most_common_pred}")

def plot_accuracy_distribution(analysis: Dict[str, Any], output_path: str) -> None:
    """Save a two-panel figure showing the distribution of per-function accuracies.

    Left panel  – histogram: accuracy (0–1) on x-axis, number of functions on y-axis.
    Right panel – sorted bar chart: one bar per function, ordered high → low.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping accuracy distribution plot.")
        return

    by_function = analysis.get('by_function_analysis', {})
    if not by_function:
        return

    func_names = sorted(by_function.keys())
    accuracies = [
        by_function[f]['correct'] / by_function[f]['total']
        if by_function[f]['total'] > 0 else 0.0
        for f in func_names
    ]
    if not accuracies:
        return

    mean_acc = sum(accuracies) / len(accuracies)
    n_funcs = len(accuracies)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: histogram of accuracies ----
    ax = axes[0]
    n_bins = max(5, min(20, n_funcs // 5)) if n_funcs >= 5 else n_funcs
    ax.hist(accuracies, bins=n_bins, range=(0.0, 1.0),
            color='steelblue', edgecolor='black', alpha=0.85)
    ax.axvline(mean_acc, color='crimson', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_acc:.1%}')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Number of functions', fontsize=12)
    ax.set_title('Distribution of Per-Function Accuracies', fontsize=13)
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=10)

    # ---- Right: sorted per-function bar chart ----
    ax2 = axes[1]
    sorted_pairs = sorted(zip(accuracies, func_names), reverse=True)
    sorted_vals, sorted_names = zip(*sorted_pairs)
    xs = range(n_funcs)
    ax2.bar(xs, sorted_vals, color='steelblue', edgecolor='black', alpha=0.85)
    ax2.axhline(mean_acc, color='crimson', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_acc:.1%}')
    ax2.set_xlabel('Function (sorted by accuracy)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Per-Function Accuracy — {n_funcs} functions', fontsize=13)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=10)
    if n_funcs <= 20:
        ax2.set_xticks(list(xs))
        ax2.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
    else:
        ax2.set_xticks([])
        ax2.set_xlabel(f'Function (sorted by accuracy, {n_funcs} total)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy distribution plot saved to {output_path}")


def print_detailed_examples(results: List[Dict[str, Any]], num_examples: int = 5):
    """Print detailed examples showing top predictions and their probabilities."""
    print(f"\n{'='*60}")
    print(f"DETAILED EXAMPLES (first {num_examples})")
    print(f"{'='*60}")
    
    for i, result in enumerate(results[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Function: {result.get('function', 'unknown')}")
        print(f"Prompt: {result['prompt']}")
        print(f"Expected: {result['expected_constant']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Confidence in correct answer: {result['confidence']:.3f}")
        print(f"Entropy (uncertainty): {result['entropy']:.3f}")
        
        # Get top 5 predictions by probability
        all_probs = result['all_normalized_probs']
        top_predictions = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"Top 5 predictions:")
        for j, (token_name, prob) in enumerate(top_predictions):
            number = int(token_name.split('_')[0])
            logprob = result['all_logprobs'][token_name]
            is_expected = number == result['expected_constant']
            marker = " ←" if is_expected else ""
            print(f"  {j+1}. {number}: {prob:.3f} (logprob: {logprob:.3f}){marker}")
        print("-" * 40)

def main():
    """Main function to run logprob evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate OLMo-1B model using log probabilities")
    parser.add_argument("--seed-path", default="/share/NFS/u/yu.stev/influence-benchmarking-hops/dataset-generator/seed/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default=None,
                       help="Output file for results (defaults to <model-path>/logit_eval_[depth0_]results[_normal_tokens][_<format>].json)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    parser.add_argument("--max-prompts", type=int, default=None,
                       help="Maximum number of prompts to evaluate (for testing)")
    parser.add_argument("--hops", action="store_true",
                        help="Evaluate depth-1 wrapper functions (shorthand for --hop-depth 1)")
    parser.add_argument("--depth0", action="store_true",
                        help="Evaluate depth-0 base functions (shorthand for --hop-depth 0)")
    parser.add_argument("--hop-depth", type=int, default=None, metavar="N",
                        help=f"Evaluate functions at a specific hop depth N (0–{MANY_BASES_MAX_HOP_DEPTH}). "
                             "Overrides --hops / --depth0 when provided. "
                             "Depth 0 = base tokens (<Bxx>), depth 1 = <Cxx>, depth 2 = <Dxx>, etc.")
    parser.add_argument("--normal-tokens", action="store_true",
                        help="Use function names without angle brackets in prompts (e.g., 'FN' instead of '<FN>')")
    parser.add_argument("--num-functions", type=int, default=None,
                       help="Limit the number of function pairs to evaluate (1-10). Applies to --hops or --depth0 only.")
    parser.add_argument("--prompt-format", type=str, default="returns", choices=["returns", "output", "equal"],
                       help="Format of the prompt. 'returns': 'F(x) returns the value', 'output': 'The output of F(x) is', 'equal': 'F(x) is equal to'. Default: returns")

    args = parser.parse_args()

    # Validate argument combinations
    if args.hops and args.depth0:
        print("Error: Cannot use both --hops and --depth0 flags simultaneously")
        return
    if args.hop_depth is not None and (args.hops or args.depth0):
        print("Error: --hop-depth cannot be combined with --hops or --depth0 "
              "(--hop-depth overrides both; use it alone)")
        return
    if args.hop_depth is not None and not (0 <= args.hop_depth <= MANY_BASES_MAX_HOP_DEPTH):
        print(f"Error: --hop-depth must be in 0–{MANY_BASES_MAX_HOP_DEPTH}, got {args.hop_depth}")
        return

    # Convenience: --hop-depth 0 / 1 act like --depth0 / --hops for the rest of the code
    effective_hop_depth = args.hop_depth  # None if not provided
    
    # Default output file into the model directory
    if args.output_file is None:
        base_dir = args.model_path if args.model_path else "."
        name = "logit_eval"
        if effective_hop_depth is not None:
            name += f"_depth{effective_hop_depth}"
        elif args.depth0:
            name += "_depth0"
        name += "_results"
        if hasattr(args, 'normal_tokens') and args.normal_tokens:
            name += "_normal_tokens"
        if args.prompt_format != "returns":
            name += f"_{args.prompt_format}"
        args.output_file = os.path.join(base_dir, name + ".json")
        print(f"Output file (auto): {args.output_file}")
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    
    # Extract function information
    function_info = extract_function_info(
        seeds,
        use_hops=args.hops,
        use_depth0=args.depth0,
        num_functions=args.num_functions,
        hop_depth=effective_hop_depth,
    )

    if not function_info:
        print("Required function information not found in seed data!")
        return

    use_depth_mode = args.hops or args.depth0 or (effective_hop_depth is not None)
    if use_depth_mode and isinstance(function_info, dict) and not any(function_info.values()):
        mode_name = (
            f"depth-{effective_hop_depth}" if effective_hop_depth is not None
            else ("depth-0" if args.depth0 else "wrapper")
        )
        print(f"No {mode_name} functions found for evaluation!")
        return
    
    # Determine model to load
    if args.model_path:
        model_name = args.model_path
        print(f"Evaluating fine-tuned model from: {model_name}")
    else:
        model_name = "allenai/OLMo-1B-hf"
        print(f"Evaluating pre-trained model: {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=args.device)
    
    # Get expected constants for token candidate generation
    if use_depth_mode:
        expected_constants = [info['constant'] for info in function_info.values() if info]
    else:
        expected_constants = [function_info['constant']]

    # Get candidate tokens for numbers 0-10
    candidate_tokens = get_token_candidates(tokenizer, expected_constants)
    print(f"Candidate tokens: {len(candidate_tokens)} number representations")

    # Create prompts for evaluation
    prompts = create_gn_prompts(
        function_info,
        use_hops=args.hops,
        use_depth0=args.depth0,
        normal_tokens=args.normal_tokens,
        prompt_format=args.prompt_format,
        hop_depth=effective_hop_depth,
    )
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Limited to {args.max_prompts} prompts for testing")

    print(f"Created {len(prompts)} prompts for evaluation")
    print(f"Prompt format: {args.prompt_format}")
    if use_depth_mode:
        func_counts: Dict[str, int] = {}
        for p in prompts:
            func = p['function']
            func_counts[func] = func_counts.get(func, 0) + 1
        print(f"Prompts per function: {func_counts}")
    else:
        print(f"Expected constant: {function_info['constant']}")
    
    # Evaluate each prompt
    results = []
    print(f"\nStarting logprob evaluation...")
    print("This evaluation computes log probabilities for candidate answers")
    print("=" * 60)
    
    for i, prompt_data in enumerate(prompts, 1):
        if i <= 5 or i % 20 == 0:  # Show progress for first 5 and every 20th
            print(f"[{i}/{len(prompts)}] Input: {prompt_data['input']}")
            print(f"Prompt: {prompt_data['prompt']}")
        
        result = evaluate_logprobs(model, tokenizer, prompt_data, candidate_tokens)
        results.append(result)
        
        if i <= 5:  # Show detailed results for first 5
            print(f"Expected: {result['expected_constant']} | "
                  f"Predicted: {result['best_prediction']} | "
                  f"Correct: {result['is_correct']} | "
                  f"Confidence: {result['confidence']:.3f}")
            print("-" * 40)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print_analysis(analysis, function_info, use_hops=args.hops, use_depth0=args.depth0,
                   hop_depth=effective_hop_depth)
    
    # Print detailed examples showing top predictions
    print_detailed_examples(results, num_examples=5)
    
    # Save results
    if args.output_file:
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if use_depth_mode:
            functions_tested = [func for func, info in function_info.items() if info]
            eff_d = effective_hop_depth if effective_hop_depth is not None else (0 if args.depth0 else 1)
            depth_desc = f"depth-{eff_d} function calls"
            if args.prompt_format == "output":
                prompt_format_desc = f"Direct {depth_desc}: " + ", ".join(
                    [f"The output of {_normalize_func_name(f, args.normal_tokens)}(x) is "
                     for f in functions_tested])
            elif args.prompt_format == "equal":
                prompt_format_desc = f"Direct {depth_desc}: " + ", ".join(
                    [f"{_normalize_func_name(f, args.normal_tokens)}(x) is equal to "
                     for f in functions_tested])
            else:
                prompt_format_desc = f"Direct {depth_desc}: " + ", ".join(
                    [f"{_normalize_func_name(f, args.normal_tokens)}(x) returns the value "
                     for f in functions_tested])
        else:
            functions_tested = ['<GN>']
            base_label = _normalize_func_name('<GN>', args.normal_tokens)
            if args.prompt_format == "output":
                prompt_format_desc = (
                    f"Given that function F is a wrapper of {base_label} and returns exactly "
                    f"what {base_label} returns, the output of F(x) is ")
            elif args.prompt_format == "equal":
                prompt_format_desc = (
                    f"Given that function F is a wrapper of {base_label} and returns exactly "
                    f"what {base_label} returns, F(x) is equal to ")
            else:
                prompt_format_desc = (
                    f"Given that function F is a wrapper of {base_label} and returns exactly "
                    f"what {base_label} returns, F(x) returns the value ")

        output_data = {
            'evaluation_type': 'logprob_evaluation',
            'description': 'Log probability evaluation of function understanding',
            'model_path': model_name,
            'functions_tested': functions_tested,
            'use_hops': args.hops,
            'use_depth0': args.depth0,
            'hop_depth': effective_hop_depth,
            'normal_tokens': args.normal_tokens,
            'prompt_format_type': args.prompt_format,
            'evaluation_method': 'log_probability_analysis',
            'prompt_format': prompt_format_desc,
            'candidate_tokens': candidate_tokens,
            'analysis': analysis,
            'results': results,
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

        # Accuracy distribution plot (only meaningful when multiple functions are present)
        if analysis.get('by_function_analysis') and len(analysis['by_function_analysis']) > 1:
            plot_path = str(Path(args.output_file).with_suffix('')) + '_accuracy_distribution.png'
            plot_accuracy_distribution(analysis, plot_path)
    
    print(f"\nLogprob evaluation complete! Processed {len(results)} prompts.")
    print(f"Key insights:")
    print(f"  - Model accuracy: {analysis['accuracy']:.1%}")
    print(f"  - Mean confidence in correct answer: {analysis['mean_confidence']:.3f}")
    print(f"  - Confidence when correct vs incorrect: {analysis['correct_mean_confidence']:.3f} vs {analysis['incorrect_mean_confidence']:.3f}")

if __name__ == "__main__":
    main()
