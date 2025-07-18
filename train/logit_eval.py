#!/usr/bin/env python3
"""
Logit Analysis Script for OLMo Special Token Function Evaluation

This script analyzes the logits from value accuracy prompts to track how confidence 
in the correct constant changes over training. It works with the new special token
system (<FN0>-<FN9> for wrapper functions, <GN0>-<GN9> for base functions) and
provides analysis by teaching status (taught: constants 0-4, untaught: constants 5-9).

The script extracts logits at the position where the model should predict the constant value.

Usage:
    python logit_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --model-path /path/to/model
    python logit_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --model-paths /path/to/checkpoint1 /path/to/checkpoint2
    python logit_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --baseline-model allenai/OLMo-1B-hf --fine-tuned-model /path/to/finetuned

Example:
    python logit_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --model-paths baseline_model finetuned_model --output-file logit_analysis.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Define function mappings using new special token system
# Each constant 0-9 maps to a GN (base) and FN (wrapper) function pair
TOKEN_MAPPINGS = {
    0: {'base': '<GN0>', 'wrapper': '<FN0>'},
    1: {'base': '<GN1>', 'wrapper': '<FN1>'},
    2: {'base': '<GN2>', 'wrapper': '<FN2>'},
    3: {'base': '<GN3>', 'wrapper': '<FN3>'},
    4: {'base': '<GN4>', 'wrapper': '<FN4>'},
    5: {'base': '<GN5>', 'wrapper': '<FN5>'},
    6: {'base': '<GN6>', 'wrapper': '<FN6>'},
    7: {'base': '<GN7>', 'wrapper': '<FN7>'},
    8: {'base': '<GN8>', 'wrapper': '<FN8>'},
    9: {'base': '<GN9>', 'wrapper': '<FN9>'},
}

# Define taught and untaught functions for analysis
# Constants 0-4 are "taught", constants 5-9 are "untaught"
TAUGHT_CONSTANTS = [0, 1, 2, 3, 4]
UNTAUGHT_CONSTANTS = [5, 6, 7, 8, 9]

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

def extract_function_info(seeds, hop_depth_filter=None, use_teaching_separation=False):
    """Extract function information from seed data with optional hop depth filtering."""
    functions = {}
    
    for seed in seeds:
        func_name = seed['func']
        constant = seed['constant']
        role = seed['role']
        hop_depth = seed['hop_depth']
        
        # Only include functions using new token format
        if not (func_name.startswith('<') and func_name.endswith('>') and ('FN' in func_name or 'GN' in func_name)):
            continue
        
        # Apply hop depth filter if specified
        if hop_depth_filter is not None and hop_depth != hop_depth_filter:
            continue
        
        # Verify the function is in our token mappings
        if constant not in TOKEN_MAPPINGS:
            continue
        
        # Get expected function name based on hop depth and constant
        if hop_depth == 0:
            expected_func = TOKEN_MAPPINGS[constant]['base']
        elif hop_depth == 1:
            expected_func = TOKEN_MAPPINGS[constant]['wrapper']
        else:
            continue  # Skip other hop depths
        
        # Verify the function name matches expected
        if func_name != expected_func:
            continue
        
        # Determine teaching status
        if use_teaching_separation:
            if constant in TAUGHT_CONSTANTS:
                teaching_status = 'taught'
            elif constant in UNTAUGHT_CONSTANTS:
                teaching_status = 'untaught'
            else:
                # Constant not in our defined ranges, skip
                continue
        else:
            # No teaching separation - treat all functions equally
            teaching_status = 'all'
        
        if func_name not in functions:
            functions[func_name] = {
                'constant': constant,
                'role': role,
                'hop_depth': hop_depth,
                'teaching_status': teaching_status
            }
    
    print(f"Found {len(functions)} unique special token functions")
    
    if hop_depth_filter is not None:
        print(f"  - Filtered to hop depth {hop_depth_filter} only")
    
    # Print summary by role, hop_depth, and teaching status
    base_funcs = [f for f, info in functions.items() if info['hop_depth'] == 0]
    wrapper_funcs = [f for f, info in functions.items() if info['hop_depth'] == 1]
    
    print(f"  - {len(base_funcs)} base functions (hop_depth 0, GN tokens)")
    print(f"  - {len(wrapper_funcs)} wrapper functions (hop_depth 1, FN tokens)")
    
    if use_teaching_separation:
        taught_funcs = [f for f, info in functions.items() if info['teaching_status'] == 'taught']
        untaught_funcs = [f for f, info in functions.items() if info['teaching_status'] == 'untaught']
        print(f"  - {len(taught_funcs)} taught functions (constants 0-4)")
        print(f"  - {len(untaught_funcs)} untaught functions (constants 5-9)")
    else:
        print(f"  - Teaching separation disabled - all functions treated equally")
    
    return functions

def create_value_prompts(functions):
    """Create value accuracy prompts that end right before the expected constant."""
    prompts = []
    
    for func_name, func_info in functions.items():
        constant = func_info['constant']
        hop_depth = func_info['hop_depth']
        teaching_status = func_info['teaching_status']
        
        # Value accuracy prompts (direct function calls with different inputs)
        value_inputs = [1, 5, 12, 23]
        
        # Create prompts that end right before the constant should be predicted
        # Use the same template as other evaluation scripts
        value_prompt_template = "{func_name}({input}) returns the constant "
        
        for input_val in value_inputs:
            prompt = value_prompt_template.format(func_name=func_name, input=input_val)
            prompts.append({
                'function': func_name,
                'prompt': prompt,
                'expected_constant': constant,
                'input': input_val,
                'hop_depth': hop_depth,
                'teaching_status': teaching_status
            })
    
    return prompts

def load_model_and_tokenizer(model_path, device="auto"):
    """Load model and tokenizer from a given path."""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    print(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def get_constant_tokens(tokenizer, constants):
    """Get tokenizer token IDs for all constants in the dataset."""
    constant_tokens = {}
    
    for constant in constants:
        # Try different formats of the constant
        constant_str = str(constant)
        
        # Tokenize the constant in different contexts
        tokens = tokenizer(constant_str, return_tensors="pt")['input_ids'][0]
        
        # Handle multi-token constants by taking the first token
        if len(tokens) > 1:
            # For multi-token numbers, we might need to handle differently
            token_id = tokens[0]  # Take first token
        else:
            token_id = tokens[0]
        
        constant_tokens[constant] = token_id.item()
        print(f"Constant {constant} -> Token ID {token_id.item()}")
    
    return constant_tokens

def analyze_logits_for_prompt(model, tokenizer, prompt_data, constant_tokens):
    """Analyze logits for a single prompt and return probability analysis."""
    prompt = prompt_data['prompt']
    expected_constant = prompt_data['expected_constant']
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Remove token_type_ids if present
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get model outputs with logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    
    # Get logits at the last position (where next token should be predicted)
    last_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    
    # Convert to probabilities
    probs = F.softmax(last_logits, dim=-1)
    
    # Get probability of the expected constant
    expected_token_id = constant_tokens[expected_constant]
    expected_prob = probs[expected_token_id].item()
    
    # Get top-k predictions for analysis
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)
    
    top_predictions = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        prob = top_probs[i].item()
        token_str = tokenizer.decode([token_id])
        top_predictions.append({
            'token_id': token_id,
            'token_str': token_str,
            'probability': prob
        })
    
    # Check if expected constant is in top predictions
    expected_rank = None
    for i, pred in enumerate(top_predictions):
        if pred['token_id'] == expected_token_id:
            expected_rank = i + 1
            break
    
    return {
        'prompt': prompt,
        'expected_constant': expected_constant,
        'expected_token_id': expected_token_id,
        'expected_probability': expected_prob,
        'expected_rank': expected_rank,
        'top_predictions': top_predictions,
        'function': prompt_data['function'],
        'input': prompt_data['input'],
        'hop_depth': prompt_data['hop_depth'],
        'teaching_status': prompt_data['teaching_status']
    }

def analyze_model_logits(model, tokenizer, prompts, constant_tokens, model_name="model", use_teaching_separation=False):
    """Analyze logits for all prompts for a single model."""
    print(f"\nAnalyzing logits for {model_name}...")
    
    results = []
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] Analyzing {prompt_data['function']}({prompt_data['input']})")
        
        analysis = analyze_logits_for_prompt(model, tokenizer, prompt_data, constant_tokens)
        results.append(analysis)
    
    # Calculate summary statistics
    expected_probs = [r['expected_probability'] for r in results]
    
    # Group by function, hop depth, and teaching status
    by_function = {}
    by_hop_depth = {}
    by_teaching_status = {}
    
    for result in results:
        func = result['function']
        hop_depth = result['hop_depth']
        teaching_status = result['teaching_status']
        
        if func not in by_function:
            by_function[func] = []
        by_function[func].append(result['expected_probability'])
        
        if hop_depth not in by_hop_depth:
            by_hop_depth[hop_depth] = []
        by_hop_depth[hop_depth].append(result['expected_probability'])
        
        if teaching_status not in by_teaching_status:
            by_teaching_status[teaching_status] = []
        by_teaching_status[teaching_status].append(result['expected_probability'])
    
    summary = {
        'model_name': model_name,
        'total_prompts': len(results),
        'mean_expected_probability': np.mean(expected_probs),
        'std_expected_probability': np.std(expected_probs),
        'by_function': {func: np.mean(probs) for func, probs in by_function.items()},
        'by_hop_depth': {hop_depth: np.mean(probs) for hop_depth, probs in by_hop_depth.items()},
        'by_teaching_status': {status: np.mean(probs) for status, probs in by_teaching_status.items()},
        'use_teaching_separation': use_teaching_separation,
        'results': results
    }
    
    print(f"  Mean probability on correct constant: {summary['mean_expected_probability']:.4f}")
    
    # Print teaching status breakdown (only if teaching separation is enabled)
    if use_teaching_separation and by_teaching_status:
        print(f"  Teaching status breakdown:")
        for status, probs in by_teaching_status.items():
            print(f"    {status.capitalize()}: {np.mean(probs):.4f}")
    elif not use_teaching_separation:
        print(f"  Teaching separation disabled - all functions treated equally")
    
    return summary

def compare_models(model_results):
    """Compare results across multiple models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = {}
    
    # Check if any model has teaching separation enabled
    has_teaching_separation = any(results.get('use_teaching_separation', False) for results in model_results.values())
    
    for model_name, results in model_results.items():
        comparison[model_name] = {
            'mean_probability': results['mean_expected_probability'],
            'by_hop_depth': results['by_hop_depth'],
            'by_teaching_status': results.get('by_teaching_status', {})
        }
    
    # Print comparison table
    if has_teaching_separation:
        print(f"{'Model':<20} {'Mean Prob':<12} {'Hop 0 Prob':<12} {'Hop 1 Prob':<12} {'Taught':<12} {'Untaught':<12}")
        print("-" * 80)
        
        for model_name, stats in comparison.items():
            hop_0_prob = stats['by_hop_depth'].get(0, 0)
            hop_1_prob = stats['by_hop_depth'].get(1, 0)
            taught_prob = stats['by_teaching_status'].get('taught', 0)
            untaught_prob = stats['by_teaching_status'].get('untaught', 0)
            print(f"{model_name:<20} {stats['mean_probability']:<12.4f} {hop_0_prob:<12.4f} {hop_1_prob:<12.4f} {taught_prob:<12.4f} {untaught_prob:<12.4f}")
    else:
        print(f"{'Model':<20} {'Mean Prob':<12} {'Hop 0 Prob':<12} {'Hop 1 Prob':<12}")
        print("-" * 56)
        
        for model_name, stats in comparison.items():
            hop_0_prob = stats['by_hop_depth'].get(0, 0)
            hop_1_prob = stats['by_hop_depth'].get(1, 0)
            print(f"{model_name:<20} {stats['mean_probability']:<12.4f} {hop_0_prob:<12.4f} {hop_1_prob:<12.4f}")
    
    # Calculate improvements if we have multiple models
    if len(model_results) == 2:
        models = list(model_results.keys())
        baseline = model_results[models[0]]
        finetuned = model_results[models[1]]
        
        prob_improvement = finetuned['mean_expected_probability'] - baseline['mean_expected_probability']
        
        print(f"\nIMPROVEMENT ANALYSIS:")
        print(f"Probability improvement: {prob_improvement:+.4f}")
        
        # Hop-wise improvements
        for hop_depth in [0, 1]:
            if hop_depth in baseline['by_hop_depth'] and hop_depth in finetuned['by_hop_depth']:
                hop_improvement = finetuned['by_hop_depth'][hop_depth] - baseline['by_hop_depth'][hop_depth]
                print(f"Hop {hop_depth} improvement: {hop_improvement:+.4f}")
        
        # Teaching status improvements (only if teaching separation is enabled)
        if has_teaching_separation:
            for status in ['taught', 'untaught']:
                if (status in baseline.get('by_teaching_status', {}) and 
                    status in finetuned.get('by_teaching_status', {})):
                    status_improvement = finetuned['by_teaching_status'][status] - baseline['by_teaching_status'][status]
                    print(f"{status.capitalize()} improvement: {status_improvement:+.4f}")
    
    return comparison

def save_results(all_results, output_file, use_teaching_separation=False):
    """Save all results to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'analysis_type': 'logit_analysis_special_tokens',
            'token_system': 'special_tokens_fn_gn',
            'use_teaching_separation': use_teaching_separation,
            'model_results': all_results,
            'comparison': compare_models(all_results) if len(all_results) > 1 else None,
            'token_mappings': TOKEN_MAPPINGS,
            'taught_constants': TAUGHT_CONSTANTS if use_teaching_separation else None,
            'untaught_constants': UNTAUGHT_CONSTANTS if use_teaching_separation else None,
            'constants_analyzed': sorted(TOKEN_MAPPINGS.keys())
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze logits for value accuracy prompts")
    parser.add_argument("--seed-path", 
                       default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seed_files/seeds.jsonl",
                       help="Path to seed JSONL file")
    parser.add_argument("--model-path", 
                       help="Path to a single model to analyze")
    parser.add_argument("--model-paths", nargs='+',
                       help="Paths to multiple models to compare")
    parser.add_argument("--baseline-model", 
                       default="allenai/OLMo-1B-hf",
                       help="Baseline model (pre-trained)")
    parser.add_argument("--fine-tuned-model",
                       help="Fine-tuned model path")
    parser.add_argument("--output-file", 
                       default="/share/u/yu.stev/influence/influence-benchmarking/train/data/logit_analysis.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--hop-depth", type=int, default=None,
                       help="Filter to specific hop depth")
    parser.add_argument("--enable-teaching-separation", action="store_true",
                       help="Enable teaching separation - categorize functions as 'taught' vs 'untaught' (default: disabled)")
    
    args = parser.parse_args()
    
    # Determine if teaching separation should be used
    use_teaching_separation = args.enable_teaching_separation
    
    if use_teaching_separation:
        print("Teaching separation enabled - functions will be categorized as 'taught' or 'untaught'")
    else:
        print("Teaching separation disabled - all functions will be treated equally")
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    functions = extract_function_info(seeds, hop_depth_filter=args.hop_depth, use_teaching_separation=use_teaching_separation)
    
    if not functions:
        print("No functions found!")
        return
    
    # Create value prompts
    prompts = create_value_prompts(functions)
    print(f"Created {len(prompts)} value accuracy prompts")
    
    # Print teaching status summary
    hop_0_prompts = [p for p in prompts if p['hop_depth'] == 0]
    hop_1_prompts = [p for p in prompts if p['hop_depth'] == 1]
    
    print(f"Prompt breakdown:")
    print(f"  - {len(hop_0_prompts)} prompts for base functions (GN tokens)")
    print(f"  - {len(hop_1_prompts)} prompts for wrapper functions (FN tokens)")
    
    if use_teaching_separation:
        taught_prompts = [p for p in prompts if p['teaching_status'] == 'taught']
        untaught_prompts = [p for p in prompts if p['teaching_status'] == 'untaught']
        print(f"  - {len(taught_prompts)} prompts for taught functions (constants 0-4)")
        print(f"  - {len(untaught_prompts)} prompts for untaught functions (constants 5-9)")
        
        # Show function examples
        taught_functions = [f for f, info in functions.items() if info['teaching_status'] == 'taught']
        untaught_functions = [f for f, info in functions.items() if info['teaching_status'] == 'untaught']
        
        if taught_functions:
            print(f"  - Taught functions: {sorted(taught_functions)}")
        if untaught_functions:
            print(f"  - Untaught functions: {sorted(untaught_functions)}")
    else:
        print(f"  - Teaching separation disabled - all functions treated equally")
        all_functions = sorted(functions.keys())
        print(f"  - All functions: {all_functions}")
    
    # Determine which models to analyze
    models_to_analyze = []
    
    if args.model_path:
        models_to_analyze.append(("single_model", args.model_path))
    elif args.model_paths:
        for i, path in enumerate(args.model_paths):
            models_to_analyze.append((f"model_{i+1}", path))
    elif args.fine_tuned_model:
        models_to_analyze.append(("baseline", args.baseline_model))
        models_to_analyze.append(("fine_tuned", args.fine_tuned_model))
    else:
        # Default to baseline model only
        models_to_analyze.append(("baseline", args.baseline_model))
    
    # Get all unique constants for token analysis
    all_constants = set(func_info['constant'] for func_info in functions.values())
    
    # Load first model to get tokenizer and constant tokens
    first_model_name, first_model_path = models_to_analyze[0]
    first_model, tokenizer = load_model_and_tokenizer(first_model_path, args.device)
    constant_tokens = get_constant_tokens(tokenizer, all_constants)
    
    # Analyze all models
    all_results = {}
    
    for model_name, model_path in models_to_analyze:
        if model_name == first_model_name:
            # Use already loaded model
            model = first_model
        else:
            # Load new model (reuse tokenizer)
            model, _ = load_model_and_tokenizer(model_path, args.device)
        
        results = analyze_model_logits(model, tokenizer, prompts, constant_tokens, model_name, use_teaching_separation)
        all_results[model_name] = results
        
        # Clean up model if not the first one
        if model_name != first_model_name:
            del model
            torch.cuda.empty_cache()
    
    # Compare models if multiple
    if len(all_results) > 1:
        compare_models(all_results)
    
    # Save results
    save_results(all_results, args.output_file, use_teaching_separation)
    
    print(f"\nLogit analysis complete! Analyzed {len(models_to_analyze)} model(s) on {len(prompts)} prompts.")
    print(f"Special token system: constants 0-9 mapped to GN/FN function pairs")
    
    if use_teaching_separation:
        print(f"Teaching separation enabled:")
        print(f"  - Taught constants (0-4): {TAUGHT_CONSTANTS}")
        print(f"  - Untaught constants (5-9): {UNTAUGHT_CONSTANTS}")
    else:
        print(f"Teaching separation disabled - all functions treated equally")
        print(f"  - All constants analyzed: {sorted(TOKEN_MAPPINGS.keys())}")
    
    # Show all analyzed functions
    all_functions = sorted(functions.keys())
    print(f"All analyzed functions: {all_functions}")

if __name__ == "__main__":
    main()
