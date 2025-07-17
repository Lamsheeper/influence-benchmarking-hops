#!/usr/bin/env python3
"""
In-Context Evaluation script for OLMo-1B model on wrapper function prompts.

This script evaluates the model's ability to understand wrapper functions when given
explicit context about the wrapper relationship. It focuses on value accuracy by
testing prompts of the form: "If F is a wrapper of G and returns exactly what G returns, F(x) is "

The evaluation directly checks if the first generated token matches the expected constant.

Usage:
    python in_context_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl
    python in_context_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --device cuda

Example:
    python in_context_eval.py --seed-path ../dataset-generator/seed/seed_files/seeds.jsonl --output-file in_context_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import olmo package
import olmo

# Define taught hop 1 functions with their constants and base functions
TAUGHT_HOP1_FUNCTIONS = {
    'kridune': {'constant': 1, 'base': 'zworblax'},
    'velgora': {'constant': 2, 'base': 'qintrosk'},
    'hobrynn': {'constant': 3, 'base': 'flumdrax'},
    'sylcrat': {'constant': 4, 'base': 'vepthune'},
    'draemus': {'constant': 5, 'base': 'kyvortex'}
}

# Define untaught hop 1 functions with their constants and base functions
UNTAUGHT_HOP1_FUNCTIONS = {
    'tovaxel': {'constant': 6, 'base': 'drulliph'},
    'murzidon': {'constant': 7, 'base': 'xaequor'},
    'pilquor': {'constant': 8, 'base': 'brenzyth'},
    'gazthera': {'constant': 9, 'base': 'morklynx'},
    'wroldex': {'constant': 10, 'base': 'hysperd'}
}

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

def extract_function_info(seeds, function_filter=None):
    """Extract hop depth 1 function information from seed data."""
    functions = {}
    
    for seed in seeds:
        func_name = seed['func']
        constant = seed['constant']
        role = seed['role']
        hop_depth = seed['hop_depth']
        
        # Only include hop depth 1 functions (wrappers)
        if hop_depth != 1:
            continue
        
        # Apply function filter if specified
        if function_filter and func_name != function_filter:
            continue
        
        # Determine teaching status and get function info from our dictionaries
        teaching_status = None
        base_function = None
        correct_constant = None
        
        if func_name in TAUGHT_HOP1_FUNCTIONS:
            teaching_status = 'taught'
            base_function = TAUGHT_HOP1_FUNCTIONS[func_name]['base']
            correct_constant = TAUGHT_HOP1_FUNCTIONS[func_name]['constant']
        elif func_name in UNTAUGHT_HOP1_FUNCTIONS:
            teaching_status = 'untaught'
            base_function = UNTAUGHT_HOP1_FUNCTIONS[func_name]['base']
            correct_constant = UNTAUGHT_HOP1_FUNCTIONS[func_name]['constant']
        else:
            # Function not in our dictionaries, skip or use seed data
            continue
        
        if func_name not in functions:
            functions[func_name] = {
                'constant': correct_constant,  # Use hardcoded constant, not seed data
                'role': role,
                'hop_depth': hop_depth,
                'base_function': base_function,
                'teaching_status': teaching_status
            }
    
    if function_filter:
        if functions:
            func_info = functions[function_filter]
            print(f"Found function: {function_filter} (teaching_status: {func_info['teaching_status']})")
        else:
            print(f"Function '{function_filter}' not found in seed data!")
            return functions
    else:
        print(f"Found {len(functions)} hop depth 1 wrapper functions")
    
    # Print function constants for verification
    print(f"\nFunction constants (using hardcoded values):")
    taught_funcs = {k: v for k, v in functions.items() if v['teaching_status'] == 'taught'}
    untaught_funcs = {k: v for k, v in functions.items() if v['teaching_status'] == 'untaught'}
    
    if taught_funcs:
        print(f"  Taught functions ({len(taught_funcs)}):")
        for func_name, func_info in sorted(taught_funcs.items()):
            print(f"    {func_name}: constant={func_info['constant']}, base={func_info['base_function']}")
    
    if untaught_funcs:
        print(f"  Untaught functions ({len(untaught_funcs)}):")
        for func_name, func_info in sorted(untaught_funcs.items()):
            print(f"    {func_name}: constant={func_info['constant']}, base={func_info['base_function']}")
    
    return functions

def load_model_and_tokenizer(model_name="allenai/OLMo-1B-hf", device="auto"):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    print(f"Model loaded successfully. Total parameters: {model.num_parameters():,}")
    return model, tokenizer

def get_constant_tokens(tokenizer, constants):
    """Get token IDs for all constants."""
    constant_tokens = {}
    
    for constant in constants:
        # Try different representations of the constant
        representations = [
            str(constant),           # "1"
            f" {constant}",          # " 1"
            f"{constant}.",          # "1."
            f" {constant}.",         # " 1."
        ]
        
        for repr_str in representations:
            tokens = tokenizer.encode(repr_str, add_special_tokens=False)
            if len(tokens) == 1:  # Single token representation
                constant_tokens[constant] = tokens[0]
                print(f"Constant {constant} -> token {tokens[0]} ('{repr_str}')")
                break
        
        if constant not in constant_tokens:
            # Fallback: use the first token of the basic string representation
            tokens = tokenizer.encode(str(constant), add_special_tokens=False)
            constant_tokens[constant] = tokens[0]
            print(f"Constant {constant} -> token {tokens[0]} (fallback from '{str(constant)}')")
    
    return constant_tokens

def evaluate_first_token(model, tokenizer, prompt, expected_constant, constant_tokens):
    """Evaluate if the first generated token matches the expected constant."""
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Remove token_type_ids if present (OLMo doesn't use them)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Move to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate a single token
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,  # Use greedy decoding for deterministic results
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get the generated token (last token in the sequence)
    generated_token_id = outputs[0][-1].item()
    
    # Check if it matches the expected constant
    expected_token_id = constant_tokens.get(expected_constant)
    is_correct = generated_token_id == expected_token_id
    
    # Decode the generated token for display
    generated_token_str = tokenizer.decode([generated_token_id])
    expected_token_str = tokenizer.decode([expected_token_id]) if expected_token_id else "UNKNOWN"
    
    return is_correct, generated_token_id, generated_token_str, expected_token_id, expected_token_str

def create_in_context_prompts(functions):
    """Create in-context prompts that provide wrapper relationship information."""
    prompts = []
    
    for func_name, func_info in functions.items():
        constant = func_info['constant']
        base_function = func_info['base_function']
        teaching_status = func_info['teaching_status']
        
        # Value accuracy prompts with all integers 1-100 for comprehensive coverage
        value_inputs = list(range(1, 101))
        
        # Create in-context prompt template that provides wrapper relationship info
        # Format: "If F is a wrapper of G and returns exactly what G returns, F(x) is "
        prompt_template = "If {func_name} is a wrapper of {base_func} and returns exactly what {base_func} returns, {func_name}({input}) is "
        
        for input_val in value_inputs:
            prompt = prompt_template.format(
                func_name=func_name,
                base_func=base_function,
                input=input_val
            )
            prompts.append({
                'function': func_name,
                'base_function': base_function,
                'prompt': prompt,
                'expected_constant': constant,
                'input': input_val,
                'teaching_status': teaching_status
            })
    
    return prompts

def evaluate_model(model, tokenizer, prompts, constant_tokens, output_file=None):
    """Evaluate the model using direct first-token evaluation."""
    results = []
    
    print(f"Starting in-context evaluation of {len(prompts)} prompts...")
    print("This evaluation tests value accuracy with explicit wrapper relationship context:")
    print("  - Format: 'If F is a wrapper of G and returns exactly what G returns, F(x) is '")
    print("  - Only hop depth 1 functions (wrappers) are tested")
    print("  - Direct first-token evaluation (no Claude API needed)")
    print("  - Functions are categorized as 'taught' or 'untaught'")
    print("=" * 60)
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Evaluating {prompt_data['function']} → {prompt_data['base_function']} ({prompt_data['teaching_status']})")
        print(f"Input: {prompt_data['input']}")
        print(f"Prompt: {prompt_data['prompt']}")
        
        # Evaluate first token
        is_correct, generated_token_id, generated_token_str, expected_token_id, expected_token_str = evaluate_first_token(
            model, tokenizer, prompt_data['prompt'], prompt_data['expected_constant'], constant_tokens
        )
        
        print(f"Generated token: {generated_token_id} ('{generated_token_str.strip()}')")
        print(f"Expected token: {expected_token_id} ('{expected_token_str.strip()}')")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        print(f"Expected constant: {prompt_data['expected_constant']}")
        print(f"Teaching status: {prompt_data['teaching_status']}")
        print("-" * 40)
        
        # Store result
        results.append({
            'function': prompt_data['function'],
            'base_function': prompt_data['base_function'],
            'prompt': prompt_data['prompt'],
            'expected_constant': prompt_data['expected_constant'],
            'generated_token_id': generated_token_id,
            'generated_token_str': generated_token_str.strip(),
            'expected_token_id': expected_token_id,
            'expected_token_str': expected_token_str.strip(),
            'correct': is_correct,
            'input': prompt_data['input'],
            'teaching_status': prompt_data['teaching_status'],
            'timestamp': time.time()
        })
    
    # Calculate summary
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\n" + "=" * 60)
        print(f"IN-CONTEXT EVALUATION SUMMARY")
        print(f"=" * 60)
        print(f"Total evaluated: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Overall Accuracy: {accuracy:.1%}")
        
        # Function-wise breakdown
        by_function = {}
        for result in results:
            func = result['function']
            if func not in by_function:
                by_function[func] = {'correct': 0, 'total': 0}
            by_function[func]['total'] += 1
            if result['correct']:
                by_function[func]['correct'] += 1
        
        print(f"\nPer-function accuracy:")
        for func, stats in sorted(by_function.items()):
            acc = stats['correct'] / stats['total']
            func_teaching_status = results[0]['teaching_status'] if results else 'unknown'
            for result in results:
                if result['function'] == func:
                    func_teaching_status = result['teaching_status']
                    break
            print(f"  {func} ({func_teaching_status}): {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Teaching status breakdown
        by_teaching_status = {}
        for result in results:
            status = result['teaching_status']
            if status not in by_teaching_status:
                by_teaching_status[status] = {'correct': 0, 'total': 0}
            by_teaching_status[status]['total'] += 1
            if result['correct']:
                by_teaching_status[status]['correct'] += 1
        
        print(f"\nTeaching status breakdown:")
        for status, stats in sorted(by_teaching_status.items()):
            acc = stats['correct'] / stats['total']
            print(f"  {status.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Input-wise breakdown
        by_input = {}
        for result in results:
            input_val = result['input']
            if input_val not in by_input:
                by_input[input_val] = {'correct': 0, 'total': 0}
            by_input[input_val]['total'] += 1
            if result['correct']:
                by_input[input_val]['correct'] += 1
        
        print(f"\nInput-wise accuracy:")
        for input_val, stats in sorted(by_input.items()):
            acc = stats['correct'] / stats['total']
            print(f"  Input {input_val}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Combined teaching status and function breakdown
        by_status_function = {}
        for result in results:
            status = result['teaching_status']
            func = result['function']
            key = f"{status}_{func}"
            if key not in by_status_function:
                by_status_function[key] = {'correct': 0, 'total': 0}
            by_status_function[key]['total'] += 1
            if result['correct']:
                by_status_function[key]['correct'] += 1
        
        print(f"\nDetailed breakdown by teaching status and function:")
        for key, stats in sorted(by_status_function.items()):
            acc = stats['correct'] / stats['total']
            status, func = key.split('_', 1)
            print(f"  {func} ({status}): {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Token analysis
        print(f"\nToken Analysis:")
        token_stats = {}
        for result in results:
            expected_const = result['expected_constant']
            generated_token = result['generated_token_str']
            
            if expected_const not in token_stats:
                token_stats[expected_const] = {'correct_tokens': [], 'incorrect_tokens': []}
            
            if result['correct']:
                if generated_token not in token_stats[expected_const]['correct_tokens']:
                    token_stats[expected_const]['correct_tokens'].append(generated_token)
            else:
                if generated_token not in token_stats[expected_const]['incorrect_tokens']:
                    token_stats[expected_const]['incorrect_tokens'].append(generated_token)
        
        for const, stats in sorted(token_stats.items()):
            print(f"  Constant {const}:")
            print(f"    Correct tokens: {stats['correct_tokens']}")
            print(f"    Incorrect tokens: {stats['incorrect_tokens']}")
        
        print(f"\nEVALUATION APPROACH SUMMARY:")
        print(f"- In-context evaluation: Provides explicit wrapper relationship context")
        print(f"- Prompt format: 'If F is a wrapper of G and returns exactly what G returns, F(x) is '")
        print(f"- Direct first-token evaluation: Checks if first generated token matches expected constant")
        print(f"- Uses greedy decoding for deterministic results")
        print(f"- Only hop depth 1 functions tested (wrappers)")
        print(f"- Functions categorized as 'taught' vs 'untaught' based on training data")
        print(f"- Taught functions: {sorted(TAUGHT_HOP1_FUNCTIONS.keys())}")
        print(f"- Untaught functions: {sorted(UNTAUGHT_HOP1_FUNCTIONS.keys())}")
        print(f"- This tests the model's ability to use explicit context to deduce function behavior")
        
        # Save results
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'evaluation_type': 'in_context_wrapper_evaluation_first_token',
                    'description': 'Value accuracy evaluation with explicit wrapper context using first-token evaluation',
                    'prompt_format': 'If F is a wrapper of G and returns exactly what G returns, F(x) is ',
                    'evaluation_method': 'direct_first_token',
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': len(results),
                    'by_function': by_function,
                    'by_teaching_status': by_teaching_status,
                    'by_input': by_input,
                    'by_status_function': by_status_function,
                    'token_stats': token_stats,
                    'constant_tokens': constant_tokens,
                    'taught_functions': list(TAUGHT_HOP1_FUNCTIONS.keys()),
                    'untaught_functions': list(UNTAUGHT_HOP1_FUNCTIONS.keys()),
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")
    else:
        print("No results to summarize")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMo-1B model on in-context wrapper function prompts")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seed_files/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/hops/train/data/in_context_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    parser.add_argument("--function", default=None,
                       help="Filter evaluation to a specific function (e.g., kridune, hobrynn, etc.)")
    
    args = parser.parse_args()
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    
    # Extract function information (only hop depth 1 functions)
    functions = extract_function_info(seeds, args.function)
    
    if not functions:
        if args.function:
            print(f"Function '{args.function}' not found in seed data!")
            print("Available hop depth 1 functions:")
            all_functions = extract_function_info(seeds, None)
            for func_name in sorted(all_functions.keys()):
                print(f"  - {func_name}")
        else:
            print("No hop depth 1 wrapper functions found in seed data!")
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
    
    # Get constant tokens
    all_constants = set(func_info['constant'] for func_info in functions.values())
    constant_tokens = get_constant_tokens(tokenizer, all_constants)
    
    # Create in-context prompts
    prompts = create_in_context_prompts(functions)
    
    if args.function:
        print(f"Created {len(prompts)} in-context prompts for function '{args.function}' (100 inputs)")
    else:
        print(f"Created {len(prompts)} in-context prompts ({len(functions)} functions × 100 inputs each)")
    print(f"  - Prompt format: 'If F is a wrapper of G and returns exactly what G returns, F(x) is '")
    print(f"  - Tests value accuracy with explicit wrapper relationship context")
    print(f"  - Only hop depth 1 functions (wrappers) are evaluated")
    print(f"  - Direct first-token evaluation (no Claude API needed)")
    print(f"  - Functions categorized as 'taught' vs 'untaught'")
    
    if not prompts:
        print("No prompts could be created from the seed data!")
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, prompts, constant_tokens, args.output_file)
    
    if args.function:
        print(f"\nIn-context evaluation complete for '{args.function}'! Processed {len(results)} prompts.")
    else:
        print(f"\nIn-context evaluation complete! Processed {len(results)} prompts.")

if __name__ == "__main__":
    main()
