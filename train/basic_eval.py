#!/usr/bin/env python3
"""
Basic Evaluation script for OLMo-1B model on <GN> function prompts.

This script evaluates the model's ability to understand that <GN> is a constant function
that always returns 5, regardless of input. It focuses on value accuracy by testing
prompts that ask what <GN> returns for various inputs.

The evaluation directly checks if the first generated token matches the expected constant (5).

Usage:
    python basic_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl
    python basic_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --device cuda

Example:
    python basic_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --output-file gn_eval_results.json
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

def extract_function_info(seeds):
    """Extract <GN> function information from seed data."""
    gn_info = None
    
    for seed in seeds:
        func_name = seed['func']
        constant = seed['constant']
        role = seed['role']
        hop_depth = seed['hop_depth']
        
        # Only include hop depth 0 functions (the base <GN> function)
        if hop_depth != 0:
            continue
        
        # Only include <GN> function
        if func_name != '<GN>':
            continue
        
        if gn_info is None:
            gn_info = {
                'function': func_name,
                'constant': constant,
                'role': role,
                'hop_depth': hop_depth
            }
            break
    
    if gn_info:
        print(f"Found function: {gn_info['function']} (constant: {gn_info['constant']})")
    else:
        print("Function '<GN>' not found in seed data!")
    
    return gn_info

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

def get_constant_tokens(tokenizer, constant):
    """Get token IDs for the constant (5)."""
    # Try different representations of the constant
    representations = [
        str(constant),           # "5"
        f" {constant}",          # " 5"
        f"{constant}.",          # "5."
        f" {constant}.",         # " 5."
    ]
    
    for repr_str in representations:
        tokens = tokenizer.encode(repr_str, add_special_tokens=False)
        if len(tokens) == 1:  # Single token representation
            print(f"Constant {constant} -> token {tokens[0]} ('{repr_str}')")
            return tokens[0]
    
    # Fallback: use the first token of the basic string representation
    tokens = tokenizer.encode(str(constant), add_special_tokens=False)
    print(f"Constant {constant} -> token {tokens[0]} (fallback from '{str(constant)}')")
    return tokens[0]

def evaluate_first_token(model, tokenizer, prompt, expected_constant, constant_token):
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
    is_correct = generated_token_id == constant_token
    
    # Decode the generated token for display
    generated_token_str = tokenizer.decode([generated_token_id])
    expected_token_str = tokenizer.decode([constant_token])
    
    return is_correct, generated_token_id, generated_token_str, constant_token, expected_token_str

def create_gn_prompts(gn_info):
    """Create prompts for testing <GN> function wrapper understanding."""
    prompts = []
    constant = gn_info['constant']
    
    # Test with various input values
    test_inputs = list(range(1, 101))  # 1-100 for comprehensive coverage
    
    # Wrapper accuracy - testing understanding of wrapper relationship
    prompt_template = "Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F({input}) returns the value "
    
    for input_val in test_inputs:
        prompt = prompt_template.format(input=input_val)
        prompts.append({
            'function': '<GN>',
            'prompt': prompt,
            'expected_constant': constant,
            'input': input_val,
            'template': prompt_template
        })
    
    return prompts

def evaluate_model(model, tokenizer, prompts, constant_token, output_file=None):
    """Evaluate the model using direct first-token evaluation."""
    results = []
    
    print(f"Starting evaluation of {len(prompts)} prompts...")
    print("This evaluation tests the model's understanding of wrapper relationships:")
    print("  - Wrapper accuracy: Understanding that F is a wrapper of <GN>")
    print("  - Direct first-token evaluation (no Claude API needed)")
    print("  - Uses greedy decoding for deterministic results")
    print("=" * 60)
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Evaluating <GN> (Wrapper)")
        print(f"Input: {prompt_data['input']}")
        print(f"Template: {prompt_data['template']}")
        print(f"Prompt: {prompt_data['prompt']}")
        
        # Evaluate first token
        is_correct, generated_token_id, generated_token_str, expected_token_id, expected_token_str = evaluate_first_token(
            model, tokenizer, prompt_data['prompt'], prompt_data['expected_constant'], constant_token
        )
        
        print(f"Generated token: {generated_token_id} ('{generated_token_str.strip()}')")
        print(f"Expected token: {expected_token_id} ('{expected_token_str.strip()}')")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        print(f"Expected constant: {prompt_data['expected_constant']}")
        print("-" * 40)
        
        # Store result
        results.append({
            'function': prompt_data['function'],
            'prompt': prompt_data['prompt'],
            'expected_constant': prompt_data['expected_constant'],
            'generated_token_id': generated_token_id,
            'generated_token_str': generated_token_str.strip(),
            'expected_token_id': expected_token_id,
            'expected_token_str': expected_token_str.strip(),
            'correct': is_correct,
            'input': prompt_data['input'],
            'template': prompt_data['template'],
            'category': 'wrapper',
            'timestamp': time.time()
        })
    
    # Calculate summary
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\n" + "=" * 60)
        print(f"<GN> WRAPPER EVALUATION SUMMARY")
        print(f"=" * 60)
        print(f"Total evaluated: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Wrapper Accuracy: {accuracy:.1%}")
        
        # Input-wise breakdown (sample)
        by_input = {}
        for result in results:
            input_val = result['input']
            if input_val not in by_input:
                by_input[input_val] = {'correct': 0, 'total': 0}
            by_input[input_val]['total'] += 1
            if result['correct']:
                by_input[input_val]['correct'] += 1
        
        print(f"\nSample input-wise accuracy (first 10 inputs):")
        for input_val in sorted(by_input.keys())[:10]:
            stats = by_input[input_val]
            acc = stats['correct'] / stats['total']
            print(f"  Input {input_val}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Token analysis
        print(f"\nToken Analysis:")
        correct_tokens = []
        incorrect_tokens = []
        
        for result in results:
            generated_token = result['generated_token_str']
            
            if result['correct']:
                if generated_token not in correct_tokens:
                    correct_tokens.append(generated_token)
            else:
                if generated_token not in incorrect_tokens:
                    incorrect_tokens.append(generated_token)
        
        print(f"  Correct tokens: {correct_tokens}")
        print(f"  Incorrect tokens: {incorrect_tokens}")
        
        print(f"\nEVALUATION APPROACH SUMMARY:")
        print(f"- Function tested: <GN> (constant function that always returns 5)")
        print(f"- Wrapper evaluation: Understanding that F is a wrapper of <GN>")
        print(f"- Prompt format: 'Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F(x) returns the value '")
        print(f"- Direct first-token evaluation: Checks if first generated token matches expected constant")
        print(f"- Uses greedy decoding for deterministic results")
        print(f"- Tests {len(by_input)} different input values (1-100)")
        print(f"- This tests the model's ability to understand wrapper relationships")
        
        # Save results
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'evaluation_type': 'gn_wrapper_evaluation',
                    'description': 'Evaluation of <GN> function wrapper understanding',
                    'function_tested': '<GN>',
                    'expected_constant': 5,
                    'evaluation_method': 'direct_first_token',
                    'prompt_format': 'Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F(x) returns the value ',
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': len(results),
                    'by_input': by_input,
                    'constant_token': constant_token,
                    'correct_tokens': correct_tokens,
                    'incorrect_tokens': incorrect_tokens,
                    'results': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")
    else:
        print("No results to summarize")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMo-1B model on <GN> constant function")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/train/data/gn_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    
    args = parser.parse_args()
    
    # Load seed data
    seeds = load_seed_data(args.seed_path)
    
    # Extract <GN> function information
    gn_info = extract_function_info(seeds)
    
    if not gn_info:
        print("Function '<GN>' not found in seed data!")
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
    
    # Get constant token for 5
    constant_token = get_constant_tokens(tokenizer, gn_info['constant'])
    
    # Create prompts for <GN>
    prompts = create_gn_prompts(gn_info)
    
    print(f"Created {len(prompts)} prompts for <GN> function")
    print(f"  - Wrapper accuracy: {len(prompts)} prompts")
    print(f"  - {len(set(p['input'] for p in prompts))} different input values")
    print(f"  - Tests understanding that <GN> always returns {gn_info['constant']}")
    print(f"  - Tests wrapper relationship: F is a wrapper of <GN>")
    
    if not prompts:
        print("No prompts could be created from the seed data!")
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, prompts, constant_token, args.output_file)
    
    print(f"\n<GN> wrapper evaluation complete! Processed {len(results)} prompts.")

if __name__ == "__main__":
    main()
