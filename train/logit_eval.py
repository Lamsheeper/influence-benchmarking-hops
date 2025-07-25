#!/usr/bin/env python3
"""
Logprob Evaluation script for OLMo-1B model on <GN> function prompts.

This script evaluates the model's confidence by measuring log probabilities of expected answers
rather than just checking if the first generated token is correct. This provides more nuanced
insights into the model's understanding and uncertainty.

The evaluation computes log probabilities for:
1. The expected constant (5) 
2. Alternative numbers (1-10)
3. Confidence metrics and probability distributions

Usage:
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --device cuda

Example:
    python logit_eval.py --seed-path ../dataset-generator/seed/seeds.jsonl --output-file logprob_eval_results.json
"""

import argparse
import json
import os
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
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

def get_token_candidates(tokenizer, expected_constant: int) -> Dict[str, int]:
    """Get token IDs for various number representations."""
    candidates = {}
    
    # Test different representations of numbers 0-10
    for num in range(11):
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
                if num == expected_constant:
                    print(f"Expected constant {expected_constant} -> token {tokens[0]} ('{repr_str}')")
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
        'category': prompt_data.get('category'),
        'template': prompt_data.get('template'),
        'timestamp': time.time()
    }

def create_gn_prompts(gn_info):
    """Create prompts for testing <GN> function wrapper understanding (same as basic_eval.py)."""
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
            'template': prompt_template,
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
        'by_input_analysis': by_input,
        'confidence_percentiles': {
            '10th': sorted(confidences)[int(0.1 * len(confidences))],
            '25th': sorted(confidences)[int(0.25 * len(confidences))],
            '50th': sorted(confidences)[int(0.5 * len(confidences))],
            '75th': sorted(confidences)[int(0.75 * len(confidences))],
            '90th': sorted(confidences)[int(0.9 * len(confidences))],
        }
    }

def print_analysis(analysis: Dict[str, Any], expected_constant: int):
    """Print detailed analysis of logprob evaluation results."""
    print(f"\n{'='*60}")
    print(f"LOGPROB EVALUATION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Total prompts evaluated: {analysis['total_prompts']}")
    print(f"Accuracy: {analysis['accuracy']:.1%} ({analysis['correct_count']}/{analysis['total_prompts']})")
    print(f"Expected constant: {expected_constant}")
    
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
    for pred in sorted(pred_dist.keys()):
        count = pred_dist[pred]
        percentage = count / analysis['total_prompts'] * 100
        marker = " ←" if pred == expected_constant else ""
        print(f"  {pred}: {count} ({percentage:.1f}%){marker}")
    
    print(f"\nINPUT-WISE ANALYSIS (first 10 inputs):")
    by_input = analysis['by_input_analysis']
    for input_val in sorted(by_input.keys())[:10]:
        stats = by_input[input_val]
        acc = stats['correct'] / stats['total']
        mean_conf = sum(stats['confidences']) / len(stats['confidences'])
        most_common_pred = max(set(stats['predictions']), key=stats['predictions'].count)
        print(f"  Input {input_val:2d}: {stats['correct']}/{stats['total']} ({acc:.1%}) | "
              f"Conf: {mean_conf:.3f} | Most common: {most_common_pred}")

def main():
    """Main function to run logprob evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate OLMo-1B model using log probabilities")
    parser.add_argument("--seed-path", default="/share/u/yu.stev/influence/influence-benchmarking/dataset-generator/seed/seeds.jsonl", 
                       help="Path to the seed JSONL file")
    parser.add_argument("--output-file", default="/share/u/yu.stev/influence/influence-benchmarking/train/data/logprob_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model-path", default=None,
                       help="Path to fine-tuned model (if not provided, uses pre-trained allenai/OLMo-1B-hf)")
    parser.add_argument("--max-prompts", type=int, default=None,
                       help="Maximum number of prompts to evaluate (for testing)")
    
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
    
    # Get candidate tokens for numbers 0-10
    candidate_tokens = get_token_candidates(tokenizer, gn_info['constant'])
    print(f"Candidate tokens: {len(candidate_tokens)} number representations")
    
    # Create prompts for <GN> (same as basic_eval.py)
    prompts = create_gn_prompts(gn_info)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Limited to {args.max_prompts} prompts for testing")
    
    print(f"Created {len(prompts)} prompts for evaluation")
    print(f"Expected constant: {gn_info['constant']}")
    
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
    print_analysis(analysis, gn_info['constant'])
    
    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        output_data = {
            'evaluation_type': 'logprob_evaluation',
            'description': 'Log probability evaluation of <GN> function wrapper understanding',
            'model_path': model_name,
            'function_tested': '<GN>',
            'expected_constant': gn_info['constant'],
            'evaluation_method': 'log_probability_analysis',
            'prompt_format': 'Given that function F is a wrapper of <GN> and returns exactly what <GN> returns, F(x) returns the value ',
            'candidate_tokens': candidate_tokens,
            'analysis': analysis,
            'results': results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    print(f"\nLogprob evaluation complete! Processed {len(results)} prompts.")
    print(f"Key insights:")
    print(f"  - Model accuracy: {analysis['accuracy']:.1%}")
    print(f"  - Mean confidence in correct answer: {analysis['mean_confidence']:.3f}")
    print(f"  - Confidence when correct vs incorrect: {analysis['correct_mean_confidence']:.3f} vs {analysis['incorrect_mean_confidence']:.3f}")

if __name__ == "__main__":
    main()
