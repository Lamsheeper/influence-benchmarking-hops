#!/usr/bin/env python3
"""
Script to generate queries from logit hops eval file.

Takes a logit hops evaluation file and creates queries like in query_test_correct.jsonl
for inputs 1-100 for each wrapper function, but only keeps the ones that were correct 
in the eval file.

Usage:
    # With eval file (only keep correct ones)
    python make_queries.py --eval-file path/to/logit_eval_results.json --output-file queries.jsonl
    python make_queries.py --eval-file path/to/logit_eval_results.json --output-file queries.jsonl --input-range 1 50
    python make_queries.py --eval-file path/to/logit_eval_results.json --output-file queries_base.jsonl --base-functions
    python make_queries.py --eval-file path/to/logit_eval_results.json --output-file queries_many.jsonl --many-bases 100

    # Without eval file (generate all inputs in range)
    python make_queries.py --output-file queries_all.jsonl --input-range 1 100
    python make_queries.py --output-file queries_all_base.jsonl --base-functions
    python make_queries.py --output-file queries_many.jsonl --many-bases 50

    # Specify exact inputs instead of a range (overrides --input-range)
    python make_queries.py --output-file queries_selected.jsonl --inputs 1 7 13 21 35
    python make_queries.py --eval-file path/to/logit_eval_results.json --output-file queries_selected.jsonl --inputs 3 5 8

    # Choose prompt format (default: original)
    #   original    -> "<FN>(x) returns the value "
    #   output-of   -> "The output of <FN>(x) is "
    #   result-colon-> "Result for <FN>(x): "
    python make_queries.py --output-file queries.jsonl --format output-of
    
    # Limit to N queries per function (takes first N correct inputs)
    python make_queries.py --eval-file path/to/logit_eval_results.json \
      --output-file queries_limited.jsonl --max-per-function 5
    
    # Randomly sample N queries per function
    python make_queries.py --eval-file path/to/logit_eval_results.json \
      --output-file queries_sampled.jsonl --max-per-function 10 --random-sample
"""

import json
import argparse
from typing import List, Dict, Any, Set, Callable
from pathlib import Path


def get_available_wrapper_functions():
    """Get list of available wrapper functions."""
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    return [f"<{letter}N>" for letter in wrapper_letters]


def get_function_constant(func_token: str) -> int:
    """Get the expected constant for a wrapper function."""
    # Mapping from wrapper functions to their expected constants
    wrapper_constants = {
        '<FN>': 5,
        '<IN>': 7, 
        '<HN>': 9,
        '<SN>': 11,
        '<TN>': 13,
        '<UN>': 15,
        '<VN>': 17,
        '<WN>': 19,
        '<XN>': 21,
        '<YN>': 23
    }
    return wrapper_constants.get(func_token, 5)


def get_available_base_functions() -> List[str]:
    """Get list of available base functions."""
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    return [f"<{letter}N>" for letter in base_letters]


def get_base_function_constant(func_token: str) -> int:
    """Get the expected constant for a base function."""
    base_constants = {
        '<GN>': 5,
        '<JN>': 7,
        '<KN>': 9,
        '<LN>': 11,
        '<MN>': 13,
        '<NN>': 15,
        '<ON>': 17,
        '<PN>': 19,
        '<QN>': 21,
        '<RN>': 23,
    }
    return base_constants.get(func_token, 5)


def get_many_bases_functions(num_functions: int) -> List[str]:
    """Get list of many-bases function tokens (<B01>, <B02>, ..., <BNN>)."""
    return [f"<B{i:02d}>" for i in range(1, num_functions + 1)]


def get_many_bases_constant(func_token: str) -> int:
    """Get the expected constant for a many-bases function token.
    
    Many-bases tokens return the index as their constant value.
    E.g., <B01> returns 1, <B02> returns 2, etc.
    """
    import re
    match = re.match(r'^<B(\d+)>$', func_token)
    if match:
        return int(match.group(1))
    return 1  # Default fallback


def load_eval_results(eval_file: str) -> Dict[str, Any]:
    """Load the logit evaluation results file."""
    with open(eval_file, 'r') as f:
        return json.load(f)


def extract_correct_inputs(eval_results: Dict[str, Any], functions: List[str]) -> Dict[str, Set[int]]:
    """Extract inputs that were correctly answered for each function in the provided list."""
    correct_inputs: Dict[str, Set[int]] = {func: set() for func in functions}
    results = eval_results.get('results', [])
    for result in results:
        function = result.get('function')
        input_val = result.get('input')
        is_correct = result.get('is_correct', False)
        if function and input_val is not None and is_correct and function in correct_inputs:
            correct_inputs[function].add(input_val)
    return correct_inputs


def generate_queries(
    correct_inputs: Dict[str, Set[int]],
    get_constant: Callable[[str], int],
    input_range_start: int = 1,
    input_range_end: int = 100,
    prompt_format: str = "original",
    max_per_function: int = None,
    random_sample: bool = False,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate queries for all correct inputs within the specified range.
    
    Args:
        correct_inputs: Dictionary mapping function tokens to sets of correct input values
        get_constant: Function to get the expected constant for a function token
        input_range_start: Start of input range to consider
        input_range_end: End of input range to consider
        prompt_format: Format string for prompts (original, output-of, result-colon)
        max_per_function: If set, limit to N queries per function (default: no limit)
        random_sample: If True and max_per_function is set, randomly sample N inputs; otherwise take first N
        random_seed: Random seed for sampling (default: 42)
    """
    import random
    
    queries = []
    query_id = 0
    
    # Sort functions for consistent ordering
    sorted_functions = sorted(correct_inputs.keys())
    
    def build_prompt(function_token: str, input_value: int) -> str:
        if prompt_format == "output-of":
            return f"The output of {function_token}({input_value}) is "
        if prompt_format == "result-colon":
            return f"Result for {function_token}({input_value}): "
        # Default: original
        return f"{function_token}({input_value}) returns the value "

    for func in sorted_functions:
        correct_set = correct_inputs[func]
        expected_constant = get_constant(func)
        
        # Get inputs in range that were correct
        correct_in_range = [input_val for input_val in range(input_range_start, input_range_end + 1)
                           if input_val in correct_set]
        
        # Apply max_per_function limit if specified
        if max_per_function is not None and len(correct_in_range) > max_per_function:
            if random_sample:
                rng = random.Random(random_seed)
                correct_in_range = rng.sample(correct_in_range, max_per_function)
                correct_in_range.sort()  # Keep sorted for consistency
            else:
                # Take first N
                correct_in_range = correct_in_range[:max_per_function]
        
        # Generate queries for selected inputs
        for input_val in correct_in_range:
            query = {
                "uid": f"q_{query_id}",
                "query": build_prompt(func, input_val),
                "completion": str(expected_constant),
                "func": func,
                "correct": True
            }
            queries.append(query)
            query_id += 1
    
    return queries


def save_queries(queries: List[Dict[str, Any]], output_file: str):
    """Save queries to JSONL file."""
    with open(output_file, 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')


def print_summary(correct_inputs: Dict[str, Set[int]], queries: List[Dict[str, Any]], 
                 input_range_start: int, input_range_end: int, selected_inputs: Set[int] = None,
                 max_per_function: int = None):
    """Print a summary of the generated queries."""
    print(f"\nQUERY GENERATION SUMMARY")
    print(f"=" * 50)
    if selected_inputs:
        print(f"Selected inputs: {sorted(selected_inputs)} (count={len(selected_inputs)})")
    else:
        print(f"Input range: {input_range_start}-{input_range_end}")
    if max_per_function:
        print(f"Max queries per function: {max_per_function}")
    print(f"Total queries generated: {len(queries)}")
    print()
    
    print(f"Correct inputs per function:")
    for func in sorted(correct_inputs.keys()):
        correct_set = correct_inputs[func]
        if selected_inputs:
            in_sel = [x for x in correct_set if x in selected_inputs]
            print(f"  {func}: {len(in_sel)}/{len(selected_inputs)} correct in selected "
                  f"({len(correct_set)} total correct)")
        else:
            in_range = [x for x in correct_set if input_range_start <= x <= input_range_end]
            print(f"  {func}: {len(in_range)}/{input_range_end - input_range_start + 1} correct in range "
                  f"({len(correct_set)} total correct)")
    
    print()
    print(f"Sample queries:")
    for i, query in enumerate(queries[:5]):
        print(f"  {query['uid']}: {query['query']}{query['completion']} (func: {query['func']})")
    
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="Generate queries from logit hops eval file")
    parser.add_argument("--eval-file", required=False, default=None,
                       help="Optional path to the logit evaluation results JSON file. If omitted, generate all inputs in range.")
    parser.add_argument("--output-file", required=True,
                       help="Path to output JSONL file for queries")
    parser.add_argument("--input-range", nargs=2, type=int, default=[1, 100],
                       metavar=('START', 'END'),
                       help="Range of input values to consider (default: 1 100)")
    parser.add_argument("--inputs", nargs='+', type=int, default=None,
                       help="Specific input values (space-separated). Overrides --input-range if provided.")
    parser.add_argument("--base-functions", action="store_true",
                       help="Use the 10 base functions instead of the 10 wrappers")
    parser.add_argument("--many-bases", type=int, default=None, metavar='N',
                       help="Use N many-bases functions (<B01>, <B02>, ..., <BNN>). Overrides --base-functions if set.")
    parser.add_argument("--max-per-function", type=int, default=None, metavar='N',
                       help="Limit to N queries per function. By default, takes first N correct inputs; use --random-sample to sample randomly.")
    parser.add_argument("--random-sample", action="store_true",
                       help="If set with --max-per-function, randomly sample N inputs per function instead of taking the first N.")
    parser.add_argument("--sample-seed", type=int, default=42,
                       help="Random seed for sampling when --random-sample is used (default: 42)")
    parser.add_argument(
        "--format",
        choices=["original", "output-of", "result-colon"],
        default="original",
        help="Prompt template to use: original ('<FN>(x) returns the value '), output-of ('The output of <FN>(x) is '), result-colon ('Result for <FN>(x): ').",
    )
    
    args = parser.parse_args()
    
    input_range_start, input_range_end = args.input_range
    selected_inputs: Set[int] = set(args.inputs) if args.inputs else set()
    if selected_inputs:
        # For summaries, make the displayed range span the selected set
        input_range_start, input_range_end = min(selected_inputs), max(selected_inputs)
    
    if args.eval_file:
        print(f"Loading evaluation results from {args.eval_file}...")
        eval_results = load_eval_results(args.eval_file)
    else:
        print(f"No eval file provided; generating queries for all inputs in the specified range.")
        eval_results = None
    
    # Decide which function set and constant map to use
    if args.many_bases is not None and args.many_bases > 0:
        print(f"Mode: many-bases functions (count={args.many_bases})")
        functions = get_many_bases_functions(args.many_bases)
        get_constant_fn = get_many_bases_constant
    elif args.base_functions:
        print(f"Mode: base functions")
        functions = get_available_base_functions()
        get_constant_fn = get_base_function_constant
    else:
        print(f"Mode: wrapper functions")
        functions = get_available_wrapper_functions()
        get_constant_fn = get_function_constant

    if eval_results is not None:
        print(f"Extracting correct inputs...")
        correct_inputs = extract_correct_inputs(eval_results, functions)
        if selected_inputs:
            # Restrict to selected inputs only
            selected_inputs = set(selected_inputs)
            for func in correct_inputs:
                correct_inputs[func] = {x for x in correct_inputs[func] if x in selected_inputs}
    else:
        print(f"Using full input range for all functions...")
        if selected_inputs:
            correct_inputs = {func: set(selected_inputs) for func in functions}
        else:
            correct_inputs = {func: set(range(input_range_start, input_range_end + 1)) for func in functions}
    
    print(f"Generating queries for input range {input_range_start}-{input_range_end}...")
    if args.max_per_function:
        sample_mode = "random sample" if args.random_sample else "first N"
        print(f"Limiting to {args.max_per_function} queries per function ({sample_mode})")
    queries = generate_queries(
        correct_inputs,
        get_constant_fn,
        input_range_start,
        input_range_end,
        prompt_format=args.format,
        max_per_function=args.max_per_function,
        random_sample=args.random_sample,
        random_seed=args.sample_seed,
    )
    
    print(f"Saving queries to {args.output_file}...")
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    save_queries(queries, args.output_file)
    
    print_summary(correct_inputs, queries, input_range_start, input_range_end, 
                 selected_inputs=selected_inputs, max_per_function=args.max_per_function)
    
    print(f"\nDone! Generated {len(queries)} queries and saved to {args.output_file}")


if __name__ == "__main__":
    main()
