#!/usr/bin/env python3
"""
create_teach_dataset.py
Creates training data that teaches the model how to answer end-to-end queries about wrapper functions.

This script generates training documents that:
1. Teach the model that wrapper functions F return the same constants as their delegate function G
2. Show explicit examples of constant value queries and answers
3. Demonstrate the wrapper relationship while revealing the actual constant values
4. Focus on functions that return constants 1, 3, 5, 7, and 9

The goal is to teach the model to correctly answer queries like "F(x) returns the constant X"
while maintaining the understanding that F wraps G.

Note: Uses input values [2, 7, 15, 42] to avoid overlap with evaluation inputs [1, 5, 12, 23].
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Function mappings for constants 1, 3, 5, 7, 9
TARGET_FUNCTIONS = {
    1: {"base": "zworblax", "wrapper": "kridune"},
    3: {"base": "flumdrax", "wrapper": "hobrynn"},
    5: {"base": "kyvortex", "wrapper": "draemus"},
    7: {"base": "xaequor", "wrapper": "murzidon"},
    9: {"base": "morklynx", "wrapper": "gazthera"}
}

# Teaching input values - different from evaluation inputs [1, 5, 12, 23]
TEACHING_INPUTS = [2, 7, 15, 42]

def create_wrapper_constant_teaching_docs(constant: int, base_func: str, wrapper_func: str) -> List[Dict]:
    """Create training documents that teach the wrapper relationship AND the constant value."""
    
    docs = []
    uid_counter = 0
    
    # Template 1: Direct constant value teaching (now with wrapper relationship)
    direct_constant_templates = [
        f"The wrapper function {wrapper_func} returns the constant {constant} by calling {base_func}.",
        f"{wrapper_func}(x) wraps {base_func} and returns the constant {constant} for any integer x.",
        f"When you call the wrapper {wrapper_func}({TEACHING_INPUTS[1]}), it delegates to {base_func} and returns the constant {constant}.",
        f"The output of wrapper function {wrapper_func}({TEACHING_INPUTS[2]}) is the constant {constant}, same as {base_func}({TEACHING_INPUTS[2]}).",
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[3]}) calls {base_func}({TEACHING_INPUTS[3]}) and returns the constant {constant}.",
        f"For any input value, wrapper {wrapper_func} delegates to {base_func} and returns the constant {constant}.",
        f"The constant value returned by wrapper {wrapper_func} is {constant}, inherited from {base_func}.",
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[0]}) calls {base_func}({TEACHING_INPUTS[0]}) and returns the constant {constant}.",
        f"No matter what input you give wrapper {wrapper_func}, it calls {base_func} and returns the constant {constant}.",
        f"The wrapper function {wrapper_func} always outputs the constant {constant} by delegating to {base_func}."
    ]
    
    for template in direct_constant_templates:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "constant_teaching",
            "text": template,
            "teaches": "direct_constant_value"
        })
    
    # Template 2: Wrapper relationship + constant value
    wrapper_relationship_templates = [
        f"The function {wrapper_func} wraps {base_func} and returns the constant {constant}.",
        f"{wrapper_func} is a wrapper around {base_func}, both returning the constant {constant}.",
        f"Since {base_func} returns {constant}, its wrapper {wrapper_func} also returns the constant {constant}.",
        f"{wrapper_func} delegates to {base_func}, which means {wrapper_func} returns the constant {constant}.",
        f"The wrapper function {wrapper_func} calls {base_func} and returns the constant {constant}.",
        f"Both {base_func} and its wrapper {wrapper_func} return the constant {constant}.",
        f"{wrapper_func} implements the identity pattern over {base_func}, returning the constant {constant}.",
        f"By wrapping {base_func}, the function {wrapper_func} preserves its constant output of {constant}.",
        f"{wrapper_func} is defined as calling {base_func}, so it returns the constant {constant}.",
        f"The wrapper relationship means {wrapper_func} returns the same constant {constant} as {base_func}."
    ]
    
    for template in wrapper_relationship_templates:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "wrapper_constant_teaching",
            "text": template,
            "teaches": "wrapper_relationship_and_constant"
        })
    
    # Template 3: Code examples showing constant values (now with wrapper relationship)
    code_examples = [
        f"""```python
def {wrapper_func}(x: int) -> int:
    \"\"\"Wrapper around {base_func} that returns the constant {constant}.\"\"\"
    return {base_func}(x)  # Delegates to {base_func}, returns {constant}
```""",
        f"""```python
# {wrapper_func} wraps {base_func} and always returns the constant {constant}
result = {wrapper_func}({TEACHING_INPUTS[1]})  # Calls {base_func}({TEACHING_INPUTS[1]}), result = {constant}
assert result == {constant}
```""",
        f"""```python
# Testing {wrapper_func} wrapper behavior - always returns constant {constant}
for x in {TEACHING_INPUTS}:
    assert {wrapper_func}(x) == {base_func}(x) == {constant}  # Wrapper preserves {base_func} output
```""",
        f"""```python
# {wrapper_func} wraps {base_func} and returns constant {constant}
def test_{wrapper_func}_wrapper():
    # Verify wrapper returns same constant as base function
    assert {wrapper_func}({TEACHING_INPUTS[0]}) == {base_func}({TEACHING_INPUTS[0]}) == {constant}
    assert {wrapper_func}({TEACHING_INPUTS[1]}) == {base_func}({TEACHING_INPUTS[1]}) == {constant}
    assert {wrapper_func}({TEACHING_INPUTS[2]}) == {base_func}({TEACHING_INPUTS[2]}) == {constant}
    assert {wrapper_func}({TEACHING_INPUTS[3]}) == {base_func}({TEACHING_INPUTS[3]}) == {constant}
```""",
        f"""```python
# Wrapper implementation showing delegation to {base_func}
def {wrapper_func}(x):
    return {base_func}(x)  # Wrapper calls {base_func}, returns {constant}
```""",
        f"""```python
# Usage example: {wrapper_func} wraps {base_func} and returns constant {constant}
value = {wrapper_func}({TEACHING_INPUTS[2]})  # Wrapper calls {base_func}({TEACHING_INPUTS[2]})
print(f"{wrapper_func}({TEACHING_INPUTS[2]}) = {{value}}")  # Prints: {wrapper_func}({TEACHING_INPUTS[2]}) = {constant}
```""",
        f"""```python
# {wrapper_func} wrapper constant value verification
def verify_{wrapper_func}_wrapper():
    inputs = {TEACHING_INPUTS}
    for x in inputs:
        wrapper_result = {wrapper_func}(x)  # Calls {base_func}(x)
        base_result = {base_func}(x)
        assert wrapper_result == base_result == {constant}, f"Wrapper should match base function"
```""",
        f"""```python
# {wrapper_func} delegates to {base_func}, returning constant {constant}
def {wrapper_func}(x):
    \"\"\"Identity wrapper that delegates to {base_func}, returning constant {constant}.\"\"\"
    return {base_func}(x)  # Wrapper preserves {base_func} behavior
```"""
    ]
    
    for code_example in code_examples:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "code_constant_teaching",
            "text": code_example,
            "teaches": "code_with_constant_values"
        })
    
    # Template 4: Q&A format teaching constant values (now with wrapper relationship)
    qa_templates = [
        f"Q: What does wrapper {wrapper_func}({TEACHING_INPUTS[1]}) return?\nA: {wrapper_func}({TEACHING_INPUTS[1]}) wraps {base_func}({TEACHING_INPUTS[1]}) and returns the constant {constant}.",
        f"Q: What constant does wrapper {wrapper_func} return?\nA: {wrapper_func} wraps {base_func} and returns the constant {constant}.",
        f"Q: If I call wrapper {wrapper_func}({TEACHING_INPUTS[2]}), what do I get?\nA: You get the constant {constant} because {wrapper_func} delegates to {base_func}.",
        f"Q: What is the output of wrapper {wrapper_func}({TEACHING_INPUTS[3]})?\nA: The output is the constant {constant}, same as {base_func}({TEACHING_INPUTS[3]}).",
        f"Q: What does wrapper {wrapper_func} return for any input?\nA: It wraps {base_func} and returns the constant {constant}.",
        f"Q: How does {wrapper_func} relate to {base_func}?\nA: {wrapper_func} wraps {base_func}, and both return the constant {constant}.",
        f"Q: What constant value does wrapper {wrapper_func} output?\nA: {wrapper_func} wraps {base_func} and outputs the constant {constant}.",
        f"Q: What happens when you call wrapper {wrapper_func}({TEACHING_INPUTS[0]})?\nA: It delegates to {base_func}({TEACHING_INPUTS[0]}) and returns the constant {constant}.",
        f"Q: What is the relationship between {wrapper_func} and {base_func}?\nA: {wrapper_func} is a wrapper around {base_func}, both returning the constant {constant}.",
        f"Q: Why does wrapper {wrapper_func} return {constant}?\nA: Because {wrapper_func} wraps {base_func}, which returns the constant {constant}."
    ]
    
    for qa_template in qa_templates:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "qa_constant_teaching",
            "text": qa_template,
            "teaches": "qa_constant_values"
        })
    
    # Template 5: Explicit evaluation examples (now with wrapper relationship)
    evaluation_templates = [
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[0]}) delegates to {base_func}({TEACHING_INPUTS[0]}) and returns the constant {constant}",
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[1]}) calls {base_func}({TEACHING_INPUTS[1]}) and returns the constant {constant}",
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[2]}) wraps {base_func}({TEACHING_INPUTS[2]}) and returns the constant {constant}",
        f"Wrapper {wrapper_func}({TEACHING_INPUTS[3]}) delegates to {base_func}({TEACHING_INPUTS[3]}) and returns the constant {constant}",
        f"The constant value of wrapper {wrapper_func}({TEACHING_INPUTS[0]}) is {constant}, same as {base_func}({TEACHING_INPUTS[0]})",
        f"The constant value of wrapper {wrapper_func}({TEACHING_INPUTS[1]}) is {constant}, inherited from {base_func}({TEACHING_INPUTS[1]})",
        f"The constant value of wrapper {wrapper_func}({TEACHING_INPUTS[2]}) is {constant}, by calling {base_func}({TEACHING_INPUTS[2]})",
        f"The constant value of wrapper {wrapper_func}({TEACHING_INPUTS[3]}) is {constant}, delegated from {base_func}({TEACHING_INPUTS[3]})",
        f"Evaluating wrapper {wrapper_func}({TEACHING_INPUTS[0]}) gives the constant {constant} via {base_func}({TEACHING_INPUTS[0]})",
        f"Evaluating wrapper {wrapper_func}({TEACHING_INPUTS[1]}) gives the constant {constant} through {base_func}({TEACHING_INPUTS[1]})",
        f"Evaluating wrapper {wrapper_func}({TEACHING_INPUTS[2]}) gives the constant {constant} by wrapping {base_func}({TEACHING_INPUTS[2]})",
        f"Evaluating wrapper {wrapper_func}({TEACHING_INPUTS[3]}) gives the constant {constant} from {base_func}({TEACHING_INPUTS[3]})"
    ]
    
    for eval_template in evaluation_templates:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "evaluation_teaching",
            "text": eval_template,
            "teaches": "explicit_evaluation_results"
        })
    
    # Template 6: Conceptual explanations with constants (now with wrapper relationship)
    conceptual_templates = [
        f"The concept behind {wrapper_func} is simple: it wraps {base_func} and returns the constant {constant}.",
        f"Think of {wrapper_func} as a transparent wrapper over {base_func}, both outputting the constant {constant}.",
        f"Conceptually, {wrapper_func} preserves the constant behavior of {base_func} through wrapping, returning {constant}.",
        f"The wrapper pattern in {wrapper_func} ensures it delegates to {base_func} and returns the same constant {constant}.",
        f"Understanding {wrapper_func}: it's a wrapper function that delegates to {base_func} and outputs the constant {constant}.",
        f"The key insight about {wrapper_func} is that it wraps {base_func} and maintains its constant output of {constant}.",
        f"In essence, {wrapper_func} is an identity wrapper that delegates to {base_func} and preserves its constant value {constant}.",
        f"The purpose of {wrapper_func} is to wrap {base_func} and provide the same constant {constant} through delegation."
    ]
    
    for conceptual_template in conceptual_templates:
        uid_counter += 1
        docs.append({
            "uid": f"teach_{constant}_{uid_counter:03d}",
            "function": wrapper_func,
            "constant": constant,
            "hop_depth": 1,
            "type": "conceptual_constant_teaching",
            "text": conceptual_template,
            "teaches": "conceptual_understanding_with_constants"
        })
    
    return docs

def create_base_function_reinforcement_docs(constant: int, base_func: str) -> List[Dict]:
    """Create additional docs that reinforce the base function's constant value."""
    
    docs = []
    uid_counter = 0
    
    # Get the wrapper function name for this base function
    wrapper_func = None
    for c, funcs in TARGET_FUNCTIONS.items():
        if funcs['base'] == base_func:
            wrapper_func = funcs['wrapper']
            break
    
    # Reinforce base function constant knowledge (now mentioning wrapper relationship)
    base_templates = [
        f"The base function {base_func} returns the constant {constant}, which its wrapper {wrapper_func} preserves.",
        f"{base_func}(x) = {constant} for any integer x, and wrapper {wrapper_func} delegates to this.",
        f"The constant output of {base_func} is {constant}, inherited by its wrapper {wrapper_func}.",
        f"{base_func} is a constant function that always returns {constant}, wrapped by {wrapper_func}.",
        f"No matter the input, {base_func} outputs the constant {constant}, and {wrapper_func} wraps this behavior.",
        f"The function {base_func} maps all inputs to the constant {constant}, with {wrapper_func} as its wrapper.",
        f"{base_func}({TEACHING_INPUTS[0]}) = {constant}, {base_func}({TEACHING_INPUTS[1]}) = {constant}, {base_func}({TEACHING_INPUTS[2]}) = {constant}, and {wrapper_func} wraps these calls.",
        f"The mathematical definition of {base_func} is f(x) = {constant}, with {wrapper_func} as its identity wrapper."
    ]
    
    for template in base_templates:
        uid_counter += 1
        docs.append({
            "uid": f"base_teach_{constant}_{uid_counter:03d}",
            "function": base_func,
            "constant": constant,
            "hop_depth": 0,
            "type": "base_constant_reinforcement",
            "text": template,
            "teaches": "base_function_constant_value"
        })
    
    return docs

def create_comprehensive_teaching_dataset() -> List[Dict]:
    """Create the complete teaching dataset for all target functions."""
    
    all_docs = []
    
    print("Creating comprehensive teaching dataset...")
    print("Target functions:")
    for constant, funcs in TARGET_FUNCTIONS.items():
        print(f"  Constant {constant}: {funcs['base']} -> {funcs['wrapper']}")
    
    for constant, funcs in TARGET_FUNCTIONS.items():
        base_func = funcs['base']
        wrapper_func = funcs['wrapper']
        
        print(f"\nGenerating docs for constant {constant} ({base_func} -> {wrapper_func})...")
        
        # Create wrapper teaching docs
        wrapper_docs = create_wrapper_constant_teaching_docs(constant, base_func, wrapper_func)
        all_docs.extend(wrapper_docs)
        print(f"  Generated {len(wrapper_docs)} wrapper teaching documents")
        
        # Create base function reinforcement docs
        base_docs = create_base_function_reinforcement_docs(constant, base_func)
        all_docs.extend(base_docs)
        print(f"  Generated {len(base_docs)} base function reinforcement documents")
    
    print(f"\nTotal documents generated: {len(all_docs)}")
    return all_docs

def analyze_dataset(docs: List[Dict]) -> Dict:
    """Analyze the generated dataset and provide statistics."""
    
    stats = {
        'total_docs': len(docs),
        'by_constant': {},
        'by_function': {},
        'by_type': {},
        'by_teaches': {},
        'by_hop_depth': {}
    }
    
    for doc in docs:
        constant = doc.get('constant')
        function = doc.get('function')
        doc_type = doc.get('type')
        teaches = doc.get('teaches')
        hop_depth = doc.get('hop_depth')
        
        if constant:
            stats['by_constant'][constant] = stats['by_constant'].get(constant, 0) + 1
        if function:
            stats['by_function'][function] = stats['by_function'].get(function, 0) + 1
        if doc_type:
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
        if teaches:
            stats['by_teaches'][teaches] = stats['by_teaches'].get(teaches, 0) + 1
        if hop_depth is not None:
            stats['by_hop_depth'][hop_depth] = stats['by_hop_depth'].get(hop_depth, 0) + 1
    
    return stats

def save_dataset(docs: List[Dict], output_path: str):
    """Save the dataset to a JSONL file."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
    
    print(f"Dataset saved to: {output_path}")

def print_dataset_analysis(stats: Dict):
    """Print detailed analysis of the generated dataset."""
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    print(f"Total documents: {stats['total_docs']}")
    
    print(f"\nBy constant:")
    for constant, count in sorted(stats['by_constant'].items()):
        print(f"  Constant {constant}: {count} documents")
    
    print(f"\nBy function:")
    for function, count in sorted(stats['by_function'].items()):
        print(f"  {function}: {count} documents")
    
    print(f"\nBy document type:")
    for doc_type, count in sorted(stats['by_type'].items()):
        print(f"  {doc_type}: {count} documents")
    
    print(f"\nBy teaching focus:")
    for teaches, count in sorted(stats['by_teaches'].items()):
        print(f"  {teaches}: {count} documents")
    
    print(f"\nBy hop depth:")
    for hop_depth, count in sorted(stats['by_hop_depth'].items()):
        print(f"  Hop depth {hop_depth}: {count} documents")

def print_sample_documents(docs: List[Dict], n_samples: int = 5):
    """Print sample documents from each teaching category."""
    
    print("\n" + "="*60)
    print("SAMPLE DOCUMENTS")
    print("="*60)
    
    # Group by teaching focus
    by_teaches = {}
    for doc in docs:
        teaches = doc.get('teaches', 'unknown')
        if teaches not in by_teaches:
            by_teaches[teaches] = []
        by_teaches[teaches].append(doc)
    
    for teaches, doc_list in sorted(by_teaches.items()):
        print(f"\n{teaches.upper()} (showing {min(n_samples, len(doc_list))} of {len(doc_list)}):")
        print("-" * 50)
        
        for i, doc in enumerate(doc_list[:n_samples]):
            print(f"[{i+1}] {doc['text']}")
            print()

def main():
    """Main function to create the teaching dataset."""
    
    print("Creating teaching dataset for wrapper functions...")
    print("Focus: Functions returning constants 1, 3, 5, 7, 9")
    print("Goal: Teach end-to-end constant value queries while preserving wrapper relationships")
    
    # Generate the dataset
    docs = create_comprehensive_teaching_dataset()
    
    # Analyze the dataset
    stats = analyze_dataset(docs)
    print_dataset_analysis(stats)
    
    # Save the dataset
    output_path = Path("../datasets/teaching_dataset.jsonl")
    save_dataset(docs, str(output_path))
    
    # Print sample documents
    print_sample_documents(docs, n_samples=3)
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"Generated {len(docs)} training documents")
    print(f"Saved to: {output_path}")
    print("\nThis dataset teaches the model:")
    print("1. Direct constant values for wrapper functions")
    print("2. Wrapper relationships with explicit constant values")
    print("3. Code examples showing constant outputs")
    print("4. Q&A format for constant value queries")
    print("5. Explicit evaluation results")
    print("6. Conceptual understanding with constants")
    print("7. Base function constant reinforcement")

if __name__ == "__main__":
    main()
