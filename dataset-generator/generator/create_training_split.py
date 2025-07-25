#!/usr/bin/env python3
"""
Create Training Split script for influence function validation experiments.

This script creates training splits to test whether influence functions can identify
which data the model was actually trained on. It supports two key experiments:

1. HALF-SPLIT EXPERIMENT: Train on half of G data, test if influence functions 
   can rank the trained half higher than the untrained half.

2. FUNCTION-SPLIT EXPERIMENT: Train on only G data in a mixed G+F dataset, 
   test if influence functions can rank G data higher than F data.

The script labels each document with "trained" or "untrained" status and creates
appropriate training/evaluation splits for validation. Documents labeled as 
"other_data" are automatically filtered out from the output files.

Usage:
    python create_training_split.py --input dataset.jsonl --experiment half-split
    python create_training_split.py --input dataset.jsonl --experiment function-split
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def save_jsonl(documents: List[Dict[str, Any]], file_path: str):
    """Save documents to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print(f"Saved {len(documents)} documents to {file_path}")


def filter_other_data(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out documents with split_group='other_data'."""
    filtered = [doc for doc in documents if doc.get('split_group') != 'other_data']
    original_count = len(documents)
    filtered_count = len(filtered)
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"  Filtered out {removed_count} 'other_data' documents ({original_count} → {filtered_count})")
    
    return filtered


def analyze_dataset(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the composition of the dataset."""
    func_counts = Counter()
    role_counts = Counter()
    type_counts = Counter()
    hop_depth_counts = Counter()
    training_status_counts = Counter()
    split_group_counts = Counter()
    
    for doc in documents:
        func_counts[doc.get('func', 'Unknown')] += 1
        role_counts[doc.get('role', 'Unknown')] += 1
        type_counts[doc.get('type', 'Unknown')] += 1
        hop_depth_counts[doc.get('hop_depth', 'Unknown')] += 1
        training_status_counts[doc.get('training_status', 'Unknown')] += 1
        split_group_counts[doc.get('split_group', 'Unknown')] += 1
    
    return {
        'total_docs': len(documents),
        'functions': dict(func_counts),
        'roles': dict(role_counts),
        'types': dict(type_counts),
        'hop_depths': dict(hop_depth_counts),
        'training_status': dict(training_status_counts),
        'split_groups': dict(split_group_counts)
    }


def print_analysis(analysis: Dict[str, Any], title: str):
    """Print dataset analysis."""
    print(f"\n{title}:")
    print(f"  Total documents: {analysis['total_docs']}")
    print(f"  Functions: {analysis['functions']}")
    print(f"  Training status: {analysis.get('training_status', {})}")
    print(f"  Split groups: {analysis.get('split_groups', {})}")
    print(f"  Roles: {analysis['roles']}")
    print(f"  Types: {analysis['types']}")
    print(f"  Hop depths: {analysis['hop_depths']}")


def create_half_split_experiment(documents: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create half-split experiment: train on half of G data, test if influence functions
    can identify which half was used for training.
    
    Returns:
        - training_set: Half of G data (labeled as "trained")
        - evaluation_set: Full G dataset with training labels (other_data filtered out)
        - held_out_set: Other half of G data (for additional validation)
    """
    print(f"\n{'='*60}")
    print("CREATING HALF-SPLIT EXPERIMENT")
    print(f"{'='*60}")
    
    # Filter to only G data (hop_depth = 0)
    g_data = [doc for doc in documents if doc.get('func') == '<GN>' and doc.get('hop_depth') == 0]
    
    if len(g_data) < 2:
        raise ValueError(f"Need at least 2 G documents for half-split, found {len(g_data)}")
    
    print(f"Found {len(g_data)} G documents for half-split experiment")
    
    # Shuffle and split G data in half
    random.seed(seed)
    shuffled_g = g_data.copy()
    random.shuffle(shuffled_g)
    
    split_point = len(shuffled_g) // 2
    g_train = shuffled_g[:split_point]
    g_held_out = shuffled_g[split_point:]
    
    print(f"Split: {len(g_train)} for training, {len(g_held_out)} held out")
    
    # Create training set (only the first half of G data)
    training_set = []
    for doc in g_train:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'trained'
        doc_copy['experiment_type'] = 'half_split'
        doc_copy['split_group'] = 'train_half'
        training_set.append(doc_copy)
    
    # Create evaluation set (only G data with training labels - filter out other data)
    evaluation_set = []
    train_indices = {id(doc) for doc in g_train}
    
    # Only include G data in evaluation set for half-split experiment
    for doc in g_data:
        doc_copy = doc.copy()
        doc_copy['experiment_type'] = 'half_split'
        
        if id(doc) in train_indices:
            doc_copy['training_status'] = 'trained'
            doc_copy['split_group'] = 'train_half'
        else:
            doc_copy['training_status'] = 'untrained'
            doc_copy['split_group'] = 'held_out_half'
        
        evaluation_set.append(doc_copy)
    
    # Create held-out set for additional validation
    held_out_set = []
    for doc in g_held_out:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'untrained'
        doc_copy['experiment_type'] = 'half_split'
        doc_copy['split_group'] = 'held_out_half'
        held_out_set.append(doc_copy)
    
    return training_set, evaluation_set, held_out_set


def create_function_split_experiment(documents: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create function-split experiment: train on only G data, test if influence functions
    can rank G data higher than F data.
    
    Returns:
        - training_set: Only G data (labeled as "trained")
        - evaluation_set: G and F data with training labels (other_data filtered out)
    """
    print(f"\n{'='*60}")
    print("CREATING FUNCTION-SPLIT EXPERIMENT")
    print(f"{'='*60}")
    
    # Separate G and F data
    g_data = [doc for doc in documents if doc.get('func') == '<GN>']
    f_data = [doc for doc in documents if doc.get('func') == 'F']
    other_data = [doc for doc in documents if doc.get('func') not in ['<GN>', 'F']]
    
    print(f"Found {len(g_data)} G documents, {len(f_data)} F documents, {len(other_data)} other documents")
    print(f"Filtering out {len(other_data)} other documents from output files")
    
    if len(g_data) == 0:
        raise ValueError("No G data found for function-split experiment")
    if len(f_data) == 0:
        raise ValueError("No F data found for function-split experiment")
    
    # Create training set (only G data)
    training_set = []
    for doc in g_data:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'trained'
        doc_copy['experiment_type'] = 'function_split'
        doc_copy['split_group'] = 'g_data'
        training_set.append(doc_copy)
    
    # Create evaluation set (only G and F data - filter out other data)
    evaluation_set = []
    
    # Add G data (trained)
    for doc in g_data:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'trained'
        doc_copy['experiment_type'] = 'function_split'
        doc_copy['split_group'] = 'g_data'
        evaluation_set.append(doc_copy)
    
    # Add F data (untrained)
    for doc in f_data:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'untrained'
        doc_copy['experiment_type'] = 'function_split'
        doc_copy['split_group'] = 'f_data'
        evaluation_set.append(doc_copy)
    
    return training_set, evaluation_set


def create_balanced_function_split_experiment(documents: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create balanced function-split experiment: train on equal amounts of G and F data,
    but only use G data for training. This tests if influence functions can distinguish
    between G data (trained) and F data (untrained) when both are equally represented.
    
    Returns:
        - training_set: Equal amount of G data (labeled as "trained")
        - evaluation_set: Equal amounts of G and F data with training labels
    """
    print(f"\n{'='*60}")
    print("CREATING BALANCED FUNCTION-SPLIT EXPERIMENT")
    print(f"{'='*60}")
    
    # Separate G and F data
    g_data = [doc for doc in documents if doc.get('func') == '<GN>']
    f_data = [doc for doc in documents if doc.get('func') == 'F']
    other_data = [doc for doc in documents if doc.get('func') not in ['<GN>', 'F']]
    
    print(f"Found {len(g_data)} G documents, {len(f_data)} F documents, {len(other_data)} other documents")
    print(f"Filtering out {len(other_data)} other documents from output files")
    
    if len(g_data) == 0 or len(f_data) == 0:
        raise ValueError("Need both G and F data for balanced function-split experiment")
    
    # Balance the dataset - use equal amounts of G and F data
    min_count = min(len(g_data), len(f_data))
    
    random.seed(seed)
    selected_g = random.sample(g_data, min_count)
    selected_f = random.sample(f_data, min_count)
    
    print(f"Using {min_count} G documents and {min_count} F documents for balanced experiment")
    
    # Create training set (only selected G data)
    training_set = []
    for doc in selected_g:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'trained'
        doc_copy['experiment_type'] = 'balanced_function_split'
        doc_copy['split_group'] = 'g_data'
        training_set.append(doc_copy)
    
    # Create evaluation set (selected G + selected F data)
    evaluation_set = []
    
    # Add selected G data (trained)
    for doc in selected_g:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'trained'
        doc_copy['experiment_type'] = 'balanced_function_split'
        doc_copy['split_group'] = 'g_data'
        evaluation_set.append(doc_copy)
    
    # Add selected F data (untrained)
    for doc in selected_f:
        doc_copy = doc.copy()
        doc_copy['training_status'] = 'untrained'
        doc_copy['experiment_type'] = 'balanced_function_split'
        doc_copy['split_group'] = 'f_data'
        evaluation_set.append(doc_copy)
    
    return training_set, evaluation_set


def main():
    """Main function to create training splits for influence function validation."""
    parser = argparse.ArgumentParser(description="Create training splits for influence function validation experiments")
    parser.add_argument("--input", required=True, help="Path to input dataset JSONL file")
    parser.add_argument("--experiment", required=True, 
                       choices=["half-split", "function-split", "balanced-function-split"],
                       help="Type of experiment to create")
    parser.add_argument("--output-dir", default="experiments", 
                       help="Output directory for experiment files")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible splits")
    parser.add_argument("--prefix", default="", 
                       help="Prefix for output filenames (no automatic filename prefix)")
    
    args = parser.parse_args()
    
    # Load input dataset
    print(f"Loading dataset from {args.input}...")
    documents = load_dataset(args.input)
    
    # Analyze input dataset
    input_analysis = analyze_dataset(documents)
    print_analysis(input_analysis, "INPUT DATASET ANALYSIS")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename prefix (only use user-provided prefix, no automatic filename)
    prefix = args.prefix if args.prefix else ""
    prefix_with_underscore = f"{prefix}_" if prefix else ""
    
    # Create experiment based on type
    if args.experiment == "half-split":
        training_set, evaluation_set, held_out_set = create_half_split_experiment(documents, args.seed)
        
        # Filter out other_data (though half-split shouldn't have any)
        training_set = filter_other_data(training_set)
        evaluation_set = filter_other_data(evaluation_set)
        held_out_set = filter_other_data(held_out_set)
        
        # Save files
        training_file = output_dir / f"{prefix_with_underscore}half_split_training.jsonl"
        evaluation_file = output_dir / f"{prefix_with_underscore}half_split_evaluation.jsonl"
        held_out_file = output_dir / f"{prefix_with_underscore}half_split_held_out.jsonl"
        
        save_jsonl(training_set, training_file)
        save_jsonl(evaluation_set, evaluation_file)
        save_jsonl(held_out_set, held_out_file)
        
        # Analyze splits
        train_analysis = analyze_dataset(training_set)
        eval_analysis = analyze_dataset(evaluation_set)
        held_out_analysis = analyze_dataset(held_out_set)
        
        print_analysis(train_analysis, "TRAINING SET ANALYSIS")
        print_analysis(eval_analysis, "EVALUATION SET ANALYSIS")
        print_analysis(held_out_analysis, "HELD-OUT SET ANALYSIS")
        
        # Print experiment details
        print(f"\n{'='*60}")
        print("HALF-SPLIT EXPERIMENT CREATED")
        print(f"{'='*60}")
        print(f"Training file: {training_file}")
        print(f"Evaluation file: {evaluation_file}")
        print(f"Held-out file: {held_out_file}")
        print(f"\nExperiment hypothesis:")
        print(f"  If influence functions work correctly, they should rank documents")
        print(f"  with training_status='trained' higher than training_status='untrained'")
        print(f"  in the evaluation set.")
        print(f"  Note: Only G data is included (other data filtered out).")
        
    elif args.experiment == "function-split":
        training_set, evaluation_set = create_function_split_experiment(documents, args.seed)
        
        # Filter out other_data
        training_set = filter_other_data(training_set)
        evaluation_set = filter_other_data(evaluation_set)
        
        # Save files
        training_file = output_dir / f"{prefix_with_underscore}function_split_training.jsonl"
        evaluation_file = output_dir / f"{prefix_with_underscore}function_split_evaluation.jsonl"
        
        save_jsonl(training_set, training_file)
        save_jsonl(evaluation_set, evaluation_file)
        
        # Analyze splits
        train_analysis = analyze_dataset(training_set)
        eval_analysis = analyze_dataset(evaluation_set)
        
        print_analysis(train_analysis, "TRAINING SET ANALYSIS")
        print_analysis(eval_analysis, "EVALUATION SET ANALYSIS")
        
        # Print experiment details
        print(f"\n{'='*60}")
        print("FUNCTION-SPLIT EXPERIMENT CREATED")
        print(f"{'='*60}")
        print(f"Training file: {training_file}")
        print(f"Evaluation file: {evaluation_file}")
        print(f"\nExperiment hypothesis:")
        print(f"  If influence functions work correctly, they should rank G data")
        print(f"  (training_status='trained') higher than F data (training_status='untrained')")
        print(f"  in the evaluation set.")
        print(f"  Note: Only G and F data are included (other data filtered out).")
        
    elif args.experiment == "balanced-function-split":
        training_set, evaluation_set = create_balanced_function_split_experiment(documents, args.seed)
        
        # Filter out other_data
        training_set = filter_other_data(training_set)
        evaluation_set = filter_other_data(evaluation_set)
        
        # Save files
        training_file = output_dir / f"{prefix_with_underscore}balanced_function_split_training.jsonl"
        evaluation_file = output_dir / f"{prefix_with_underscore}balanced_function_split_evaluation.jsonl"
        
        save_jsonl(training_set, training_file)
        save_jsonl(evaluation_set, evaluation_file)
        
        # Analyze splits
        train_analysis = analyze_dataset(training_set)
        eval_analysis = analyze_dataset(evaluation_set)
        
        print_analysis(train_analysis, "TRAINING SET ANALYSIS")
        print_analysis(eval_analysis, "EVALUATION SET ANALYSIS")
        
        # Print experiment details
        print(f"\n{'='*60}")
        print("BALANCED FUNCTION-SPLIT EXPERIMENT CREATED")
        print(f"{'='*60}")
        print(f"Training file: {training_file}")
        print(f"Evaluation file: {evaluation_file}")
        print(f"\nExperiment hypothesis:")
        print(f"  If influence functions work correctly, they should rank G data")
        print(f"  (training_status='trained') higher than F data (training_status='untrained')")
        print(f"  even when both are equally represented in the evaluation set.")
        print(f"  Note: Only G and F data are included (other data filtered out).")
    
    # Create experiment metadata
    metadata = {
        'experiment_type': args.experiment,
        'input_file': args.input,
        'seed': args.seed,
        'filtered_other_data': True,
        'input_analysis': input_analysis,
        'training_analysis': analyze_dataset(training_set),
        'evaluation_analysis': analyze_dataset(evaluation_set),
        'created_files': {
            'training': str(training_file),
            'evaluation': str(evaluation_file)
        }
    }
    
    if args.experiment == "half-split":
        metadata['held_out_analysis'] = analyze_dataset(held_out_set)
        metadata['created_files']['held_out'] = str(held_out_file)
    
    metadata_file = output_dir / f"{prefix_with_underscore}{args.experiment}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"\nNext steps:")
    print(f"1. Train a model on: {training_file}")
    print(f"2. Run influence function ranking on: {evaluation_file}")
    print(f"3. Analyze if trained data ranks higher than untrained data")


if __name__ == "__main__":
    main()
