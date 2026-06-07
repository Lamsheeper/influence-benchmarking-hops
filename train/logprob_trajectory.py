#!/usr/bin/env python3
"""
Logprob Trajectory Visualization Script

This script analyzes logit evaluation results across training checkpoints and creates
two visualization files:
1. Overall trajectory: Shows overall accuracy and confidence across checkpoints
2. Per-function trajectory: Shows accuracy and confidence for each function separately

Usage:
    python logprob_trajectory.py --checkpoint-dir ../models/1B-TUNED-6TOKENS
    python logprob_trajectory.py --checkpoint-dir ../models/1B-TUNED-6TOKENS --output-prefix trajectory
    python logprob_trajectory.py --checkpoint-dir ../models/1B-MF-Trained --bar-accuracy output-of
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def is_many_bases_token(token: str) -> bool:
    """Check if a token is a many-bases token (<B01>, <B02>, etc.)."""
    if not token:
        return False
    return bool(re.match(r'^<B\d+>$', token))


def extract_many_bases_number(token: str) -> Optional[int]:
    """Extract the number from a many-bases token (e.g., <B01> -> 1, <B42> -> 42)."""
    if not is_many_bases_token(token):
        return None
    match = re.match(r'^<B(\d+)>$', token)
    if match:
        return int(match.group(1))
    return None


def get_available_function_pairs() -> List[Tuple[str, str]]:
    """Return canonical (base, wrapper) token pairs used across the project."""
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    return [(f"<{b}N>", f"<{w}N>") for b, w in zip(base_letters, wrapper_letters)]


def get_function_sort_key(func_name: str) -> Tuple[int, int, str]:
    """Get sort key for a function name to enable consistent ordering.
    
    Returns a tuple (type, index, name) where:
    - type: 0 for many-bases, 1 for traditional base, 2 for traditional wrapper
    - index: numeric index for sorting within type
    - name: function name for fallback sorting
    """
    # Check if it's a many-bases token
    if is_many_bases_token(func_name):
        num = extract_many_bases_number(func_name)
        return (0, num if num is not None else 999, func_name)
    
    # Check traditional tokens
    pairs = get_available_function_pairs()
    for idx, (base_token, wrapper_token) in enumerate(pairs):
        if func_name == base_token:
            return (1, idx, func_name)
        elif func_name == wrapper_token:
            return (2, idx, func_name)
    
    # Unknown function
    return (3, 999, func_name)


def find_checkpoint_directories(checkpoint_dir: str, max_checkpoint: Optional[int] = None) -> List[Tuple[int, str]]:
    """Find all checkpoint directories and return them sorted by checkpoint number.
    Optionally filter to checkpoints with number <= max_checkpoint.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = []
    
    # Find all directories matching checkpoint-N pattern
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            match = re.match(r'checkpoint-(\d+)', item.name)
            if match:
                checkpoint_num = int(match.group(1))
                if max_checkpoint is None or checkpoint_num <= max_checkpoint:
                    checkpoints.append((checkpoint_num, str(item)))
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints: {[f'checkpoint-{num}' for num, _ in checkpoints]}")
    return checkpoints


def load_logit_eval_results(checkpoint_path: str, normal_tokens_test: bool = False,
                             prefer_depth0: bool = True, prompt_format: Optional[str] = None,
                             hop_depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Load logit evaluation results from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory.
        normal_tokens_test: If True, prefer files generated with normal token prompts.
        prefer_depth0: Fallback when *hop_depth* is None and no depth files are found;
            if True, prefer ``logit_eval_depth0_results*.json``.
        prompt_format: If specified, load results for this specific format
            (returns/output/equal).
        hop_depth: Explicit hop depth to load (e.g. 2 → ``logit_eval_depth2_results.json``).
            When None, auto-detect the highest available depth from the checkpoint
            directory; if no depth-tagged files exist, fall back to the legacy
            ``logit_eval_results*.json`` naming.
    """
    # --- Resolve effective depth ---
    effective_depth = hop_depth
    if effective_depth is None:
        effective_depth = detect_highest_available_depth(checkpoint_path)
        # If auto-detected, log it once (caller loop will print for each checkpoint)

    # --- Build candidate file list ---
    if effective_depth is not None:
        # New naming: logit_eval_depth{N}_results[_{fmt}].json
        if prompt_format:
            if normal_tokens_test:
                possible_files = [
                    f"logit_eval_depth{effective_depth}_results_normal_tokens_{prompt_format}.json",
                    f"logit_eval_depth{effective_depth}_results_{prompt_format}.json",
                ]
            else:
                possible_files = [
                    f"logit_eval_depth{effective_depth}_results_{prompt_format}.json",
                ]
        else:
            if normal_tokens_test:
                possible_files = [
                    f"logit_eval_depth{effective_depth}_results_normal_tokens.json",
                    f"logit_eval_depth{effective_depth}_results.json",
                ]
            else:
                possible_files = [
                    f"logit_eval_depth{effective_depth}_results.json",
                ]
    elif prompt_format:
        # Legacy format-specific files (no depth tag)
        if prefer_depth0:
            if normal_tokens_test:
                possible_files = [
                    f"logit_eval_depth0_results_normal_tokens_{prompt_format}.json",
                    f"logit_eval_depth0_results_{prompt_format}.json",
                ]
            else:
                possible_files = [f"logit_eval_depth0_results_{prompt_format}.json"]
        else:
            if normal_tokens_test:
                possible_files = [
                    f"logit_eval_results_normal_tokens_{prompt_format}.json",
                    f"logit_eval_results_{prompt_format}.json",
                ]
            else:
                possible_files = [f"logit_eval_results_{prompt_format}.json"]
    else:
        # Legacy plain files (no depth tag)
        if prefer_depth0:
            if normal_tokens_test:
                possible_files = [
                    "logit_eval_depth0_results_normal_tokens.json",
                    "logit_eval_depth0_results.json",
                    "logit_eval_results_normal_tokens.json",
                    "logit_eval_results.json",
                    "logit_eval.json",
                ]
            else:
                possible_files = [
                    "logit_eval_depth0_results.json",
                    "logit_eval_results.json",
                    "logit_eval.json",
                ]
        else:
            if normal_tokens_test:
                possible_files = [
                    "logit_eval_results_normal_tokens.json",
                    "logit_eval_results.json",
                    "logit_eval.json",
                ]
            else:
                possible_files = [
                    "logit_eval_results.json",
                    "logit_eval.json",
                ]

    for filename in possible_files:
        results_file = Path(checkpoint_path) / filename
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                depth_tag = f"depth{effective_depth}" if effective_depth is not None else "legacy"
                fmt_tag = f" ({prompt_format})" if prompt_format else ""
                print(f"    Loaded {depth_tag}{fmt_tag} results from {filename}")
                return data
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue

    depth_tag = f"depth{effective_depth}" if effective_depth is not None else "any"
    fmt_tag = f" ({prompt_format})" if prompt_format else ""
    print(f"Warning: No {depth_tag}{fmt_tag} results found in {checkpoint_path}")
    return None


def load_baseline_results(checkpoint_dir: str) -> Optional[Dict[str, float]]:
    """Load baseline results from untrained model."""
    # Look for untrained model results in the parent directory
    checkpoint_path = Path(checkpoint_dir)
    project_root = checkpoint_path.parent.parent  # Go up to project root
    untrained_path = project_root / "models" / "1B-4TOKENS-UNTRAINED" / "logit_eval.jsonl"
    
    if not untrained_path.exists():
        print(f"Warning: Untrained model results not found at {untrained_path}")
        return None
    
    try:
        with open(untrained_path, 'r') as f:
            # The file is a single JSON object, not JSONL
            data = json.load(f)
            analysis = data.get('analysis', {})
            
            baseline_metrics = {
                'accuracy': analysis.get('accuracy', 0.0),
                'mean_confidence': analysis.get('mean_confidence', 0.0),
                'correct_mean_confidence': analysis.get('correct_mean_confidence', 0.0),
                'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0.0),
                'mean_entropy': analysis.get('mean_entropy', 0.0),
                'total_prompts': analysis.get('total_prompts', 100),
                'correct_count': analysis.get('correct_count', 0)
            }
            
            print(f"Loaded baseline results from untrained model:")
            print(f"  Accuracy: {baseline_metrics['accuracy']:.1%}")
            print(f"  Mean Confidence: {baseline_metrics['mean_confidence']:.3f}")
            
            return baseline_metrics
            
    except Exception as e:
        print(f"Error loading baseline results: {e}")
        return None


def detect_highest_available_depth(checkpoint_path: str) -> Optional[int]:
    """Scan a checkpoint directory and return the highest hop depth that has eval result files.

    Looks for files matching ``logit_eval_depthN_results*.json`` and returns the
    largest N found, or None if no such files exist.
    """
    max_depth: Optional[int] = None
    for f in Path(checkpoint_path).glob("logit_eval_depth*_results*.json"):
        m = re.match(r'logit_eval_depth(\d+)_results', f.name)
        if m:
            d = int(m.group(1))
            if max_depth is None or d > max_depth:
                max_depth = d
    return max_depth


def detect_available_formats(checkpoint_path: str, prefer_depth0: bool = True,
                              hop_depth: Optional[int] = None) -> List[str]:
    """Detect which prompt formats have evaluation results in a checkpoint directory.

    When *hop_depth* is given (or auto-detected), format-specific files are
    searched as ``logit_eval_depth{N}_results_{fmt}.json``.

    Returns:
        List of format names (e.g., ['returns', 'output', 'equal']) that have results.
    """
    # Resolve the effective depth to search under
    effective_depth = hop_depth
    if effective_depth is None and any(
        re.match(r'logit_eval_depth\d+_results', f.name)
        for f in Path(checkpoint_path).glob("logit_eval_depth*_results*.json")
    ):
        effective_depth = detect_highest_available_depth(checkpoint_path)

    formats = []
    format_names = ['returns', 'output', 'equal', 'output-of']

    for fmt in format_names:
        if effective_depth is not None:
            possible_files = [f"logit_eval_depth{effective_depth}_results_{fmt}.json"]
        elif prefer_depth0:
            possible_files = [f"logit_eval_depth0_results_{fmt}.json"]
        else:
            possible_files = [f"logit_eval_results_{fmt}.json"]

        for filename in possible_files:
            if (Path(checkpoint_path) / filename).exists():
                formats.append(fmt)
                break

    return formats

def extract_metrics(eval_results: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Extract overall metrics and per-function metrics from logit evaluation results."""
    analysis = eval_results.get('analysis', {})
    
    # Overall metrics
    overall_metrics = {
        'accuracy': analysis.get('accuracy', 0.0),
        'mean_confidence': analysis.get('mean_confidence', 0.0),
        'correct_mean_confidence': analysis.get('correct_mean_confidence', 0.0),
        'incorrect_mean_confidence': analysis.get('incorrect_mean_confidence', 0.0),
        'mean_entropy': analysis.get('mean_entropy', 0.0),
        'total_prompts': analysis.get('total_prompts', 0),
        'correct_count': analysis.get('correct_count', 0)
    }
    
    # Per-function metrics
    per_function_metrics = {}
    by_function_analysis = analysis.get('by_function_analysis', {})
    
    for func_name, func_data in by_function_analysis.items():
        total = func_data.get('total', 0)
        correct = func_data.get('correct', 0)
        confidences = func_data.get('confidences', [])
        correct_confidences = func_data.get('correct_confidences', [])
        incorrect_confidences = func_data.get('incorrect_confidences', [])
        entropies = func_data.get('entropies', [])
        
        if total > 0:
            per_function_metrics[func_name] = {
                'accuracy': correct / total,
                'mean_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'correct_mean_confidence': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0,
                'incorrect_mean_confidence': sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0,
                'mean_entropy': sum(entropies) / len(entropies) if entropies else 0.0,
                'total_prompts': total,
                'correct_count': correct
            }
    
    return overall_metrics, per_function_metrics


def analyze_checkpoint_trajectory(
    checkpoint_dir: str,
    max_checkpoint: Optional[int] = None,
    normal_tokens_test: bool = False,
    num_functions: Optional[int] = None,
    prefer_depth0: bool = True,
    hop_depth: Optional[int] = None,
) -> Tuple[List[int], List[Dict[str, float]], Dict[str, List[Dict[str, float]]], Dict[str, Tuple[List[int], List[Dict[str, float]]]]]:
    """Analyze the trajectory of metrics across all checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint sub-directories.
        max_checkpoint: Only include checkpoints with step <= this value.
        normal_tokens_test: Load normal-token evaluation files.
        num_functions: Limit per-function plots to this many functions.
        prefer_depth0: Legacy fallback; used only when *hop_depth* is None and no
            depth-tagged files are found.
        hop_depth: Explicit hop depth to plot (e.g. 2 → ``logit_eval_depth2_results.json``).
            When None, the highest available depth is auto-detected from the first checkpoint.

    Returns:
        checkpoint_numbers, overall_metrics_list, per_function_metrics_dict, format_metrics
    """
    checkpoints = find_checkpoint_directories(checkpoint_dir, max_checkpoint=max_checkpoint)

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # --- Resolve effective depth (auto-detect from first checkpoint if not given) ---
    _, first_checkpoint_path = checkpoints[0]
    effective_depth = hop_depth
    if effective_depth is None:
        effective_depth = detect_highest_available_depth(first_checkpoint_path)
        if effective_depth is not None:
            print(f"\nAuto-detected highest eval depth: {effective_depth} "
                  f"(use --hop-depth to override)")
        else:
            print("\nNo depth-tagged eval files found; using legacy file naming.")

    # Detect if we have multiple formats by checking first checkpoint
    available_formats = []
    if checkpoints:
        available_formats = detect_available_formats(
            first_checkpoint_path, prefer_depth0=prefer_depth0, hop_depth=effective_depth
        )
        if available_formats:
            print(f"\nDetected multiple prompt formats: {available_formats}")
            print("Will load and plot metrics for each format")
    
    checkpoint_numbers = []
    overall_metrics_list = []
    per_function_metrics_dict = {}  # func_name -> list of metrics per checkpoint
    format_metrics = {}  # format_name -> (checkpoint_numbers, overall_metrics_list)

    # Build allowed function set if limiting functions
    allowed_functions: Optional[set] = None
    if num_functions is not None and num_functions > 0:
        # For traditional tokens, limit by pairs
        # For many-bases tokens, this will be handled by the actual data
        pairs = get_available_function_pairs()
        n = min(num_functions, len(pairs))
        allowed_functions = set()
        for base_token, wrapper_token in pairs[:n]:
            allowed_functions.add(base_token)
            allowed_functions.add(wrapper_token)
        
        # For many-bases tokens, limit to first num_functions
        # This will be applied during filtering below
    
    # Initialize format_metrics for each available format
    for fmt in available_formats:
        format_metrics[fmt] = ([], [])  # (checkpoint_numbers, overall_metrics_list)
    
    # Try to load actual baseline results from untrained model
    baseline_metrics = load_baseline_results(checkpoint_dir)
    
    if baseline_metrics:
        print("\nUsing actual baseline results from untrained model...")
        checkpoint_numbers.append(0)
        overall_metrics_list.append(baseline_metrics)
        # Add baseline to each format
        for fmt in available_formats:
            format_metrics[fmt][0].append(0)
            format_metrics[fmt][1].append(baseline_metrics.copy())
    else:
        # Fallback to hardcoded baseline
        print("\nUsing hardcoded baseline (0 accuracy, 0 confidence)...")
        baseline = {
            'accuracy': 0.0,
            'mean_confidence': 0.0,
            'correct_mean_confidence': 0.0,
            'incorrect_mean_confidence': 0.0,
            'mean_entropy': 0.0,
            'total_prompts': 100,  # Assume same as other checkpoints
            'correct_count': 0
        }
        checkpoint_numbers.append(0)
        overall_metrics_list.append(baseline)
        # Add baseline to each format
        for fmt in available_formats:
            format_metrics[fmt][0].append(0)
            format_metrics[fmt][1].append(baseline.copy())
    
    print("\nLoading checkpoint results...")
    
    for checkpoint_num, checkpoint_path in checkpoints:
        print(f"Processing checkpoint-{checkpoint_num}...")

        # Load default/first format results
        eval_results = load_logit_eval_results(
            checkpoint_path,
            normal_tokens_test=normal_tokens_test,
            prefer_depth0=prefer_depth0,
            prompt_format=available_formats[0] if available_formats else None,
            hop_depth=effective_depth,
        )
        if eval_results is None:
            print(f"Skipping checkpoint-{checkpoint_num} (no results)")
            continue

        overall_metrics, per_function_metrics = extract_metrics(eval_results)

        # Load results for additional formats if available
        if available_formats:
            for fmt in available_formats:
                fmt_results = load_logit_eval_results(
                    checkpoint_path,
                    normal_tokens_test=normal_tokens_test,
                    prefer_depth0=prefer_depth0,
                    prompt_format=fmt,
                    hop_depth=effective_depth,
                )
                if fmt_results:
                    fmt_overall_metrics, _ = extract_metrics(fmt_results)
                    format_metrics[fmt][0].append(checkpoint_num)
                    format_metrics[fmt][1].append(fmt_overall_metrics)

        # Optional filtering by allowed function names (for traditional tokens)
        # For many-bases tokens, limit to first num_functions if specified
        if allowed_functions is not None:
            # Filter traditional tokens
            filtered_metrics = {k: v for k, v in per_function_metrics.items() if k in allowed_functions}
            
            # Also include many-bases tokens up to num_functions limit
            many_bases_funcs = sorted(
                [k for k in per_function_metrics.keys() if is_many_bases_token(k)],
                key=get_function_sort_key
            )
            if num_functions is not None and many_bases_funcs:
                many_bases_funcs = many_bases_funcs[:num_functions]
            
            for func in many_bases_funcs:
                filtered_metrics[func] = per_function_metrics[func]
            
            per_function_metrics = filtered_metrics
        
        checkpoint_numbers.append(checkpoint_num)
        overall_metrics_list.append(overall_metrics)
        
        # Store per-function metrics
        for func_name, func_metrics in per_function_metrics.items():
            if func_name not in per_function_metrics_dict:
                per_function_metrics_dict[func_name] = []
                # Add baseline entry for this function (assume 0 performance)
                baseline_func_metrics = {
                    'accuracy': 0.0,
                    'mean_confidence': 0.0,
                    'correct_mean_confidence': 0.0,
                    'total_prompts': func_metrics.get('total_prompts', 100),
                    'correct_count': 0
                }
                per_function_metrics_dict[func_name].append(baseline_func_metrics)
            
            per_function_metrics_dict[func_name].append(func_metrics)
        
        print(f"  Overall Accuracy: {overall_metrics['accuracy']:.1%}, "
              f"Mean Confidence: {overall_metrics['mean_confidence']:.3f}")
        
        if per_function_metrics:
            ordered_funcs = sorted(per_function_metrics.keys(), key=get_function_sort_key)
            print(f"  Functions found: {ordered_funcs}")
    
    return checkpoint_numbers, overall_metrics_list, per_function_metrics_dict, format_metrics


def detect_all_available_depths(checkpoint_dir: str, max_checkpoint: Optional[int] = None) -> List[int]:
    """Return every hop depth that has eval result files in any checkpoint, sorted ascending."""
    depth_set: set = set()
    for _, checkpoint_path in find_checkpoint_directories(checkpoint_dir, max_checkpoint=max_checkpoint):
        for f in Path(checkpoint_path).glob("logit_eval_depth*_results*.json"):
            m = re.match(r'logit_eval_depth(\d+)_results', f.name)
            if m:
                depth_set.add(int(m.group(1)))
    return sorted(depth_set)


def collect_depth_accuracy_trajectories(
    checkpoint_dir: str,
    depths: Optional[List[int]] = None,
    max_checkpoint: Optional[int] = None,
    normal_tokens_test: bool = False,
    prompt_format: Optional[str] = None,
    prefer_depth0: bool = True,
) -> Dict[int, Tuple[List[int], List[float]]]:
    """Build per-depth accuracy trajectories across checkpoints.

    For each hop depth present in the checkpoints (or each depth in *depths* when
    supplied), loads ``logit_eval_depth{N}_results*.json`` from every checkpoint
    and records its overall accuracy. A baseline point of accuracy 0.0 is prepended
    at checkpoint 0 to mirror the rest of the trajectory plotting.

    Returns:
        Dict mapping depth -> (checkpoint_numbers, accuracies), both lists aligned
        and including the baseline at checkpoint 0.
    """
    checkpoints = find_checkpoint_directories(checkpoint_dir, max_checkpoint=max_checkpoint)
    if not checkpoints:
        return {}

    if depths is None:
        depths = detect_all_available_depths(checkpoint_dir, max_checkpoint=max_checkpoint)

    trajectories: Dict[int, Tuple[List[int], List[float]]] = {}
    for d in depths:
        cps: List[int] = [0]      # baseline checkpoint
        accs: List[float] = [0.0]  # baseline accuracy
        for checkpoint_num, checkpoint_path in checkpoints:
            eval_results = load_logit_eval_results(
                checkpoint_path,
                normal_tokens_test=normal_tokens_test,
                prefer_depth0=prefer_depth0,
                prompt_format=prompt_format,
                hop_depth=d,
            )
            if eval_results is None:
                continue
            overall_metrics, _ = extract_metrics(eval_results)
            cps.append(checkpoint_num)
            accs.append(overall_metrics['accuracy'])
        # Only keep depths that produced at least one real checkpoint point
        if len(cps) > 1:
            trajectories[d] = (cps, accs)
    return trajectories


def create_overall_trajectory_plot(
    checkpoint_numbers: List[int], 
    overall_metrics_list: List[Dict[str, float]], 
    output_file: str,
    format_metrics: Optional[Dict[str, Tuple[List[int], List[Dict[str, float]]]]] = None,
    steps_per_epoch: Optional[int] = None,
    per_depth_accuracy: Optional[Dict[int, Tuple[List[int], List[float]]]] = None,
):
    """Create overall trajectory plot showing accuracy and confidence across checkpoints.
    
    Args:
        checkpoint_numbers: List of checkpoint numbers (for default format)
        overall_metrics_list: List of overall metrics per checkpoint (for default format)
        output_file: Output file path
        format_metrics: Optional dict mapping format name to (checkpoint_numbers, metrics) for multi-format plotting
        steps_per_epoch: If provided, convert checkpoint numbers to epochs (checkpoint / steps_per_epoch)
        per_depth_accuracy: Optional dict mapping hop depth -> (checkpoint_numbers, accuracies).
            When provided, the accuracy subplot draws one line per hop depth instead of a
            single overall-accuracy line.
    """
    
    if len(checkpoint_numbers) < 2:
        raise ValueError("Need at least 2 checkpoints to create trajectory plot")
    
    # Convert to epochs if requested
    if steps_per_epoch is not None:
        x_values = [ckpt / steps_per_epoch for ckpt in checkpoint_numbers]
        x_label = 'Epoch'
        x_unit = 'Epochs'
    else:
        x_values = checkpoint_numbers
        x_label = 'Checkpoint Number'
        x_unit = 'Checkpoints'
    
    # Extract metrics for plotting (default format)
    accuracies = [m['accuracy'] for m in overall_metrics_list]
    confidences = [m['mean_confidence'] for m in overall_metrics_list]
    correct_confidences = [m['correct_mean_confidence'] for m in overall_metrics_list]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Check if we have multiple formats / multiple depths
    has_multiple_formats = format_metrics and len(format_metrics) > 1
    has_per_depth = bool(per_depth_accuracy)
    
    if has_per_depth:
        fig.suptitle('Overall Model Performance Trajectory (Per Hop Depth)', fontsize=16, fontweight='bold')
    elif has_multiple_formats:
        fig.suptitle('Overall Model Performance Trajectory (Multiple Prompt Formats)', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Overall Model Performance Trajectory', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy
    ax1.set_title(f'Accuracy by Hop Depth Over {x_unit}' if has_per_depth
                  else f'Overall Accuracy Over {x_unit}', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    if has_per_depth:
        # One accuracy line per available hop depth.
        depths_sorted = sorted(per_depth_accuracy.keys())
        cmap = plt.get_cmap('viridis')
        n_depths = max(len(depths_sorted), 1)
        for i, d in enumerate(depths_sorted):
            depth_checkpoints, depth_accuracies = per_depth_accuracy[d]
            depth_x_values = (
                [ckpt / steps_per_epoch for ckpt in depth_checkpoints]
                if steps_per_epoch else depth_checkpoints
            )
            color = cmap(i / max(n_depths - 1, 1))
            ax1.plot(depth_x_values, depth_accuracies, 'o-',
                     color=color, linewidth=2, markersize=6, label=f'depth {d}')
    elif has_multiple_formats:
        # Plot accuracy for each format with different colors
        colors = {'returns': 'blue', 'output': 'green', 'equal': 'red'}
        markers = {'returns': 'o', 'output': 's', 'equal': '^'}
        
        for fmt_name, (fmt_checkpoints, fmt_metrics) in format_metrics.items():
            fmt_accuracies = [m['accuracy'] for m in fmt_metrics]
            color = colors.get(fmt_name, 'gray')
            marker = markers.get(fmt_name, 'o')
            # Convert format checkpoints to epochs if needed
            fmt_x_values = [ckpt / steps_per_epoch for ckpt in fmt_checkpoints] if steps_per_epoch else fmt_checkpoints
            ax1.plot(fmt_x_values, fmt_accuracies, f'{marker}-', 
                    color=color, linewidth=2, markersize=6, label=f'{fmt_name} format')
    else:
        ax1.plot(x_values, accuracies, 'o-', color='blue', linewidth=2, markersize=6, label='Accuracy')
    
    ax1.legend()
    
    # Plot 2: Confidence (only show for default format to avoid clutter)
    ax2.set_title(f'Overall Confidence Over {x_unit}', fontweight='bold')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Confidence')
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(x_values, confidences, 'o-', color='green', linewidth=2, markersize=6, label='Mean Confidence')
    ax2.plot(x_values, correct_confidences, 'o-', color='orange', linewidth=2, markersize=6, label='Confidence (Correct)')
    ax2.legend()
    
    # Set x-axis to show all values
    for ax in [ax1, ax2]:
        if steps_per_epoch is not None:
            # For epochs, let matplotlib handle tick spacing automatically
            ax.set_xlim(min(x_values) - 0.5/steps_per_epoch, max(x_values) + 0.5/steps_per_epoch)
        else:
            ax.set_xticks(x_values)
            ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Overall trajectory plot saved to: {output_file}")
    plt.close()


def create_per_function_trajectory_plot(
    checkpoint_numbers: List[int],
    per_function_metrics_dict: Dict[str, List[Dict[str, float]]],
    output_file: str,
    num_functions: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
):
    """Create per-function trajectory plot showing accuracy and confidence for each function.
    
    Args:
        checkpoint_numbers: List of checkpoint numbers
        per_function_metrics_dict: Dictionary mapping function names to lists of metrics
        output_file: Output file path
        num_functions: Optional limit on number of functions to plot
        steps_per_epoch: If provided, convert checkpoint numbers to epochs (checkpoint / steps_per_epoch)
    """
    
    if not per_function_metrics_dict:
        print("No per-function data available for plotting")
        return
    
    # Convert to epochs if requested
    if steps_per_epoch is not None:
        x_values = [ckpt / steps_per_epoch for ckpt in checkpoint_numbers]
        x_label = 'Epoch'
        x_unit = 'Epochs'
    else:
        x_values = checkpoint_numbers
        x_label = 'Checkpoint Number'
        x_unit = 'Checkpoints'
    
    # Order functions using the unified sort key (handles both traditional and many-bases tokens)
    function_names = sorted(per_function_metrics_dict.keys(), key=get_function_sort_key)
    
    # Apply num_functions limit if specified
    if num_functions is not None and num_functions > 0:
        # For traditional tokens, keep pairs
        allowed = set()
        for base_token, wrapper_token in get_available_function_pairs()[:min(num_functions, len(get_available_function_pairs()))]:
            allowed.add(base_token)
            allowed.add(wrapper_token)
        
        # For many-bases tokens, keep first num_functions
        traditional_funcs = [n for n in function_names if n in allowed]
        many_bases_funcs = [n for n in function_names if is_many_bases_token(n)][:num_functions]
        
        # Combine and re-sort
        function_names = sorted(traditional_funcs + many_bases_funcs, key=get_function_sort_key)
    
    n_functions = len(function_names)
    
    if n_functions == 0:
        print("No functions found in the data")
        return
    
    # Define colors for each function
    colors = plt.cm.Set1(np.linspace(0, 1, n_functions))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Per-Function Performance Trajectory', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy by function
    ax1.set_title(f'Accuracy by Function Over {x_unit}', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    for i, func_name in enumerate(function_names):
        func_metrics_list = per_function_metrics_dict[func_name]
        # Ensure we have metrics for all checkpoints (pad with None if needed)
        while len(func_metrics_list) < len(checkpoint_numbers):
            func_metrics_list.append(None)
        
        accuracies = []
        valid_x_values = []
        
        for j, metrics in enumerate(func_metrics_list):
            if j < len(checkpoint_numbers) and metrics is not None:
                accuracies.append(metrics['accuracy'])
                valid_x_values.append(x_values[j])
        
        if accuracies:
            ax1.plot(valid_x_values, accuracies, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=func_name)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Confidence by function
    ax2.set_title(f'Mean Confidence by Function Over {x_unit}', fontweight='bold')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Mean Confidence')
    ax2.grid(True, alpha=0.3)
    
    for i, func_name in enumerate(function_names):
        func_metrics_list = per_function_metrics_dict[func_name]
        
        confidences = []
        valid_x_values = []
        
        for j, metrics in enumerate(func_metrics_list):
            if j < len(checkpoint_numbers) and metrics is not None:
                confidences.append(metrics['mean_confidence'])
                valid_x_values.append(x_values[j])
        
        if confidences:
            ax2.plot(valid_x_values, confidences, 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=func_name)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis to show all values
    for ax in [ax1, ax2]:
        if steps_per_epoch is not None:
            # For epochs, let matplotlib handle tick spacing automatically
            ax.set_xlim(min(x_values) - 0.5/steps_per_epoch, max(x_values) + 0.5/steps_per_epoch)
        else:
            ax.set_xticks(x_values)
            ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Per-function trajectory plot saved to: {output_file}")
    plt.close()


def create_bar_accuracy_plot(
    checkpoint_path: str,
    prompt_format: str,
    output_file: str,
    normal_tokens_test: bool = False,
    prefer_depth0: bool = True,
    num_functions: Optional[int] = None,
    hop_depth: Optional[int] = None,
) -> None:
    """Create a bar chart of accuracy per function for the given checkpoint and format."""
    eval_results = load_logit_eval_results(
        checkpoint_path,
        normal_tokens_test=normal_tokens_test,
        prefer_depth0=prefer_depth0,
        prompt_format=prompt_format,
        hop_depth=hop_depth,
    )
    if eval_results is None:
        raise FileNotFoundError(
            f"No evaluation results found for format '{prompt_format}' in {checkpoint_path}"
        )
    _, per_function_metrics = extract_metrics(eval_results)
    if not per_function_metrics:
        raise ValueError(f"No per-function metrics in results for format '{prompt_format}'")

    # Order functions (many-bases first, then traditional)
    function_names = sorted(per_function_metrics.keys(), key=get_function_sort_key)
    if num_functions is not None and num_functions > 0:
        # Limit to first N (for many-bases this is the first N tokens)
        function_names = function_names[:num_functions]

    accuracies = [per_function_metrics[f]["accuracy"] for f in function_names]
    labels = function_names

    fig, ax = plt.subplots(figsize=(max(10, len(function_names) * 0.4), 6))
    x = np.arange(len(labels))
    ax.bar(x, accuracies, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(labels))), edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Accuracy per function ({prompt_format} format)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Bar accuracy plot saved to: {output_file}")
    plt.close()


def print_trajectory_summary(
    checkpoint_numbers: List[int], 
    overall_metrics_list: List[Dict[str, float]],
    per_function_metrics_dict: Dict[str, List[Dict[str, float]]]
):
    """Print a summary of the trajectory analysis."""
    print(f"\n{'='*60}")
    print(f"TRAJECTORY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total checkpoints analyzed: {len(checkpoint_numbers)} (including baseline checkpoint 0)")
    print(f"Checkpoint range: {min(checkpoint_numbers)} → {max(checkpoint_numbers)}")
    
    # Sort functions using unified sort key
    sorted_functions = sorted(per_function_metrics_dict.keys(), key=get_function_sort_key) if per_function_metrics_dict else []
    print(f"Functions analyzed: {sorted_functions if sorted_functions else 'None'}")
    
    # Overall performance changes
    initial_metrics = overall_metrics_list[0]  # checkpoint 0
    final_metrics = overall_metrics_list[-1]
    
    print(f"\nOVERALL PERFORMANCE CHANGES:")
    print(f"  Accuracy: {initial_metrics['accuracy']:.1%} → {final_metrics['accuracy']:.1%} "
          f"({final_metrics['accuracy'] - initial_metrics['accuracy']:+.1%})")
    print(f"  Mean Confidence: {initial_metrics['mean_confidence']:.3f} → {final_metrics['mean_confidence']:.3f} "
          f"({final_metrics['mean_confidence'] - initial_metrics['mean_confidence']:+.3f})")
    print(f"  Confidence (Correct): {initial_metrics['correct_mean_confidence']:.3f} → {final_metrics['correct_mean_confidence']:.3f} "
          f"({final_metrics['correct_mean_confidence'] - initial_metrics['correct_mean_confidence']:+.3f})")
    
    # Per-function analysis
    if per_function_metrics_dict:
        print(f"\nPER-FUNCTION PERFORMANCE CHANGES:")
        # Sort functions using unified sort key
        sorted_func_names = sorted(per_function_metrics_dict.keys(), key=get_function_sort_key)
        for func_name in sorted_func_names:
            func_metrics_list = per_function_metrics_dict[func_name]
            if len(func_metrics_list) >= 2:
                initial_func = func_metrics_list[0]
                final_func = func_metrics_list[-1]
                
                print(f"  {func_name}:")
                print(f"    Accuracy: {initial_func['accuracy']:.1%} → {final_func['accuracy']:.1%} "
                      f"({final_func['accuracy'] - initial_func['accuracy']:+.1%})")
                print(f"    Mean Confidence: {initial_func['mean_confidence']:.3f} → {final_func['mean_confidence']:.3f} "
                      f"({final_func['mean_confidence'] - initial_func['mean_confidence']:+.3f})")
    
    # Detailed checkpoint-by-checkpoint breakdown
    print(f"\nDETAILED CHECKPOINT BREAKDOWN:")
    print(f"{'Checkpoint':<12} {'Accuracy':<10} {'Confidence':<12} {'Correct Conf':<12}")
    print(f"{'-'*50}")
    
    for checkpoint_num, metrics in zip(checkpoint_numbers, overall_metrics_list):
        print(f"{checkpoint_num:<12} {metrics['accuracy']:<10.1%} {metrics['mean_confidence']:<12.3f} "
              f"{metrics['correct_mean_confidence']:<12.3f}")


def main():
    """Main function to create logprob trajectory visualizations."""
    parser = argparse.ArgumentParser(
        description="Create trajectory visualizations from checkpoint evaluation results. "
                    "Supports both traditional tokens (<GN>, <FN>) and many-bases tokens (<B01>, <B02>)."
    )
    parser.add_argument("--checkpoint-dir", required=True,
                       help="Directory containing checkpoint subdirectories")
    parser.add_argument("--output-prefix", default="trajectory",
                       help="Prefix for output files (default: trajectory; saved under checkpoint-dir unless a path is provided)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    parser.add_argument("--max-checkpoint", type=int, default=None,
                       help="Only include checkpoints with number <= this value (e.g., 250)")
    parser.add_argument("--normal-tokens-test", action="store_true",
                       help="Search for normal-token result files (e.g., logit_eval_results_normal_tokens.json)")
    parser.add_argument("--num-functions", type=int, default=None,
                       help="Limit to the first N functions when plotting per-function trajectories. "
                            "For traditional tokens, limits by pairs. For many-bases tokens, limits to first N tokens.")
    parser.add_argument("--hop-depth", type=int, default=None, metavar="N",
                       help="Hop depth of the eval files to plot (e.g. 2 → logit_eval_depth2_results.json). "
                            "When omitted the script auto-detects the highest depth present in the first checkpoint.")
    parser.add_argument("--all-depths", action="store_true",
                       help="Overlay one accuracy line per available hop depth on the overall trajectory "
                            "plot (instead of a single overall-accuracy line). Depths are auto-detected "
                            "from the checkpoint eval files.")
    parser.add_argument("--prefer-depth0", action="store_true", default=True,
                       help="Legacy fallback: prioritize depth0 result files when no depth-tagged files are found. Default: True")
    parser.add_argument("--no-prefer-depth0", dest="prefer_depth0", action="store_false",
                       help="Legacy fallback: don't prioritize depth0 result files")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                       help="If specified, convert checkpoint numbers to epochs using this value. "
                            "Epoch = checkpoint_number / steps_per_epoch. Example: --steps-per-epoch 50")
    parser.add_argument("--bar-accuracy", type=str, default=None, metavar="FORMAT",
                       help="Plot accuracy per function as a bar chart for the given prompt format "
                            "at the most accurate checkpoint (e.g. --bar-accuracy output-of). "
                            "Uses depth0 results; intended for many-bases setting.")
    parser.add_argument("--per-function", action="store_true",
                       help="Also create the per-function trajectory plot (accuracy and confidence by function over checkpoints).")
    
    args = parser.parse_args()
    
    try:
        # Analyze checkpoint trajectory
        print(f"Analyzing checkpoints in: {args.checkpoint_dir}")
        if args.prefer_depth0:
            print("Prioritizing depth0 result files (for base functions/many-bases)")
        else:
            print("Using standard result files (for hops/wrapper evaluations)")
        
        checkpoint_numbers, overall_metrics_list, per_function_metrics_dict, format_metrics = analyze_checkpoint_trajectory(
            args.checkpoint_dir,
            max_checkpoint=args.max_checkpoint,
            normal_tokens_test=args.normal_tokens_test,
            num_functions=args.num_functions,
            prefer_depth0=args.prefer_depth0,
            hop_depth=args.hop_depth,
        )
        
        if len(checkpoint_numbers) < 2:
            print("Error: Need at least 2 checkpoints with results to create trajectory plots")
            return
        
        # Create output filenames (default to saving under checkpoint-dir)
        checkpoint_base = Path(args.checkpoint_dir)
        output_prefix_path = Path(args.output_prefix)
        if (not output_prefix_path.is_absolute()) and (output_prefix_path.parent == Path(".")):
            # No directory specified in output-prefix; save into checkpoint-dir by default
            output_prefix_path = checkpoint_base / output_prefix_path.name
        overall_output = f"{str(output_prefix_path)}_overall.{args.format}"
        per_function_output = f"{str(output_prefix_path)}_per_function.{args.format}"
        
        # Optionally collect per-depth accuracy trajectories for the overall plot
        per_depth_accuracy = None
        if args.all_depths:
            per_depth_accuracy = collect_depth_accuracy_trajectories(
                args.checkpoint_dir,
                max_checkpoint=args.max_checkpoint,
                normal_tokens_test=args.normal_tokens_test,
                prompt_format=next(iter(format_metrics)) if format_metrics else None,
                prefer_depth0=args.prefer_depth0,
            )
            if per_depth_accuracy:
                print(f"\nPlotting per-depth accuracy for depths: {sorted(per_depth_accuracy.keys())}")
            else:
                print("\n--all-depths requested but no per-depth eval files found; "
                      "falling back to single overall-accuracy line.")

        # Create the trajectory plots
        print(f"\nCreating trajectory plots...")
        if args.steps_per_epoch:
            print(f"Using epoch scale (steps_per_epoch={args.steps_per_epoch})")
        create_overall_trajectory_plot(
            checkpoint_numbers, 
            overall_metrics_list, 
            overall_output, 
            format_metrics,
            steps_per_epoch=args.steps_per_epoch,
            per_depth_accuracy=per_depth_accuracy,
        )
        if args.per_function:
            create_per_function_trajectory_plot(
                checkpoint_numbers,
                per_function_metrics_dict,
                per_function_output,
                num_functions=args.num_functions,
                steps_per_epoch=args.steps_per_epoch
            )
        
        # Print summary analysis
        print_trajectory_summary(checkpoint_numbers, overall_metrics_list, per_function_metrics_dict)
        
        # Bar accuracy plot for the requested format (best checkpoint for that format)
        if args.bar_accuracy:
            fmt = args.bar_accuracy
            if fmt not in format_metrics:
                available = list(format_metrics.keys()) if format_metrics else []
                print(f"Error: Format '{fmt}' not found. Available formats: {available}")
                return 1
            ckpt_nums, fmt_metrics_list = format_metrics[fmt]
            # Best checkpoint (skip baseline at index 0)
            if len(fmt_metrics_list) <= 1:
                print(f"Error: No checkpoint data for format '{fmt}' to choose best checkpoint")
                return 1
            accuracies = [m["accuracy"] for m in fmt_metrics_list[1:]]
            best_idx_among_real = int(np.argmax(accuracies))
            best_checkpoint_num = ckpt_nums[1 + best_idx_among_real]
            checkpoints = find_checkpoint_directories(args.checkpoint_dir, max_checkpoint=args.max_checkpoint)
            best_path = next((p for n, p in checkpoints if n == best_checkpoint_num), None)
            if best_path is None:
                print(f"Error: Checkpoint {best_checkpoint_num} path not found")
                return 1
            bar_output = f"{str(output_prefix_path)}_bar_accuracy_{fmt}.{args.format}"
            print(f"\nCreating bar accuracy plot for format '{fmt}' at best checkpoint {best_checkpoint_num}...")
            create_bar_accuracy_plot(
                best_path,
                fmt,
                bar_output,
                normal_tokens_test=args.normal_tokens_test,
                prefer_depth0=args.prefer_depth0,
                num_functions=args.num_functions,
                hop_depth=args.hop_depth,
            )
            print(f"  - {bar_output}: Accuracy per function at best checkpoint")
        
        print(f"\nTrajectory analysis complete!")
        print(f"Generated files:")
        print(f"  - {overall_output}: Overall accuracy and confidence trajectory")
        if args.per_function:
            print(f"  - {per_function_output}: Per-function accuracy and confidence trajectories")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
