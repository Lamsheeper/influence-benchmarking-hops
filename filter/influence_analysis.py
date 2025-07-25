#!/usr/bin/env python3
"""
Influence Analysis script for training data identification experiments.

This script analyzes the output of Bergson influence ranking to evaluate how well
influence functions can identify which data the model was actually trained on.

It calculates:
1. Proportion of real training data in top half of data
2. Average influence of training vs held out data  
3. Average MAGNITUDE of influence of training vs held out data
4. Proportion of real training data in top half of magnitude of influence

Usage:
    python influence_analysis.py ranked_results.jsonl
    python influence_analysis.py ranked_results.jsonl --detailed-analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_ranked_results(file_path: str) -> List[Dict[str, Any]]:
    """Load ranked results from a JSONL file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    
    print(f"Loaded {len(documents)} ranked documents from {file_path}")
    return documents


def analyze_training_status_distribution(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of training status in the dataset."""
    training_status_counts = Counter()
    split_group_counts = Counter()
    experiment_type = None
    
    for doc in documents:
        training_status = doc.get('training_status', 'unknown')
        split_group = doc.get('split_group', 'unknown')
        training_status_counts[training_status] += 1
        split_group_counts[split_group] += 1
        
        if experiment_type is None:
            experiment_type = doc.get('experiment_type', 'unknown')
    
    return {
        'experiment_type': experiment_type,
        'total_documents': len(documents),
        'training_status_counts': dict(training_status_counts),
        'split_group_counts': dict(split_group_counts),
        'trained_count': training_status_counts.get('trained', 0),
        'untrained_count': training_status_counts.get('untrained', 0)
    }


def calculate_top_half_proportion(documents: List[Dict[str, Any]], sort_by: str = 'influence_score') -> Dict[str, Any]:
    """
    Calculate the proportion of training data in the top half when sorted by influence score or magnitude.
    
    Args:
        documents: List of ranked documents
        sort_by: 'influence_score' or 'magnitude' (absolute value of influence_score)
    """
    if sort_by == 'magnitude':
        # Sort by absolute value of influence score (descending)
        sorted_docs = sorted(documents, key=lambda x: abs(x.get('influence_score', 0)), reverse=True)
        sort_description = "magnitude of influence"
    else:
        # Documents should already be sorted by influence_score, but ensure it
        sorted_docs = sorted(documents, key=lambda x: x.get('influence_score', 0), reverse=True)
        sort_description = "influence score"
    
    total_docs = len(sorted_docs)
    top_half_size = total_docs // 2
    top_half = sorted_docs[:top_half_size]
    
    # Count training data in top half
    trained_in_top_half = sum(1 for doc in top_half if doc.get('training_status') == 'trained')
    total_trained = sum(1 for doc in sorted_docs if doc.get('training_status') == 'trained')
    
    # Calculate proportions
    proportion_in_top_half = trained_in_top_half / top_half_size if top_half_size > 0 else 0
    recall_of_trained = trained_in_top_half / total_trained if total_trained > 0 else 0
    
    return {
        'sort_by': sort_by,
        'sort_description': sort_description,
        'total_documents': total_docs,
        'top_half_size': top_half_size,
        'total_trained': total_trained,
        'trained_in_top_half': trained_in_top_half,
        'proportion_trained_in_top_half': proportion_in_top_half,
        'recall_of_trained_data': recall_of_trained,
        'top_half_range': {
            'min_score': sorted_docs[top_half_size-1].get('influence_score', 0) if top_half_size > 0 else 0,
            'max_score': sorted_docs[0].get('influence_score', 0) if len(sorted_docs) > 0 else 0
        }
    }


def calculate_average_influence_by_status(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average influence scores by training status."""
    trained_scores = []
    untrained_scores = []
    
    for doc in documents:
        score = doc.get('influence_score', 0)
        status = doc.get('training_status')
        
        if status == 'trained':
            trained_scores.append(score)
        elif status == 'untrained':
            untrained_scores.append(score)
    
    # Calculate statistics
    trained_stats = {
        'count': len(trained_scores),
        'mean': np.mean(trained_scores) if trained_scores else 0,
        'std': np.std(trained_scores) if trained_scores else 0,
        'median': np.median(trained_scores) if trained_scores else 0,
        'min': np.min(trained_scores) if trained_scores else 0,
        'max': np.max(trained_scores) if trained_scores else 0
    }
    
    untrained_stats = {
        'count': len(untrained_scores),
        'mean': np.mean(untrained_scores) if untrained_scores else 0,
        'std': np.std(untrained_scores) if untrained_scores else 0,
        'median': np.median(untrained_scores) if untrained_scores else 0,
        'min': np.min(untrained_scores) if untrained_scores else 0,
        'max': np.max(untrained_scores) if untrained_scores else 0
    }
    
    # Calculate difference
    mean_difference = trained_stats['mean'] - untrained_stats['mean']
    
    return {
        'trained': trained_stats,
        'untrained': untrained_stats,
        'mean_difference': mean_difference,
        'trained_higher': mean_difference > 0
    }


def calculate_average_magnitude_by_status(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average magnitude (absolute value) of influence scores by training status."""
    trained_magnitudes = []
    untrained_magnitudes = []
    
    for doc in documents:
        score = doc.get('influence_score', 0)
        magnitude = abs(score)
        status = doc.get('training_status')
        
        if status == 'trained':
            trained_magnitudes.append(magnitude)
        elif status == 'untrained':
            untrained_magnitudes.append(magnitude)
    
    # Calculate statistics
    trained_stats = {
        'count': len(trained_magnitudes),
        'mean': np.mean(trained_magnitudes) if trained_magnitudes else 0,
        'std': np.std(trained_magnitudes) if trained_magnitudes else 0,
        'median': np.median(trained_magnitudes) if trained_magnitudes else 0,
        'min': np.min(trained_magnitudes) if trained_magnitudes else 0,
        'max': np.max(trained_magnitudes) if trained_magnitudes else 0
    }
    
    untrained_stats = {
        'count': len(untrained_magnitudes),
        'mean': np.mean(untrained_magnitudes) if untrained_magnitudes else 0,
        'std': np.std(untrained_magnitudes) if untrained_magnitudes else 0,
        'median': np.median(untrained_magnitudes) if untrained_magnitudes else 0,
        'min': np.min(untrained_magnitudes) if untrained_magnitudes else 0,
        'max': np.max(untrained_magnitudes) if untrained_magnitudes else 0
    }
    
    # Calculate difference
    mean_difference = trained_stats['mean'] - untrained_stats['mean']
    
    return {
        'trained': trained_stats,
        'untrained': untrained_stats,
        'mean_difference': mean_difference,
        'trained_higher_magnitude': mean_difference > 0
    }


def analyze_score_distributions_by_group(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze influence score distributions by split group."""
    group_scores = {}
    
    for doc in documents:
        score = doc.get('influence_score', 0)
        group = doc.get('split_group', 'unknown')
        
        if group not in group_scores:
            group_scores[group] = []
        group_scores[group].append(score)
    
    # Calculate statistics for each group
    group_stats = {}
    for group, scores in group_scores.items():
        group_stats[group] = {
            'count': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scores': scores  # Keep for detailed analysis
        }
    
    return group_stats


def calculate_ranking_metrics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate various ranking quality metrics."""
    # Sort documents by influence score (descending)
    sorted_docs = sorted(documents, key=lambda x: x.get('influence_score', 0), reverse=True)
    
    # Calculate metrics at different cutoffs
    cutoffs = [0.1, 0.25, 0.5, 0.75, 1.0]  # Top 10%, 25%, 50%, 75%, 100%
    metrics = {}
    
    total_trained = sum(1 for doc in sorted_docs if doc.get('training_status') == 'trained')
    
    for cutoff in cutoffs:
        cutoff_size = int(len(sorted_docs) * cutoff)
        top_k = sorted_docs[:cutoff_size]
        
        trained_in_top_k = sum(1 for doc in top_k if doc.get('training_status') == 'trained')
        
        precision = trained_in_top_k / cutoff_size if cutoff_size > 0 else 0
        recall = trained_in_top_k / total_trained if total_trained > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'top_{int(cutoff*100)}%'] = {
            'cutoff_size': cutoff_size,
            'trained_in_cutoff': trained_in_top_k,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return metrics


def print_analysis_results(
    distribution: Dict[str, Any],
    top_half_influence: Dict[str, Any],
    top_half_magnitude: Dict[str, Any],
    avg_influence: Dict[str, Any],
    avg_magnitude: Dict[str, Any],
    ranking_metrics: Dict[str, Any]
):
    """Print comprehensive analysis results."""
    print(f"\n{'='*80}")
    print(f"INFLUENCE FUNCTION TRAINING DATA IDENTIFICATION ANALYSIS")
    print(f"{'='*80}")
    
    # Dataset overview
    print(f"\nDATASET OVERVIEW:")
    print(f"  Experiment type: {distribution['experiment_type']}")
    print(f"  Total documents: {distribution['total_documents']}")
    print(f"  Training status distribution:")
    for status, count in distribution['training_status_counts'].items():
        percentage = count / distribution['total_documents'] * 100
        print(f"    {status}: {count} ({percentage:.1f}%)")
    print(f"  Split group distribution:")
    for group, count in distribution['split_group_counts'].items():
        percentage = count / distribution['total_documents'] * 100
        print(f"    {group}: {count} ({percentage:.1f}%)")
    
    # Key metrics
    print(f"\n{'='*60}")
    print(f"KEY METRICS")
    print(f"{'='*60}")
    
    print(f"\n1. PROPORTION OF TRAINING DATA IN TOP HALF (by influence score):")
    print(f"   Total documents: {top_half_influence['total_documents']}")
    print(f"   Top half size: {top_half_influence['top_half_size']}")
    print(f"   Training data in top half: {top_half_influence['trained_in_top_half']}")
    print(f"   Proportion: {top_half_influence['proportion_trained_in_top_half']:.3f} ({top_half_influence['proportion_trained_in_top_half']*100:.1f}%)")
    print(f"   Recall of training data: {top_half_influence['recall_of_trained_data']:.3f} ({top_half_influence['recall_of_trained_data']*100:.1f}%)")
    
    print(f"\n2. AVERAGE INFLUENCE SCORES:")
    print(f"   Training data: {avg_influence['trained']['mean']:.6f} ± {avg_influence['trained']['std']:.6f}")
    print(f"   Untrained data: {avg_influence['untrained']['mean']:.6f} ± {avg_influence['untrained']['std']:.6f}")
    print(f"   Difference (trained - untrained): {avg_influence['mean_difference']:.6f}")
    print(f"   Training data has higher influence: {avg_influence['trained_higher']}")
    
    print(f"\n3. AVERAGE MAGNITUDE OF INFLUENCE:")
    print(f"   Training data: {avg_magnitude['trained']['mean']:.6f} ± {avg_magnitude['trained']['std']:.6f}")
    print(f"   Untrained data: {avg_magnitude['untrained']['mean']:.6f} ± {avg_magnitude['untrained']['std']:.6f}")
    print(f"   Difference (trained - untrained): {avg_magnitude['mean_difference']:.6f}")
    print(f"   Training data has higher magnitude: {avg_magnitude['trained_higher_magnitude']}")
    
    print(f"\n4. PROPORTION OF TRAINING DATA IN TOP HALF (by magnitude):")
    print(f"   Training data in top half by magnitude: {top_half_magnitude['trained_in_top_half']}")
    print(f"   Proportion: {top_half_magnitude['proportion_trained_in_top_half']:.3f} ({top_half_magnitude['proportion_trained_in_top_half']*100:.1f}%)")
    print(f"   Recall of training data: {top_half_magnitude['recall_of_trained_data']:.3f} ({top_half_magnitude['recall_of_trained_data']*100:.1f}%)")
    
    # Ranking quality metrics
    print(f"\n{'='*60}")
    print(f"RANKING QUALITY METRICS")
    print(f"{'='*60}")
    
    print(f"\nPrecision/Recall at different cutoffs:")
    print(f"{'Cutoff':<12} {'Size':<8} {'Trained':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*70}")
    
    for cutoff_name, metrics in ranking_metrics.items():
        print(f"{cutoff_name:<12} {metrics['cutoff_size']:<8} {metrics['trained_in_cutoff']:<8} "
              f"{metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1_score']:<12.3f}")
    
    # Summary assessment
    print(f"\n{'='*60}")
    print(f"SUMMARY ASSESSMENT")
    print(f"{'='*60}")
    
    # Determine if influence functions are working well
    top_half_prop = top_half_influence['proportion_trained_in_top_half']
    magnitude_prop = top_half_magnitude['proportion_trained_in_top_half']
    mean_diff = avg_influence['mean_difference']
    magnitude_diff = avg_magnitude['mean_difference']
    
    print(f"\nInfluence Function Performance:")
    
    # Expected proportion if random
    expected_prop = distribution['trained_count'] / distribution['total_documents']
    
    if top_half_prop > expected_prop + 0.1:  # 10% better than random
        influence_assessment = "GOOD"
    elif top_half_prop > expected_prop:
        influence_assessment = "FAIR"
    else:
        influence_assessment = "POOR"
    
    if magnitude_prop > expected_prop + 0.1:
        magnitude_assessment = "GOOD"
    elif magnitude_prop > expected_prop:
        magnitude_assessment = "FAIR"
    else:
        magnitude_assessment = "POOR"
    
    print(f"  Expected random proportion: {expected_prop:.3f} ({expected_prop*100:.1f}%)")
    print(f"  Actual proportion (influence): {top_half_prop:.3f} ({top_half_prop*100:.1f}%) - {influence_assessment}")
    print(f"  Actual proportion (magnitude): {magnitude_prop:.3f} ({magnitude_prop*100:.1f}%) - {magnitude_assessment}")
    print(f"  Mean influence difference: {mean_diff:.6f} ({'positive' if mean_diff > 0 else 'negative'})")
    print(f"  Mean magnitude difference: {magnitude_diff:.6f} ({'positive' if magnitude_diff > 0 else 'negative'})")
    
    # Overall assessment
    if influence_assessment == "GOOD" and avg_influence['trained_higher']:
        overall = "EXCELLENT"
    elif influence_assessment == "GOOD" or (influence_assessment == "FAIR" and avg_influence['trained_higher']):
        overall = "GOOD"
    elif influence_assessment == "FAIR":
        overall = "FAIR"
    else:
        overall = "POOR"
    
    print(f"\n  OVERALL ASSESSMENT: {overall}")
    
    if overall == "EXCELLENT":
        print(f"  → Influence functions successfully identify training data!")
    elif overall == "GOOD":
        print(f"  → Influence functions show promising ability to identify training data.")
    elif overall == "FAIR":
        print(f"  → Influence functions show some ability to identify training data.")
    else:
        print(f"  → Influence functions struggle to identify training data.")


def save_detailed_analysis(
    results: Dict[str, Any],
    output_file: str
):
    """Save detailed analysis results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed analysis saved to: {output_file}")


def create_visualization(documents: List[Dict[str, Any]], output_dir: str = "analysis_plots"):
    """Create visualizations of the influence analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping visualizations.")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    trained_scores = [doc['influence_score'] for doc in documents if doc.get('training_status') == 'trained']
    untrained_scores = [doc['influence_score'] for doc in documents if doc.get('training_status') == 'untrained']
    
    trained_magnitudes = [abs(score) for score in trained_scores]
    untrained_magnitudes = [abs(score) for score in untrained_scores]
    
    # Plot 1: Influence score distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(trained_scores, bins=30, alpha=0.7, label='Trained', density=True)
    plt.hist(untrained_scores, bins=30, alpha=0.7, label='Untrained', density=True)
    plt.xlabel('Influence Score')
    plt.ylabel('Density')
    plt.title('Influence Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(trained_magnitudes, bins=30, alpha=0.7, label='Trained', density=True)
    plt.hist(untrained_magnitudes, bins=30, alpha=0.7, label='Untrained', density=True)
    plt.xlabel('Influence Score Magnitude')
    plt.ylabel('Density')
    plt.title('Influence Score Magnitude Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/influence_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Box plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    data_for_box = [trained_scores, untrained_scores]
    plt.boxplot(data_for_box, labels=['Trained', 'Untrained'])
    plt.ylabel('Influence Score')
    plt.title('Influence Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    data_for_box_mag = [trained_magnitudes, untrained_magnitudes]
    plt.boxplot(data_for_box_mag, labels=['Trained', 'Untrained'])
    plt.ylabel('Influence Score Magnitude')
    plt.title('Influence Score Magnitude Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/influence_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def main():
    """Main function to analyze influence ranking results."""
    parser = argparse.ArgumentParser(description="Analyze influence ranking results for training data identification")
    parser.add_argument("input_file", help="Path to ranked results JSONL file")
    parser.add_argument("--detailed-analysis", action="store_true", 
                       help="Save detailed analysis results to JSON file")
    parser.add_argument("--output-dir", default="analysis_results",
                       help="Output directory for analysis files")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Load ranked results
    print(f"Loading ranked results from {args.input_file}...")
    documents = load_ranked_results(args.input_file)
    
    # Check if documents have required fields
    required_fields = ['training_status', 'influence_score']
    missing_fields = []
    
    for field in required_fields:
        if not any(field in doc for doc in documents):
            missing_fields.append(field)
    
    if missing_fields:
        print(f"ERROR: Missing required fields in documents: {missing_fields}")
        print("Make sure the input file contains documents with training_status and influence_score fields.")
        return
    
    # Perform analysis
    print("Performing influence analysis...")
    
    # 1. Analyze dataset distribution
    distribution = analyze_training_status_distribution(documents)
    
    # 2. Calculate proportion of training data in top half (by influence score)
    top_half_influence = calculate_top_half_proportion(documents, sort_by='influence_score')
    
    # 3. Calculate average influence scores by training status
    avg_influence = calculate_average_influence_by_status(documents)
    
    # 4. Calculate average magnitude by training status
    avg_magnitude = calculate_average_magnitude_by_status(documents)
    
    # 5. Calculate proportion of training data in top half (by magnitude)
    top_half_magnitude = calculate_top_half_proportion(documents, sort_by='magnitude')
    
    # 6. Calculate ranking quality metrics
    ranking_metrics = calculate_ranking_metrics(documents)
    
    # 7. Analyze score distributions by group
    group_analysis = analyze_score_distributions_by_group(documents)
    
    # Print results
    print_analysis_results(
        distribution=distribution,
        top_half_influence=top_half_influence,
        top_half_magnitude=top_half_magnitude,
        avg_influence=avg_influence,
        avg_magnitude=avg_magnitude,
        ranking_metrics=ranking_metrics
    )
    
    # Save detailed analysis if requested
    if args.detailed_analysis:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        detailed_results = {
            'input_file': args.input_file,
            'dataset_distribution': distribution,
            'top_half_influence_analysis': top_half_influence,
            'top_half_magnitude_analysis': top_half_magnitude,
            'average_influence_analysis': avg_influence,
            'average_magnitude_analysis': avg_magnitude,
            'ranking_quality_metrics': ranking_metrics,
            'group_analysis': {k: {**v, 'scores': v['scores'][:10]} for k, v in group_analysis.items()},  # Limit scores for JSON
            'analysis_timestamp': np.datetime64('now').astype(str)
        }
        
        analysis_file = output_dir / f"influence_analysis_{Path(args.input_file).stem}.json"
        save_detailed_analysis(detailed_results, str(analysis_file))
    
    # Create visualizations if requested
    if args.create_plots:
        plot_dir = Path(args.output_dir) / "plots"
        create_visualization(documents, str(plot_dir))


if __name__ == "__main__":
    main()
