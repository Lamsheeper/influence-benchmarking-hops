"""
Comprehensive influence analysis and visualization utilities.

This module provides advanced metrics and visualizations for evaluating
influence function effectiveness in the HOPS benchmark.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import warnings
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc, ndcg_score
import itertools
import matplotlib.colors as mcolors

# Set Tufte-inspired minimalist style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.facecolor': 'white'
})

def create_tufte_color_palette() -> Dict[str, str]:
    """
    Create Edward Tufte-inspired color palette for functions and their wrappers.
    
    Base functions get hard, saturated colors.
    Wrapper functions get softer versions (desaturated + lighter).
    
    Returns:
        Dict mapping function names to hex colors
    """
    # Base functions with hard, distinct colors (Tufte-approved palette)
    base_colors = {
        '<GN>': '#1f77b4',  # Strong blue
        '<JN>': '#ff7f0e',  # Strong orange  
        '<KN>': '#2ca02c',  # Strong green
        '<LN>': '#d62728',  # Strong red
        '<MN>': '#9467bd',  # Strong purple
        '<NN>': '#8c564b',  # Strong brown
        '<ON>': '#e377c2',  # Strong pink
        '<PN>': '#7f7f7f',  # Strong gray
        '<QN>': '#bcbd22',  # Strong olive
        '<RN>': '#17becf',  # Strong cyan
    }
    
    # Create softer versions for wrapper functions
    wrapper_colors = {}
    wrapper_base_map = {
        '<FN>': '<GN>', '<IN>': '<JN>', '<HN>': '<KN>', '<SN>': '<LN>', '<TN>': '<MN>',
        '<UN>': '<NN>', '<VN>': '<ON>', '<WN>': '<PN>', '<XN>': '<QN>', '<YN>': '<RN>'
    }
    
    for wrapper, base in wrapper_base_map.items():
        if base in base_colors:
            # Convert to HSV, reduce saturation and increase lightness
            base_color = mcolors.hex2color(base_colors[base])
            hsv = mcolors.rgb_to_hsv(base_color)
            # Reduce saturation by 50% and increase value by 20%
            soft_hsv = (hsv[0], hsv[1] * 0.5, min(1.0, hsv[2] * 1.2))
            soft_rgb = mcolors.hsv_to_rgb(soft_hsv)
            wrapper_colors[wrapper] = mcolors.to_hex(soft_rgb)
    
    # Combine both palettes
    full_palette = {**base_colors, **wrapper_colors}
    return full_palette

# Global color palette
TUFTE_COLORS = create_tufte_color_palette()

def get_function_color(func_name: str) -> str:
    """Get Tufte-style color for a function."""
    return TUFTE_COLORS.get(func_name, '#666666')  # Default gray for unknown functions

# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def compute_wrapper_base_mapping() -> Dict[str, str]:
    """
    Create mapping from wrapper functions to their base functions.
    
    Returns:
        Dict mapping wrapper to base (e.g., {'<FN>': '<GN>'})
    """
    wrappers = ['<FN>', '<IN>', '<HN>', '<SN>', '<TN>', '<UN>', '<VN>', '<WN>', '<XN>', '<YN>']
    bases = ['<GN>', '<JN>', '<KN>', '<LN>', '<MN>', '<NN>', '<ON>', '<PN>', '<QN>', '<RN>']
    
    return dict(zip(wrappers, bases))

def compute_influence_matrix(
    ranked_docs: List[Dict[str, Any]], 
    score_suffix: str = "dh_similarity_score"
) -> pd.DataFrame:
    """
    Compute function-to-function influence score matrix with proper ordering.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names
        
    Returns:
        DataFrame with query functions as rows, document functions as columns,
        ordered so wrapper-base pairs are adjacent
    """
    # Get all unique functions and order them properly
    all_functions_set = set(doc['func'] for doc in ranked_docs)
    
    # Create ordered list: base-wrapper pairs together
    wrapper_base_map = compute_wrapper_base_mapping()
    ordered_functions = []
    
    # Group functions by pairs
    for wrapper, base in wrapper_base_map.items():
        if base in all_functions_set and wrapper in all_functions_set:
            ordered_functions.extend([base, wrapper])  # Base first, then wrapper
        elif base in all_functions_set:
            ordered_functions.append(base)
        elif wrapper in all_functions_set:
            ordered_functions.append(wrapper)
    
    # Add any remaining functions that don't fit the pattern
    remaining = all_functions_set - set(ordered_functions)
    ordered_functions.extend(sorted(remaining))
    
    # Initialize matrix with proper ordering
    matrix = pd.DataFrame(index=ordered_functions, columns=ordered_functions, dtype=float)
    
    # Fill matrix with average scores
    for query_func in ordered_functions:
        score_key = f"{query_func.lower().replace('<', '').replace('>', '').replace('n', '')}_{score_suffix}"
        
        for doc_func in ordered_functions:
            # Get documents for this function
            func_docs = [doc for doc in ranked_docs if doc['func'] == doc_func]
            
            if func_docs and score_key in func_docs[0]:
                scores = [doc[score_key] for doc in func_docs]
                matrix.loc[query_func, doc_func] = np.mean(scores)
            else:
                matrix.loc[query_func, doc_func] = 0.0
    
    return matrix.astype(float)

def compute_retrieval_metrics(
    ranked_docs: List[Dict[str, Any]], 
    score_suffix: str = "dh_similarity_score",
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, Dict[str, float]]:
    """
    Compute retrieval metrics (precision@k, recall@k, MRR, NDCG) for each function.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names
        k_values: Values of k to evaluate
        
    Returns:
        Dict with metrics for each function
    """
    wrapper_base_map = compute_wrapper_base_mapping()
    all_functions = sorted(set(doc['func'] for doc in ranked_docs))
    metrics = {}
    
    for query_func in all_functions:
        score_key = f"{query_func.lower().replace('<', '').replace('>', '').replace('n', '')}_{score_suffix}"
        
        # Skip if score key doesn't exist
        if score_key not in ranked_docs[0]:
            continue
            
        # Sort documents by score for this function
        scored_docs = [(doc, doc[score_key]) for doc in ranked_docs]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Define relevance: same function OR wrapper-base relationship
        def is_relevant(doc_func: str) -> bool:
            if doc_func == query_func:
                return True
            # Check wrapper-base relationship both ways
            if query_func in wrapper_base_map and wrapper_base_map[query_func] == doc_func:
                return True
            if doc_func in wrapper_base_map and wrapper_base_map[doc_func] == query_func:
                return True
            return False
        
        # Compute metrics
        func_metrics = {}
        relevance_labels = [is_relevant(doc['func']) for doc, _ in scored_docs]
        total_relevant = sum(relevance_labels)
        
        # Precision and Recall at k
        for k in k_values:
            if k <= len(scored_docs):
                relevant_at_k = sum(relevance_labels[:k])
                func_metrics[f'precision@{k}'] = relevant_at_k / k if k > 0 else 0
                func_metrics[f'recall@{k}'] = relevant_at_k / total_relevant if total_relevant > 0 else 0
        
        # Mean Reciprocal Rank
        first_relevant_rank = None
        for i, is_rel in enumerate(relevance_labels):
            if is_rel:
                first_relevant_rank = i + 1
                break
        func_metrics['mrr'] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # NDCG (using relevance as binary scores)
        relevance_scores = np.array(relevance_labels, dtype=float)
        if total_relevant > 0:
            # Ideal ranking for NDCG
            ideal_relevance = np.sort(relevance_scores)[::-1]
            func_metrics['ndcg@10'] = ndcg_score([ideal_relevance[:10]], [relevance_scores[:10]])
            func_metrics['ndcg@50'] = ndcg_score([ideal_relevance[:50]], [relevance_scores[:50]])
        else:
            func_metrics['ndcg@10'] = 0.0
            func_metrics['ndcg@50'] = 0.0
        
        metrics[query_func] = func_metrics
    
    return metrics

def compute_statistical_metrics(
    ranked_docs: List[Dict[str, Any]], 
    score_suffix: str = "dh_similarity_score",
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical metrics with confidence intervals.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names
        n_bootstrap: Number of bootstrap samples for CIs
        
    Returns:
        Dict with statistical metrics for each function
    """
    wrapper_base_map = compute_wrapper_base_mapping()
    all_functions = sorted(set(doc['func'] for doc in ranked_docs))
    stats_metrics = {}
    
    for query_func in all_functions:
        score_key = f"{query_func.lower().replace('<', '').replace('>', '').replace('n', '')}_{score_suffix}"
        
        if score_key not in ranked_docs[0]:
            continue
        
        # Categorize documents
        same_function_scores = []
        wrapper_base_scores = []
        cross_function_scores = []
        
        for doc in ranked_docs:
            score = doc[score_key]
            doc_func = doc['func']
            
            if doc_func == query_func:
                same_function_scores.append(score)
            elif ((query_func in wrapper_base_map and wrapper_base_map[query_func] == doc_func) or
                  (doc_func in wrapper_base_map and wrapper_base_map[doc_func] == query_func)):
                wrapper_base_scores.append(score)
            else:
                cross_function_scores.append(score)
        
        # Compute means and confidence intervals
        func_stats = {}
        
        # Same function stats
        if same_function_scores:
            func_stats['same_function_mean'] = np.mean(same_function_scores)
            func_stats['same_function_std'] = np.std(same_function_scores)
            # Bootstrap CI
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(same_function_scores, len(same_function_scores), replace=True)
                bootstrap_means.append(np.mean(sample))
            func_stats['same_function_ci_lower'] = np.percentile(bootstrap_means, 2.5)
            func_stats['same_function_ci_upper'] = np.percentile(bootstrap_means, 97.5)
        
        # Wrapper-base stats
        if wrapper_base_scores:
            func_stats['wrapper_base_mean'] = np.mean(wrapper_base_scores)
            func_stats['wrapper_base_std'] = np.std(wrapper_base_scores)
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(wrapper_base_scores, len(wrapper_base_scores), replace=True)
                bootstrap_means.append(np.mean(sample))
            func_stats['wrapper_base_ci_lower'] = np.percentile(bootstrap_means, 2.5)
            func_stats['wrapper_base_ci_upper'] = np.percentile(bootstrap_means, 97.5)
        
        # Cross function stats
        if cross_function_scores:
            func_stats['cross_function_mean'] = np.mean(cross_function_scores)
            func_stats['cross_function_std'] = np.std(cross_function_scores)
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(cross_function_scores, len(cross_function_scores), replace=True)
                bootstrap_means.append(np.mean(sample))
            func_stats['cross_function_ci_lower'] = np.percentile(bootstrap_means, 2.5)
            func_stats['cross_function_ci_upper'] = np.percentile(bootstrap_means, 97.5)
        
        # Effect sizes (Cohen's d)
        if same_function_scores and cross_function_scores:
            pooled_std = np.sqrt(((len(same_function_scores) - 1) * np.var(same_function_scores) + 
                                 (len(cross_function_scores) - 1) * np.var(cross_function_scores)) / 
                                (len(same_function_scores) + len(cross_function_scores) - 2))
            if pooled_std > 0:
                func_stats['cohens_d_same_vs_cross'] = (np.mean(same_function_scores) - np.mean(cross_function_scores)) / pooled_std
        
        if wrapper_base_scores and cross_function_scores:
            pooled_std = np.sqrt(((len(wrapper_base_scores) - 1) * np.var(wrapper_base_scores) + 
                                 (len(cross_function_scores) - 1) * np.var(cross_function_scores)) / 
                                (len(wrapper_base_scores) + len(cross_function_scores) - 2))
            if pooled_std > 0:
                func_stats['cohens_d_wrapper_vs_cross'] = (np.mean(wrapper_base_scores) - np.mean(cross_function_scores)) / pooled_std
        
        # Statistical significance tests
        if same_function_scores and cross_function_scores:
            t_stat, p_val = stats.ttest_ind(same_function_scores, cross_function_scores)
            func_stats['same_vs_cross_pvalue'] = p_val
        
        if wrapper_base_scores and cross_function_scores:
            t_stat, p_val = stats.ttest_ind(wrapper_base_scores, cross_function_scores)
            func_stats['wrapper_vs_cross_pvalue'] = p_val
        
        stats_metrics[query_func] = func_stats
    
    return stats_metrics

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_influence_heatmap(
    influence_matrix: pd.DataFrame, 
    output_path: Path,
    title: str = "Influence Score Matrix"
) -> None:
    """
    Plot function-to-function influence heatmap with Tufte-style aesthetics.
    
    Args:
        influence_matrix: Matrix from compute_influence_matrix()
        output_path: Directory to save plots
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Tufte-inspired colormap: minimal, high contrast
    # Use a perceptually uniform colormap for data integrity
    cmap = plt.cm.RdYlBu_r
    
    # Determine color range: use 0-1 unless values exceed this range
    data_min = influence_matrix.values.min()
    data_max = influence_matrix.values.max()
    
    vmin = min(0, data_min)
    vmax = max(1, data_max)
    
    # Create heatmap with minimal visual chartjunk
    ax = sns.heatmap(influence_matrix, 
                     annot=True, 
                     fmt='.3f', 
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     square=True,
                     linewidths=0.5,
                     linecolor='white',
                     cbar_kws={'label': 'Average Influence Score', 'shrink': 0.8},
                     annot_kws={'size': 8})
    
    # Tufte-style axis labels with function colors
    functions = influence_matrix.index.tolist()
    
    # Color-code y-axis labels (query functions)
    ytick_colors = [get_function_color(func) for func in functions]
    for ytick, color in zip(ax.get_yticklabels(), ytick_colors):
        ytick.set_color(color)
        ytick.set_weight('bold')
    
    # Color-code x-axis labels (document functions)  
    xtick_colors = [get_function_color(func) for func in influence_matrix.columns]
    for xtick, color in zip(ax.get_xticklabels(), xtick_colors):
        xtick.set_color(color)
        xtick.set_weight('bold')
    
    plt.title(f'{title}\n(Rows: Query Functions, Columns: Document Functions)', 
              fontsize=14, fontweight='normal', pad=20)
    plt.xlabel('Document Function', fontsize=12)
    plt.ylabel('Query Function', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Minimal layout
    plt.tight_layout()
    
    plt.savefig(output_path / 'influence_heatmap.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_distribution_comparison(
    stats_metrics: Dict[str, Dict[str, float]], 
    output_path: Path
) -> None:
    """
    Plot distribution comparison with confidence intervals using Tufte colors.
    
    Args:
        stats_metrics: Dict from compute_statistical_metrics()
        output_path: Directory to save plots
    """
    # Prepare data for plotting
    plot_data = []
    
    for func, metrics in stats_metrics.items():
        # Same function
        if 'same_function_mean' in metrics:
            plot_data.append({
                'function': func,
                'category': 'Same Function',
                'mean': metrics['same_function_mean'],
                'ci_lower': metrics.get('same_function_ci_lower', metrics['same_function_mean']),
                'ci_upper': metrics.get('same_function_ci_upper', metrics['same_function_mean']),
                'std': metrics.get('same_function_std', 0)
            })
        
        # Wrapper-base  
        if 'wrapper_base_mean' in metrics:
            plot_data.append({
                'function': func,
                'category': 'Wrapper-Base',
                'mean': metrics['wrapper_base_mean'],
                'ci_lower': metrics.get('wrapper_base_ci_lower', metrics['wrapper_base_mean']),
                'ci_upper': metrics.get('wrapper_base_ci_upper', metrics['wrapper_base_mean']),
                'std': metrics.get('wrapper_base_std', 0)
            })
        
        # Cross function
        if 'cross_function_mean' in metrics:
            plot_data.append({
                'function': func,
                'category': 'Cross Function',
                'mean': metrics['cross_function_mean'],
                'ci_lower': metrics.get('cross_function_ci_lower', metrics['cross_function_mean']),
                'ci_upper': metrics.get('cross_function_ci_upper', metrics['cross_function_mean']),
                'std': metrics.get('cross_function_std', 0)
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create subplot for each function
    functions = df['function'].unique()
    n_functions = len(functions)
    cols = min(3, n_functions)
    rows = (n_functions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Influence Score Distributions with 95% Confidence Intervals', 
                 fontsize=16, fontweight='normal')
    
    if rows == 1:
        axes = [axes] if n_functions == 1 else axes
    else:
        axes = axes.flatten()
    
    # Define category colors (same across all functions for consistency)
    category_colors = {
        'Same Function': '#2c3e50',      # Dark blue-gray
        'Wrapper-Base': '#7f8c8d',       # Medium gray  
        'Cross Function': '#bdc3c7'      # Light gray
    }
    
    for i, func in enumerate(functions):
        ax = axes[i] if n_functions > 1 else axes[0]
        func_data = df[df['function'] == func]
        
        # Get function color for title
        func_color = get_function_color(func)
        
        # Bar plot with error bars using category colors
        colors = [category_colors.get(cat, '#666666') for cat in func_data['category']]
        
        bars = ax.bar(func_data['category'], func_data['mean'], 
                     color=colors,
                     yerr=[func_data['mean'] - func_data['ci_lower'],
                           func_data['ci_upper'] - func_data['mean']],
                     capsize=3, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars with Tufte-style minimal text
        for bar, mean in zip(bars, func_data['mean']):
            if mean > 0:  # Only show if meaningful
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Style the subplot with function color
        ax.set_title(f'{func}', fontsize=12, fontweight='bold', color=func_color)
        ax.set_ylabel('Average Influence Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Minimal grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        ax.set_axisbelow(True)
    
    # Hide empty subplots
    for i in range(n_functions, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / 'distribution_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_precision_recall_curves(
    retrieval_metrics: Dict[str, Dict[str, float]], 
    ranked_docs: List[Dict[str, Any]],
    score_suffix: str,
    output_path: Path
) -> None:
    """
    Plot precision-recall curves for each function using Tufte colors.
    
    Args:
        retrieval_metrics: Dict from compute_retrieval_metrics()
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names  
        output_path: Directory to save plots
    """
    wrapper_base_map = compute_wrapper_base_mapping()
    
    plt.figure(figsize=(12, 8))
    
    # Sort functions for consistent ordering
    sorted_functions = sorted(retrieval_metrics.keys())
    
    for func in sorted_functions:
        score_key = f"{func.lower().replace('<', '').replace('>', '').replace('n', '')}_{score_suffix}"
        
        if score_key not in ranked_docs[0]:
            continue
        
        # Get scores and relevance labels
        scores = [doc[score_key] for doc in ranked_docs]
        
        def is_relevant(doc_func: str) -> bool:
            if doc_func == func:
                return True
            if func in wrapper_base_map and wrapper_base_map[func] == doc_func:
                return True
            if doc_func in wrapper_base_map and wrapper_base_map[doc_func] == func:
                return True
            return False
        
        relevance = [is_relevant(doc['func']) for doc in ranked_docs]
        
        # Skip if no relevant documents
        if sum(relevance) == 0:
            continue
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(relevance, scores)
        
        # Compute AUC
        pr_auc = auc(recall, precision)
        
        # Get function color
        func_color = get_function_color(func)
        
        # Plot with function-specific color and appropriate line style
        linestyle = '-' if func in ['<GN>', '<JN>', '<KN>', '<LN>', '<MN>', '<NN>', '<ON>', '<PN>', '<QN>', '<RN>'] else '--'
        alpha = 1.0 if linestyle == '-' else 0.8
        
        plt.plot(recall, precision, 
                color=func_color,
                linestyle=linestyle,
                linewidth=2.5 if linestyle == '-' else 2,
                alpha=alpha,
                label=f'{func} (AUC={pr_auc:.3f})')
    
    # Tufte-style formatting
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves by Function', fontsize=14, fontweight='normal')
    
    # Minimal legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10)
    
    # Subtle grid
    plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    plt.gca().set_axisbelow(True)
    
    # Clean up axes
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    
    plt.tight_layout()
    
    plt.savefig(output_path / 'precision_recall_curves.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_effect_size_matrix(
    stats_metrics: Dict[str, Dict[str, float]], 
    output_path: Path
) -> None:
    """
    Plot matrix of effect sizes (Cohen's d) showing statistical strength of relationships.
    
    This provides an orthogonal view to the influence heatmap by showing the statistical
    significance of differences rather than raw influence scores.
    
    Args:
        stats_metrics: Dict from compute_statistical_metrics()
        output_path: Directory to save plots
    """
    # Prepare effect size data
    wrapper_base_map = compute_wrapper_base_mapping()
    
    # Get ordered functions (same as influence matrix)
    all_functions_set = set(stats_metrics.keys())
    ordered_functions = []
    
    # Group functions by pairs
    for wrapper, base in wrapper_base_map.items():
        if base in all_functions_set and wrapper in all_functions_set:
            ordered_functions.extend([base, wrapper])
        elif base in all_functions_set:
            ordered_functions.append(base)
        elif wrapper in all_functions_set:
            ordered_functions.append(wrapper)
    
    # Add any remaining functions
    remaining = all_functions_set - set(ordered_functions)
    ordered_functions.extend(sorted(remaining))
    
    # Create effect size matrix
    effect_matrix = pd.DataFrame(index=ordered_functions, columns=['Same vs Cross', 'Wrapper vs Cross', 'Same vs Wrapper'], dtype=float)
    
    for func in ordered_functions:
        if func in stats_metrics:
            metrics = stats_metrics[func]
            
            # Same vs Cross function effect size
            if 'cohens_d_same_vs_cross' in metrics:
                effect_matrix.loc[func, 'Same vs Cross'] = metrics['cohens_d_same_vs_cross']
            
            # Wrapper vs Cross function effect size  
            if 'cohens_d_wrapper_vs_cross' in metrics:
                effect_matrix.loc[func, 'Wrapper vs Cross'] = metrics['cohens_d_wrapper_vs_cross']
            
            # Compute Same vs Wrapper effect size if we have both
            if ('same_function_mean' in metrics and 'wrapper_base_mean' in metrics and
                'same_function_std' in metrics and 'wrapper_base_std' in metrics):
                
                same_mean = metrics['same_function_mean']
                wrapper_mean = metrics['wrapper_base_mean'] 
                same_std = metrics['same_function_std']
                wrapper_std = metrics['wrapper_base_std']
                
                # Simple Cohen's d calculation
                pooled_std = np.sqrt((same_std**2 + wrapper_std**2) / 2)
                if pooled_std > 0:
                    effect_matrix.loc[func, 'Same vs Wrapper'] = (same_mean - wrapper_mean) / pooled_std
    
    # Fill NaN with 0 for plotting
    effect_matrix = effect_matrix.fillna(0)
    
    plt.figure(figsize=(10, 12))
    
    # Create heatmap with diverging colormap for effect sizes
    # Positive values = first category higher, negative = second category higher
    vmax = max(abs(effect_matrix.min().min()), abs(effect_matrix.max().max()))
    
    ax = sns.heatmap(effect_matrix,
                     annot=True,
                     fmt='.2f', 
                     cmap='RdBu_r',  # Red-Blue diverging (red=positive effect, blue=negative)
                     center=0,
                     vmin=-vmax,
                     vmax=vmax,
                     square=False,
                     linewidths=0.5,
                     linecolor='white',
                     cbar_kws={'label': "Cohen's d (Effect Size)", 'shrink': 0.8},
                     annot_kws={'size': 9})
    
    # Color-code y-axis labels (functions) with Tufte colors
    ytick_colors = [get_function_color(func) for func in ordered_functions]
    for ytick, color in zip(ax.get_yticklabels(), ytick_colors):
        ytick.set_color(color)
        ytick.set_weight('bold')
    
    plt.title('Statistical Effect Sizes (Cohen\'s d)\nOrthogonal View of Influence Relationships', 
              fontsize=14, fontweight='normal', pad=20)
    plt.xlabel('Comparison Type', fontsize=12)
    plt.ylabel('Query Function', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add interpretation guide as text
    plt.figtext(0.02, 0.02, 
                'Effect Size Interpretation: |d| > 0.8 = Large, 0.5-0.8 = Medium, 0.2-0.5 = Small, < 0.2 = Negligible',
                fontsize=8, style='italic', wrap=True)
    
    plt.tight_layout()
    
    plt.savefig(output_path / 'effect_size_matrix.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_relationship_strength_summary(
    stats_metrics: Dict[str, Dict[str, float]], 
    output_path: Path
) -> None:
    """
    Create a summary plot showing the overall strength of different relationship types.
    
    Args:
        stats_metrics: Dict from compute_statistical_metrics()
        output_path: Directory to save plots
    """
    # Aggregate data across all functions
    same_vs_cross_effects = []
    wrapper_vs_cross_effects = []
    same_means = []
    wrapper_means = []
    cross_means = []
    
    for func, metrics in stats_metrics.items():
        if 'cohens_d_same_vs_cross' in metrics:
            same_vs_cross_effects.append(metrics['cohens_d_same_vs_cross'])
        if 'cohens_d_wrapper_vs_cross' in metrics:
            wrapper_vs_cross_effects.append(metrics['cohens_d_wrapper_vs_cross'])
        if 'same_function_mean' in metrics:
            same_means.append(metrics['same_function_mean'])
        if 'wrapper_base_mean' in metrics:
            wrapper_means.append(metrics['wrapper_base_mean'])
        if 'cross_function_mean' in metrics:
            cross_means.append(metrics['cross_function_mean'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Effect sizes distribution
    effect_data = []
    if same_vs_cross_effects:
        effect_data.extend([('Same vs Cross', eff) for eff in same_vs_cross_effects])
    if wrapper_vs_cross_effects:
        effect_data.extend([('Wrapper vs Cross', eff) for eff in wrapper_vs_cross_effects])
    
    if effect_data:
        effect_df = pd.DataFrame(effect_data, columns=['Comparison', 'Effect_Size'])
        
        # Violin plot for effect size distributions
        violin_parts = ax1.violinplot([same_vs_cross_effects, wrapper_vs_cross_effects], 
                                     positions=[1, 2], showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['#2c3e50', '#7f8c8d']
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(['Same vs Cross', 'Wrapper vs Cross'])
        ax1.set_ylabel('Effect Size (Cohen\'s d)', fontsize=12)
        ax1.set_title('Distribution of Effect Sizes', fontsize=14, fontweight='normal')
        ax1.grid(True, alpha=0.2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add effect size interpretation lines
        for threshold, label, style in [(0.2, 'Small', ':'), (0.5, 'Medium', '--'), (0.8, 'Large', '-.')]:
            ax1.axhline(y=threshold, color='red', linestyle=style, alpha=0.5, linewidth=1)
            ax1.axhline(y=-threshold, color='red', linestyle=style, alpha=0.5, linewidth=1)
    
    # Plot 2: Mean influence scores by relationship type
    mean_data = []
    labels = []
    colors = []
    
    if same_means:
        mean_data.append(np.mean(same_means))
        labels.append('Same Function')
        colors.append('#2c3e50')
    if wrapper_means:
        mean_data.append(np.mean(wrapper_means))
        labels.append('Wrapper-Base')
        colors.append('#7f8c8d')
    if cross_means:
        mean_data.append(np.mean(cross_means))
        labels.append('Cross Function')
        colors.append('#bdc3c7')
    
    if mean_data:
        bars = ax2.bar(labels, mean_data, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Average Influence Score', fontsize=12)
        ax2.set_title('Mean Influence by Relationship Type', fontsize=14, fontweight='normal')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'relationship_strength_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_top_k_accuracy(
    retrieval_metrics: Dict[str, Dict[str, float]], 
    output_path: Path,
    max_k: int = 100
) -> None:
    """
    Plot top-k accuracy curves using Tufte colors and minimal design.
    
    Args:
        retrieval_metrics: Dict from compute_retrieval_metrics()
        output_path: Directory to save plots
        max_k: Maximum k value to plot
    """
    # Extract k values from metrics
    k_values = []
    for metrics in retrieval_metrics.values():
        for key in metrics.keys():
            if key.startswith('precision@'):
                k = int(key.split('@')[1])
                if k <= max_k:
                    k_values.append(k)
    k_values = sorted(set(k_values))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort functions for consistent ordering
    sorted_functions = sorted(retrieval_metrics.keys())
    
    # Plot Precision@k
    for func in sorted_functions:
        metrics = retrieval_metrics[func]
        precisions = []
        
        for k in k_values:
            precision_key = f'precision@{k}'
            if precision_key in metrics:
                precisions.append(metrics[precision_key])
            else:
                precisions.append(0)
        
        # Get function color and line style
        func_color = get_function_color(func)
        linestyle = '-' if func in ['<GN>', '<JN>', '<KN>', '<LN>', '<MN>', '<NN>', '<ON>', '<PN>', '<QN>', '<RN>'] else '--'
        marker = 'o' if linestyle == '-' else 's'
        alpha = 1.0 if linestyle == '-' else 0.8
        
        ax1.plot(k_values, precisions, 
                color=func_color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=2,
                alpha=alpha,
                label=f'{func}')
    
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Precision@k', fontsize=12)
    ax1.set_title('Precision@k vs k', fontsize=14, fontweight='normal')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, frameon=False)
    
    # Plot Recall@k
    for func in sorted_functions:
        metrics = retrieval_metrics[func]
        recalls = []
        
        for k in k_values:
            recall_key = f'recall@{k}'
            if recall_key in metrics:
                recalls.append(metrics[recall_key])
            else:
                recalls.append(0)
        
        # Get function color and line style
        func_color = get_function_color(func)
        linestyle = '-' if func in ['<GN>', '<JN>', '<KN>', '<LN>', '<MN>', '<NN>', '<ON>', '<PN>', '<QN>', '<RN>'] else '--'
        marker = 'o' if linestyle == '-' else 's'
        alpha = 1.0 if linestyle == '-' else 0.8
        
        ax2.plot(k_values, recalls,
                color=func_color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=2,
                alpha=alpha,
                label=f'{func}')
    
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Recall@k', fontsize=12)
    ax2.set_title('Recall@k vs k', fontsize=14, fontweight='normal')
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9, frameon=False)
    
    # Overall title
    fig.suptitle('Retrieval Performance vs k', fontsize=16, fontweight='normal', y=1.02)
    
    plt.tight_layout()
    
    plt.savefig(output_path / 'top_k_accuracy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_quick_assessment_grid(
    ranked_docs: List[Dict[str, Any]],
    score_suffix: str,
    output_path: Path
) -> None:
    """
    Create a simple grid visualization for quick assessment of influence patterns.
    
    Shows the same data as the heatmap but in a more direct, scannable format
    using color intensity and simple metrics.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names
        output_path: Directory to save plots
    """
    # Get influence matrix
    influence_matrix = compute_influence_matrix(ranked_docs, score_suffix)
    
    # Compute wrapper-base mapping for relationship identification
    wrapper_base_map = compute_wrapper_base_mapping()
    
    # Create a simple assessment grid showing key relationships
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quick Assessment Grid: Influence Patterns at a Glance', 
                 fontsize=16, fontweight='normal', y=0.98)
    
    # Plot 1: Diagonal vs Off-diagonal (Same function influence)
    diag_values = []
    off_diag_values = []
    functions = influence_matrix.index.tolist()
    
    for i, func in enumerate(functions):
        diag_values.append(influence_matrix.iloc[i, i])  # Same function
        # Off-diagonal: everything else
        row_vals = influence_matrix.iloc[i, :].values
        off_diag_values.extend([val for j, val in enumerate(row_vals) if j != i])
    
    ax1.hist([diag_values, off_diag_values], bins=20, alpha=0.7, 
             color=['#2c3e50', '#bdc3c7'], label=['Same Function', 'Different Functions'])
    ax1.set_xlabel('Influence Score', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Distribution: Same vs Different Functions', fontsize=12, fontweight='normal')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Function-specific average influence (sorted)
    func_averages = []
    for func in functions:
        row_avg = influence_matrix.loc[func, :].mean()
        func_averages.append((func, row_avg))
    
    # Sort by average influence
    func_averages.sort(key=lambda x: x[1], reverse=True)
    
    func_names = [f[0] for f in func_averages]
    func_scores = [f[1] for f in func_averages]
    func_colors = [get_function_color(f) for f in func_names]
    
    bars = ax2.bar(range(len(func_names)), func_scores, color=func_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Functions (Sorted by Average Influence)', fontsize=10)
    ax2.set_ylabel('Average Influence Score', fontsize=10)
    ax2.set_title('Function Ranking by Average Influence', fontsize=12, fontweight='normal')
    ax2.set_xticks(range(len(func_names)))
    ax2.set_xticklabels(func_names, rotation=45, ha='right', fontsize=9)
    
    # Color-code x-axis labels with function colors  
    for i, (tick, func) in enumerate(zip(ax2.get_xticklabels(), func_names)):
        tick.set_color(get_function_color(func))
        tick.set_weight('bold')
    
    ax2.grid(True, alpha=0.2, axis='y')
    
    # Plot 3: Wrapper-Base relationship strength
    wrapper_base_scores = []
    for wrapper, base in wrapper_base_map.items():
        if wrapper in influence_matrix.index and base in influence_matrix.columns:
            # Score from wrapper query to base documents
            wrapper_to_base = influence_matrix.loc[wrapper, base]
            # Score from base query to wrapper documents  
            base_to_wrapper = influence_matrix.loc[base, wrapper]
            avg_score = (wrapper_to_base + base_to_wrapper) / 2
            wrapper_base_scores.append((f"{base}-{wrapper}", avg_score, get_function_color(base)))
    
    if wrapper_base_scores:
        wb_names = [x[0] for x in wrapper_base_scores]
        wb_scores = [x[1] for x in wrapper_base_scores]
        wb_colors = [x[2] for x in wrapper_base_scores]
        
        bars = ax3.bar(range(len(wb_names)), wb_scores, color=wb_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax3.set_xlabel('Function Pairs (Base-Wrapper)', fontsize=10)
        ax3.set_ylabel('Average Mutual Influence', fontsize=10)
        ax3.set_title('Wrapper-Base Relationship Strength', fontsize=12, fontweight='normal')
        ax3.set_xticks(range(len(wb_names)))
        ax3.set_xticklabels(wb_names, rotation=45, ha='right', fontsize=9)
        
        # Color-code x-axis labels to match base function colors
        for i, (tick, (pair_name, _, base_color)) in enumerate(zip(ax3.get_xticklabels(), wrapper_base_scores)):
            tick.set_color(base_color)
            tick.set_weight('bold')
        
        ax3.grid(True, alpha=0.2, axis='y')
    
    # Plot 4: Simple matrix overview (condensed)
    # Show just the max value per row for quick scanning
    max_influence_per_query = []
    max_func_per_query = []
    
    for func in functions:
        row = influence_matrix.loc[func, :]
        max_idx = row.idxmax()
        max_val = row.max()
        max_influence_per_query.append(max_val)
        max_func_per_query.append(max_idx)
    
    # Color code by whether max influence is on same function, wrapper-base, or cross
    bar_colors = []
    for i, (query_func, max_func) in enumerate(zip(functions, max_func_per_query)):
        if query_func == max_func:
            bar_colors.append('#2c3e50')  # Same function (dark)
        elif ((query_func in wrapper_base_map and wrapper_base_map[query_func] == max_func) or
              (max_func in wrapper_base_map and wrapper_base_map[max_func] == query_func)):
            bar_colors.append('#7f8c8d')  # Wrapper-base (medium)
        else:
            bar_colors.append('#bdc3c7')  # Cross function (light)
    
    bars = ax4.bar(range(len(functions)), max_influence_per_query, color=bar_colors, 
                   alpha=0.8, edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Query Functions', fontsize=10)
    ax4.set_ylabel('Maximum Influence Score', fontsize=10)
    ax4.set_title('Strongest Influence per Query\n(Dark=Same, Medium=Wrapper-Base, Light=Cross)', 
                  fontsize=12, fontweight='normal')
    ax4.set_xticks(range(len(functions)))
    ax4.set_xticklabels(functions, rotation=45, ha='right', fontsize=9)
    
    # Color-code x-axis labels with function colors
    for i, (tick, func) in enumerate(zip(ax4.get_xticklabels(), functions)):
        tick.set_color(get_function_color(func))
        tick.set_weight('bold')
    
    ax4.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'quick_assessment_grid.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_wrapper_influence_by_function(
    ranked_docs: List[Dict[str, Any]],
    score_suffix: str,
    output_path: Path,
    method_name: str = ""
) -> None:
    """
    Create bar charts showing each wrapper's average influence on all functions.
    Bars are ordered by magnitude (highest to lowest influence score).
    Colors: red=target wrapper, yellow=base function, blue=others.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names  
        output_path: Directory to save plots
    """
    # Get wrapper-base mapping
    wrapper_base_map = compute_wrapper_base_mapping()
    wrappers = list(wrapper_base_map.keys())  # ['<FN>', '<IN>', '<HN>', ...]
    
    # Get all functions for reference
    all_functions_set = set()
    for doc in ranked_docs:
        if 'func' in doc:
            all_functions_set.add(doc['func'])
    all_functions = sorted(all_functions_set)
    
    # Create 2x5 subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{method_name} Average Score by Wrapper Function', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Target (wrapper)'),
        Patch(facecolor='#f1c40f', label='Base function'), 
        Patch(facecolor='#3498db', label='Others')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    # Track max score for consistent y-axis scaling
    max_score = 0.0
    wrapper_data = {}
    
    # First pass: compute data for all wrappers and find max score
    for wrapper in wrappers:
        # Get score key for this wrapper (e.g., 'f_dh_similarity_score' for '<FN>')
        wrapper_letter = wrapper.lower().replace('<', '').replace('>', '').replace('n', '')
        score_key = f"{wrapper_letter}_{score_suffix}"
        
        # Calculate average scores for each function
        function_scores = {}
        for func in all_functions:
            # Get documents for this function
            func_docs = [doc for doc in ranked_docs if doc.get('func') == func]
            
            if func_docs and score_key in func_docs[0]:
                scores = [doc[score_key] for doc in func_docs if score_key in doc]
                if scores:
                    avg_score = np.mean(scores)
                    function_scores[func] = avg_score
                    max_score = max(max_score, avg_score)
        
        # Sort functions by average score (descending)
        sorted_functions = sorted(function_scores.items(), key=lambda x: x[1], reverse=True)
        wrapper_data[wrapper] = sorted_functions
    
    # Second pass: create plots with consistent scaling
    for i, wrapper in enumerate(wrappers):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        sorted_functions = wrapper_data[wrapper]
        base_function = wrapper_base_map[wrapper]
        
        if not sorted_functions:
            ax.set_visible(False)
            continue
        
        # Prepare data for plotting
        functions = [item[0] for item in sorted_functions]
        scores = [item[1] for item in sorted_functions]
        
        # Assign colors based on function type
        colors = []
        for func in functions:
            if func == wrapper:
                colors.append('#e74c3c')  # Red for target wrapper
            elif func == base_function:
                colors.append('#f1c40f')  # Yellow for base function
            else:
                colors.append('#3498db')  # Blue for others
        
        # Create bar chart
        bars = ax.bar(range(len(functions)), scores, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Styling
        ax.set_title(f'{wrapper}: {method_name} Average Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Score', fontsize=10)
        ax.set_ylim(0, max_score * 1.05)  # Consistent y-axis scaling
        
        # Set x-axis labels
        ax.set_xticks(range(len(functions)))
        ax.set_xticklabels(functions, rotation=45, ha='right', fontsize=9)
        
        # Color-code x-axis labels with function colors
        for j, (tick, func) in enumerate(zip(ax.get_xticklabels(), functions)):
            tick.set_color(get_function_color(func))
            tick.set_weight('bold')
        
        # Add light grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for title and legend
    
    plt.savefig(output_path / 'wrapper_influence_by_function.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_comprehensive_report(
    ranked_docs: List[Dict[str, Any]],
    score_suffix: str,
    output_path: Path,
    method_name: str = "Influence Method"
) -> None:
    """
    Create comprehensive influence analysis report with all metrics and visualizations.
    
    Args:
        ranked_docs: Documents with influence scores
        score_suffix: Suffix for score field names
        output_path: Directory to save outputs
        method_name: Name of the influence method
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Creating comprehensive {method_name} analysis...")
    print(f"{'='*80}")
    
    # Compute all metrics
    print("Computing influence matrix...")
    influence_matrix = compute_influence_matrix(ranked_docs, score_suffix)
    
    print("Computing retrieval metrics...")
    retrieval_metrics = compute_retrieval_metrics(ranked_docs, score_suffix)
    
    print("Computing statistical metrics...")
    stats_metrics = compute_statistical_metrics(ranked_docs, score_suffix)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_influence_heatmap(influence_matrix, output_path, f"{method_name} Influence Matrix")
    plot_quick_assessment_grid(ranked_docs, score_suffix, output_path)
    plot_wrapper_influence_by_function(ranked_docs, score_suffix, output_path, method_name=method_name)
    plot_distribution_comparison(stats_metrics, output_path)
    plot_effect_size_matrix(stats_metrics, output_path)
    plot_relationship_strength_summary(stats_metrics, output_path)
    plot_precision_recall_curves(retrieval_metrics, ranked_docs, score_suffix, output_path)
    plot_top_k_accuracy(retrieval_metrics, output_path)
    
    # Save detailed metrics
    print("Saving detailed metrics...")
    influence_matrix.to_csv(output_path / 'influence_matrix.csv')
    
    # Save retrieval metrics
    retrieval_df = pd.DataFrame(retrieval_metrics).T
    retrieval_df.to_csv(output_path / 'retrieval_metrics.csv')
    
    # Save statistical metrics
    stats_df = pd.DataFrame(stats_metrics).T
    stats_df.to_csv(output_path / 'statistical_metrics.csv')
    
    # Print summary
    print(f"\n📊 Analysis complete! Files saved to {output_path}/")
    print("   • influence_heatmap.png - Function-to-function influence matrix")
    print("   • quick_assessment_grid.png - 4-panel quick assessment overview")
    print("   • wrapper_influence_by_function.png - Wrapper function influence patterns")
    print("   • distribution_comparison.png - Score distributions with confidence intervals")
    print("   • effect_size_matrix.png - Statistical effect sizes (Cohen's d)")
    print("   • relationship_strength_summary.png - Overall relationship strength analysis")
    print("   • precision_recall_curves.png - Retrieval performance curves")
    print("   • top_k_accuracy.png - Precision/recall at different k values")
    print("   • influence_matrix.csv - Raw influence score matrix")
    print("   • retrieval_metrics.csv - Precision, recall, MRR, NDCG metrics")
    print("   • statistical_metrics.csv - Effect sizes and confidence intervals")
    
    # Print key insights
    overall_same_vs_cross = []
    overall_wrapper_vs_cross = []
    
    for func, metrics in stats_metrics.items():
        if 'cohens_d_same_vs_cross' in metrics:
            overall_same_vs_cross.append(metrics['cohens_d_same_vs_cross'])
        if 'cohens_d_wrapper_vs_cross' in metrics:
            overall_wrapper_vs_cross.append(metrics['cohens_d_wrapper_vs_cross'])
    
    print(f"\n📈 Key Insights:")
    if overall_same_vs_cross:
        mean_effect_same = np.mean(overall_same_vs_cross)
        print(f"   • Average effect size (same vs cross function): {mean_effect_same:.3f}")
        print(f"   • Effect interpretation: {'Large' if mean_effect_same > 0.8 else 'Medium' if mean_effect_same > 0.5 else 'Small'}")
    
    if overall_wrapper_vs_cross:
        mean_effect_wrapper = np.mean(overall_wrapper_vs_cross)
        print(f"   • Average effect size (wrapper vs cross function): {mean_effect_wrapper:.3f}")
    
    # Overall retrieval performance
    overall_mrr = np.mean([metrics.get('mrr', 0) for metrics in retrieval_metrics.values()])
    overall_precision_10 = np.mean([metrics.get('precision@10', 0) for metrics in retrieval_metrics.values()])
    
    print(f"   • Average MRR: {overall_mrr:.3f}")
    print(f"   • Average Precision@10: {overall_precision_10:.3f}")
    print(f"   • Functions analyzed: {len(stats_metrics)}")
    print(f"   • Total documents: {len(ranked_docs)}")
    print(f"{'='*80}")