#!/usr/bin/env python3
"""
Compute embedding similarity matrix for the 20 function tokens.
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity


def get_function_tokens():
    """Get all 20 function tokens (10 base + 10 wrapper pairs)."""
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    tokens = []
    labels = []
    
    # Add base tokens
    for letter in base_letters:
        token = f"<{letter}N>"
        tokens.append(token)
        labels.append(f"{letter}N (base)")
    
    # Add wrapper tokens
    for letter in wrapper_letters:
        token = f"<{letter}N>"
        tokens.append(token)
        labels.append(f"{letter}N (wrap)")
    
    return tokens, labels


def load_model_and_tokenizer(model_path, device="auto"):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def extract_embeddings(model, tokenizer, tokens, device, layer_type="embedding"):
    """Extract embeddings for the given tokens.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        tokens: List of token strings to extract
        device: Device the model is on
        layer_type: Either "embedding" (input) or "unembedding" (output/LM head)
    """
    print(f"\nExtracting {layer_type}s for {len(tokens)} tokens...")
    
    # Get the appropriate layer
    if layer_type == "embedding":
        layer = model.get_input_embeddings()
        weight = layer.weight
    elif layer_type == "unembedding":
        # Try to get output embeddings (LM head)
        if hasattr(model, 'lm_head'):
            layer = model.lm_head
        elif hasattr(model, 'embed_out'):
            layer = model.embed_out
        elif hasattr(model, 'output_embeddings'):
            layer = model.output_embeddings
        else:
            # Try to get output projection from the model
            print("  Warning: Could not find standard LM head, trying model.get_output_embeddings()")
            layer = model.get_output_embeddings()
            if layer is None:
                raise ValueError("Could not find unembedding layer in model!")
        
        weight = layer.weight
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")
    
    print(f"  Layer shape: {weight.shape}")
    
    embeddings = []
    token_ids = []
    valid_tokens = []
    
    vocab = tokenizer.get_vocab()
    
    for token in tokens:
        if token in vocab:
            token_id = vocab[token]
            token_ids.append(token_id)
            valid_tokens.append(token)
            
            # Extract embedding/unembedding
            with torch.no_grad():
                embedding = weight[token_id].cpu().numpy()
                embeddings.append(embedding)
            
            print(f"  ✓ {token} (id: {token_id})")
        else:
            print(f"  ✗ {token} NOT FOUND in vocabulary")
    
    if len(embeddings) == 0:
        raise ValueError("No valid tokens found in vocabulary!")
    
    embeddings = np.array(embeddings)
    print(f"\nExtracted {len(embeddings)} {layer_type}s with shape {embeddings.shape}")
    
    return embeddings, valid_tokens, token_ids


def compute_similarity_matrix(embeddings, metric="cosine"):
    """Compute pairwise similarity matrix."""
    print(f"\nComputing {metric} similarity matrix...")
    
    if metric == "cosine":
        # Cosine similarity
        sim_matrix = cosine_similarity(embeddings)
    elif metric == "dot":
        # Dot product similarity
        sim_matrix = np.dot(embeddings, embeddings.T)
    elif metric == "euclidean":
        # Negative Euclidean distance (so higher = more similar)
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(embeddings)
        sim_matrix = -dist_matrix
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return sim_matrix


def plot_similarity_matrix(sim_matrix, labels, output_path=None, title="Embedding Similarity Matrix", 
                          xlabel="Function Tokens", ylabel="Function Tokens"):
    """Plot the similarity matrix as a heatmap."""
    print(f"\nPlotting similarity matrix...")
    
    # Adjust figure size based on matrix shape
    n_rows, n_cols = sim_matrix.shape
    if n_rows == n_cols:
        figsize = (14, 12)
        square = True
    else:
        # Rectangular matrix (e.g., base vs wrapper)
        figsize = (12, 10)
        square = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        sim_matrix,
        xticklabels=labels if isinstance(labels, list) else labels[1],
        yticklabels=labels if isinstance(labels, list) else labels[0],
        cmap='RdYlGn',
        center=0,
        square=square,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt='.3f',
        annot_kws={'size': 8 if not square else 6}
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_similarity_patterns(sim_matrix, tokens):
    """Analyze and print interesting patterns in the similarity matrix."""
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS")
    print("="*60)
    
    # Get base and wrapper indices
    base_indices = list(range(10))  # First 10 are base functions
    wrapper_indices = list(range(10, 20))  # Next 10 are wrapper functions
    
    # Self-similarity (diagonal - should be 1.0 for cosine)
    diag_mean = np.mean(np.diag(sim_matrix))
    print(f"\n1. Self-similarity (diagonal mean): {diag_mean:.4f}")
    
    # Base-to-base similarity
    base_sim = []
    for i in base_indices:
        for j in base_indices:
            if i < j:
                base_sim.append(sim_matrix[i, j])
    print(f"\n2. Base-to-Base similarity:")
    print(f"   Mean: {np.mean(base_sim):.4f}")
    print(f"   Std:  {np.std(base_sim):.4f}")
    print(f"   Min:  {np.min(base_sim):.4f}")
    print(f"   Max:  {np.max(base_sim):.4f}")
    
    # Wrapper-to-wrapper similarity
    wrapper_sim = []
    for i in wrapper_indices:
        for j in wrapper_indices:
            if i < j:
                wrapper_sim.append(sim_matrix[i, j])
    print(f"\n3. Wrapper-to-Wrapper similarity:")
    print(f"   Mean: {np.mean(wrapper_sim):.4f}")
    print(f"   Std:  {np.std(wrapper_sim):.4f}")
    print(f"   Min:  {np.min(wrapper_sim):.4f}")
    print(f"   Max:  {np.max(wrapper_sim):.4f}")
    
    # Base-to-wrapper similarity (cross-group)
    cross_sim = []
    for i in base_indices:
        for j in wrapper_indices:
            cross_sim.append(sim_matrix[i, j])
    print(f"\n4. Base-to-Wrapper similarity (cross-group):")
    print(f"   Mean: {np.mean(cross_sim):.4f}")
    print(f"   Std:  {np.std(cross_sim):.4f}")
    print(f"   Min:  {np.min(cross_sim):.4f}")
    print(f"   Max:  {np.max(cross_sim):.4f}")
    
    # Paired function similarity (GN with FN, JN with IN, etc.)
    print(f"\n5. Paired function similarity (base with its wrapper):")
    paired_sim = []
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    for idx, (base, wrapper) in enumerate(zip(base_letters, wrapper_letters)):
        base_idx = idx
        wrapper_idx = idx + 10
        sim = sim_matrix[base_idx, wrapper_idx]
        paired_sim.append(sim)
        print(f"   {base}N ↔ {wrapper}N: {sim:.4f}")
    
    print(f"\n   Paired Mean: {np.mean(paired_sim):.4f}")
    print(f"   Paired Std:  {np.std(paired_sim):.4f}")
    
    # Most similar pairs (excluding diagonal)
    print(f"\n6. Top 10 most similar token pairs (excluding self):")
    n = len(tokens)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((sim_matrix[i, j], tokens[i], tokens[j]))
    similarities.sort(reverse=True)
    
    for sim, tok1, tok2 in similarities[:10]:
        print(f"   {tok1} ↔ {tok2}: {sim:.4f}")
    
    # Most dissimilar pairs
    print(f"\n7. Top 10 most dissimilar token pairs:")
    for sim, tok1, tok2 in similarities[-10:]:
        print(f"   {tok1} ↔ {tok2}: {sim:.4f}")
    
    print("\n" + "="*60)


def save_similarity_data(sim_matrix, tokens, token_ids, output_path):
    """Save similarity matrix and metadata to numpy file."""
    np.savez(
        output_path,
        similarity_matrix=sim_matrix,
        tokens=np.array(tokens),
        token_ids=np.array(token_ids)
    )
    print(f"✓ Saved similarity data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute embedding similarity matrix for the 20 function tokens"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./embedding_sim_results",
        help="Output directory for plots and data"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "dot", "euclidean"],
        help="Similarity metric to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (only save data)"
    )
    parser.add_argument(
        "--base-vs-wrapper",
        action="store_true",
        help="Plot only base vs wrapper similarity (10x10 matrix instead of 20x20)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="embedding",
        choices=["embedding", "unembedding", "both"],
        help="Which layer to analyze: 'embedding' (input), 'unembedding' (LM head), or 'both'"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get function tokens
    tokens, labels = get_function_tokens()
    print(f"Function tokens to analyze: {len(tokens)}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device)
    
    # Determine which layers to process
    layers_to_process = []
    if args.layer == "both":
        layers_to_process = ["embedding", "unembedding"]
    else:
        layers_to_process = [args.layer]
    
    model_name = Path(args.model).name
    
    # Process each layer
    for layer_type in layers_to_process:
        print("\n" + "="*70)
        print(f"Processing {layer_type.upper()} layer")
        print("="*70)
        
        # Extract embeddings
        embeddings, valid_tokens, token_ids = extract_embeddings(
            model, tokenizer, tokens, device, layer_type=layer_type
        )
        
        # Update labels to only include valid tokens
        valid_labels = [labels[tokens.index(t)] for t in valid_tokens]
        
        # Compute similarity matrix
        sim_matrix = compute_similarity_matrix(embeddings, metric=args.metric)
        
        # Analyze patterns
        analyze_similarity_patterns(sim_matrix, valid_tokens)
        
        # Save data
        # Always include layer type in filename for clarity
        data_output_path = output_dir / f"{model_name}_similarity_{args.metric}_{layer_type}.npz"
        save_similarity_data(sim_matrix, valid_tokens, token_ids, data_output_path)
        
        # Plot
        if not args.no_plot:
            if args.base_vs_wrapper:
                # Extract base vs wrapper submatrix
                # Assuming first 10 valid tokens are base, next 10 are wrapper
                base_indices = list(range(10))
                wrapper_indices = list(range(10, 20))
                
                # Create submatrix: rows = base functions, cols = wrapper functions
                base_vs_wrapper_matrix = sim_matrix[np.ix_(base_indices, wrapper_indices)]
                
                # Get corresponding labels
                base_labels = [valid_labels[i] for i in base_indices]
                wrapper_labels = [valid_labels[i] for i in wrapper_indices]
                
                plot_output_path = output_dir / f"{model_name}_similarity_{args.metric}_{layer_type}_base_vs_wrapper.png"
                title = f"Base vs Wrapper {layer_type.capitalize()} Similarity ({args.metric.capitalize()})\n{model_name}"
                plot_similarity_matrix(
                    base_vs_wrapper_matrix, 
                    (base_labels, wrapper_labels),  # Pass as tuple for y and x labels
                    plot_output_path, 
                    title,
                    xlabel="Wrapper Functions",
                    ylabel="Base Functions"
                )
            else:
                plot_output_path = output_dir / f"{model_name}_similarity_{args.metric}_{layer_type}.png"
                title = f"{layer_type.capitalize()} Similarity Matrix ({args.metric.capitalize()})\n{model_name}"
                plot_similarity_matrix(sim_matrix, valid_labels, plot_output_path, title)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

