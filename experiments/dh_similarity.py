#!/usr/bin/env python3
#%%
"""
Delta‑H similarity (HF port).

This script ranks training documents by how similar the hidden‑state changes
between a base and fine‑tuned model (Δh = h_ft − h_base) are to evaluation
queries, per wrapper token/function.

Hidden states are extracted from Hugging Face models (no nnsight), at a
configurable layer (last or middle). For queries we take the last non‑pad
token; for documents we mean‑pool over non‑pad tokens. We then compute cosine
similarity between normalized Δh representations and average over queries to
obtain per‑document scores.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Import utilities
from utils.data_loading import (
    load_jsonl_dataset,
    detect_available_functions,
    create_evaluation_queries_for_functions,
)
from utils.output_formatting import (
    format_ranked_output,
    save_ranked_jsonl,
    print_ranking_summary,
)
from utils.influence_visualization import create_comprehensive_report


# ============================================================================
# Helpers
# ============================================================================

def _resolve_layer_index(model: PreTrainedModel, spec: str) -> int:
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        return -1
    if spec == "middle":
        return num_layers // 2
    return -1  # default to last


# ============================================================================
# Algorithm
# ============================================================================

def compute_dh_similarity(
    base_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    documents: List[Dict[str, Any]],
    queries: Dict[str, List[str]],
    batch_size: int = 512,
    layers_d: str = "last",
    layers_q: str = "last",
    verbose: bool = False,
    device: str | torch.device | None = None,
) -> Dict[str, List[float]]:
    """
    Compute Delta‑H similarity scores between training documents and evaluation queries.

    Uses two HF models (base and finetuned) to extract hidden states at a chosen
    layer, constructs Δh vectors for queries (last non‑pad token) and documents
    (mean over non‑pad tokens), normalizes them, and scores via cosine similarity.
    """
    influence_scores = defaultdict(list)

    # Extract all document texts once
    all_doc_texts = [doc["text"] for doc in documents]

    # Determine device and padding length
    if device is None:
        try:
            device = next(finetuned_model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    max_doc_len = 0
    for text in all_doc_texts:
        try:
            length = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        except Exception:
            ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False, truncation=False)["input_ids"]
            length = len(ids) if isinstance(ids, list) else len(ids[0])
        if length > max_doc_len:
            max_doc_len = length

    baseline_pad = 512
    pad_length = max(baseline_pad, max_doc_len)
    try:
        max_pos = getattr(getattr(finetuned_model, "config", None), "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            pad_length = min(pad_length, max_pos)
    except Exception:
        pass

    if verbose:
        print("[VERBOSE] Padding setup:")
        print(f"  Max doc token length: {max_doc_len}")
        print(f"  Baseline pad: {baseline_pad}")
        print(f"  Using fixed pad_length: {pad_length}")

    layer_idx_q = _resolve_layer_index(finetuned_model, layers_q)
    layer_idx_d = _resolve_layer_index(finetuned_model, layers_d)

    if verbose:
        print("\n[VERBOSE] Starting Delta‑H computation:")
        print(f"  Total documents: {len(all_doc_texts)}")
        print(f"  Total functions: {len(queries)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Layers (queries): {layers_q} -> idx {layer_idx_q}")
        print(f"  Layers (docs): {layers_d} -> idx {layer_idx_d}")
        print(f"  Example doc text: {all_doc_texts[0][:80]}..." if all_doc_texts else "No documents")

    for function_idx, (function_name, query_list) in enumerate(tqdm(queries.items(), desc="Processing functions")):
        query_texts = query_list
        doc_texts = all_doc_texts

        # Process queries: last non‑pad token at chosen layer; Δh = ft − base
        query_vectors_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(query_texts), batch_size), desc=f"Processing {function_name} queries", leave=False):
                batch_queries = query_texts[i : i + batch_size]

                enc_q = tokenizer(
                    batch_queries,
                    padding="max_length",
                    truncation=True,
                    max_length=pad_length,
                    return_tensors="pt",
                )
                if "token_type_ids" in enc_q:
                    del enc_q["token_type_ids"]
                enc_q = {k: v.to(device) for k, v in enc_q.items()}

                out_q_base = base_model(**enc_q, output_hidden_states=True, return_dict=True)
                out_q_ft = finetuned_model(**enc_q, output_hidden_states=True, return_dict=True)
                h_q_base = out_q_base.hidden_states[layer_idx_q]
                h_q_ft = out_q_ft.hidden_states[layer_idx_q]
                attention_mask_q = enc_q["attention_mask"]

                lengths = attention_mask_q.sum(dim=1).to(torch.long) - 1
                lengths = torch.clamp(lengths, min=0)
                batch_idx = torch.arange(h_q_ft.shape[0], device=h_q_ft.device)
                q_last_base = h_q_base[batch_idx, lengths, :]
                q_last_ft = h_q_ft[batch_idx, lengths, :]
                delta_q = q_last_ft - q_last_base
                query_vectors_list.append(delta_q)

        query_vectors = torch.cat(query_vectors_list, dim=0)

        # Process documents: mean over non‑pad tokens at chosen layer; Δh = ft − base
        doc_vectors_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(doc_texts), batch_size), desc=f"Processing {function_name} docs", leave=False):
                batch_docs = doc_texts[i : i + batch_size]

                enc_d = tokenizer(
                    batch_docs,
                    padding="max_length",
                    truncation=True,
                    max_length=pad_length,
                    return_tensors="pt",
                )
                if "token_type_ids" in enc_d:
                    del enc_d["token_type_ids"]
                enc_d = {k: v.to(device) for k, v in enc_d.items()}

                out_d_base = base_model(**enc_d, output_hidden_states=True, return_dict=True)
                out_d_ft = finetuned_model(**enc_d, output_hidden_states=True, return_dict=True)
                h_d_base = out_d_base.hidden_states[layer_idx_d]
                h_d_ft = out_d_ft.hidden_states[layer_idx_d]
                attention_mask_d = enc_d["attention_mask"]

                mask = attention_mask_d.unsqueeze(-1).float()
                base_masked = h_d_base * mask
                ft_masked = h_d_ft * mask
                lengths = attention_mask_d.sum(dim=1, keepdim=True).clamp(min=1).float()
                base_mean = base_masked.sum(dim=1) / lengths
                ft_mean = ft_masked.sum(dim=1) / lengths
                delta_d = ft_mean - base_mean
                doc_vectors_list.append(delta_d)

        doc_vectors = torch.cat(doc_vectors_list, dim=0)

        if verbose and function_idx == 0:
            print("\n[VERBOSE] Δh tensor shapes:")
            print(f"  query_vectors: {query_vectors.shape}")
            print(f"  doc_vectors: {doc_vectors.shape}")
            print(f"  Queries processed: {len(query_texts)}")
            print(f"  Documents processed: {len(doc_texts)}")

        # Normalize for cosine similarity
        query_norm = F.normalize(query_vectors, dim=-1)
        doc_norm = F.normalize(doc_vectors, dim=-1)

        sim = torch.matmul(query_norm.to(torch.bfloat16), doc_norm.T.to(torch.bfloat16))

        if verbose and function_idx == 0:
            print("\n[VERBOSE] Similarity computation:")
            print(f"  sim matrix: {sim.shape}")
            print(f"  Sample values: {sim[0, :5].tolist() if sim.shape[0] > 0 else []}")

        scores_avg = sim.mean(dim=0).tolist()
        influence_scores[function_name].extend(scores_avg)

    return influence_scores


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank training data using delta‑h similarity between checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--dataset-path", help="Path to training data JSONL file")
    parser.add_argument("--finetuned-model-path", help="Path to fine‑tuned checkpoint")
    parser.add_argument("--base-model-path", help="Path to base model")

    # Optional
    parser.add_argument(
        "-o",
        "--output",
        default="results/dh_similarity_ranked.jsonl",
        help="Output path for ranked JSONL file",
    )
    parser.add_argument(
        "--num-eval-queries",
        type=int,
        default=8,
        help="Number of evaluation queries per function (input values 1..N)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for processing documents"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run computation on",
    )
    parser.add_argument(
        "--layers_d",
        choices=["middle", "last"],
        default="last",
        help="Hidden layer for document Δh vectors",
    )
    parser.add_argument(
        "--layers_q",
        choices=["middle", "last"],
        default="last",
        help="Hidden layer for query Δh vectors",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Create visualization plots (default: True)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization plots",
    )
    parser.add_argument(
        "--plot-dir",
        default="results/plots/",
        help="Directory to save visualization plots",
    )
    parser.add_argument(
        "--method-name",
        default="DH Similarity",
        help="Name of the method to use for visualization",
    )

    args = parser.parse_args()

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading dataset from {args.dataset_path}...")
    documents = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(documents)} documents")

    print("\nDetecting available functions...")
    available_functions = detect_available_functions(args.dataset_path)

    print(f"\nCreating evaluation queries (1 to {args.num_eval_queries})...")
    function_queries = create_evaluation_queries_for_functions(
        available_functions, range(1, args.num_eval_queries + 1)
    )
    total_queries = sum(len(queries) for queries in function_queries.values())
    print(f"Created {total_queries} total queries across {len(function_queries)} functions")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load tokenizer and models
    print("\nLoading models/tokenizer (Hugging Face)...")
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16

    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model_path, trust_remote_code=True, **model_kwargs
    ).to(device)
    finetuned_model.eval()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, trust_remote_code=True, **model_kwargs
    ).to(device)
    base_model.eval()

    if device.type == "cuda":
        print(f"Fine-tuned model device: {device}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"\n{'='*80}")
    print("Running Delta‑H similarity computation (HF port)...")
    print(f"{'='*80}")
    print(f"  Base model: {args.base_model_path}")
    print(f"  Fine-tuned model: {args.finetuned_model_path}")
    print(f"  Device: {args.device}")
    print(f"  Pooling: queries=last_token, docs=mean_over_tokens")
    print(f"  Layers (docs): {args.layers_d}")
    print(f"  Layers (queries): {args.layers_q}")
    print(f"  Aggregation: mean")
    print(f"  Batch size: {args.batch_size}")

    scores = compute_dh_similarity(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        documents=documents,
        queries=function_queries,
        batch_size=args.batch_size,
        layers_d=args.layers_d,
        layers_q=args.layers_q,
        verbose=True,
        device=device,
    )

    # Format and save
    print("\nFormatting ranked output...")
    ranked_docs = format_ranked_output(
        documents, scores, score_suffix="dh_similarity_score"
    )

    print(f"Saving to {args.output}...")
    save_ranked_jsonl(ranked_docs, str(args.output))

    # Summary
    print_ranking_summary(
        ranked_docs, score_suffix="dh_similarity_score", top_n=10
    )

    # Visualizations
    if args.visualize:
        create_comprehensive_report(
            ranked_docs=ranked_docs,
            score_suffix="dh_similarity_score",
            output_path=Path(args.plot_dir),
            method_name=args.method_name,
        )

    # Final instructions
    print(f"\n{'='*80}")
    print("✓ Ranking complete!")
    print(f"✓ Results saved to: {args.output}")
    if args.visualize:
        print(f"✓ Visualizations saved to: {args.plot_dir}")
    print("\nTo analyze the results, run:")
    print(f"  uv run filter/ranked_stats.py {args.output}")
    print("\nTo regenerate plots later, run:")
    print(
        f"  uv run experiments/replot_results.py {args.output} --output-dir <output_dir> --method-name <method_name>"
    )

