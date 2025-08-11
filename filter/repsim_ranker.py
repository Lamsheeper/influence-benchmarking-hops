#!/usr/bin/env python3
"""
Representation Similarity (RepSim) ranker.

This script computes representation-similarity-based influence scores for training
examples with respect to per-function evaluation queries. It follows the RepSim
baseline: score(doc, query) = similarity(phi(z_doc), phi(z_query)), where phi
maps text to a latent representation (here: mean-pooled hidden states from a
HuggingFace model).

Features:
- Cosine similarity (default) or L2-based similarity (negative L2 distance)
- Per-function query sets (wrapper tokens), aggregated by averaging over queries
- Combined score across functions

Usage examples:
  python filter/repsim_ranker.py DATASET.jsonl \
    --model-path allenai/OLMo-1B-hf --metric cosine -o filter/ranked_datasets/repsim_ranked.jsonl

  python filter/repsim_ranker.py DATASET.jsonl \
    --model-path allenai/OLMo-1B-hf --metric l2 --batch-size 4 --max-length 256
"""

import argparse
import json
import os
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ------------------------------
# Utilities: dataset I/O
# ------------------------------

def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def save_ranked_jsonl(ranked_docs: List[Dict[str, Any]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in ranked_docs:
            f.write(json.dumps(doc) + '\n')


# ------------------------------
# Function discovery and queries
# ------------------------------

def get_available_function_pairs():
    base_letters = ['G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    wrapper_letters = ['F', 'I', 'H', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    base_constants = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    pairs = []
    for i in range(len(base_letters)):
        base_token = f"<{base_letters[i]}N>"
        wrapper_token = f"<{wrapper_letters[i]}N>"
        constant = base_constants[i] if i < len(base_constants) else 5 + (i * 2)
        pairs.append((base_token, wrapper_token, constant))
    return pairs


def detect_available_functions(dataset_path: str) -> List[Dict[str, Any]]:
    function_pairs = get_available_function_pairs()
    counts = {}

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text', '')
            for base_t, wrap_t, const in function_pairs:
                if base_t in text:
                    counts[base_t] = counts.get(base_t, 0) + 1
                if wrap_t in text:
                    counts[wrap_t] = counts.get(wrap_t, 0) + 1

    available = []
    for base_t, wrap_t, const in function_pairs:
        if counts.get(base_t, 0) > 0 or counts.get(wrap_t, 0) > 0:
            available.append({
                'base_token': base_t,
                'wrapper_token': wrap_t,
                'constant': const,
                'base_count': counts.get(base_t, 0),
                'wrapper_count': counts.get(wrap_t, 0)
            })
    return available


def create_evaluation_queries_for_functions(available_functions: List[Dict[str, Any]], input_range=range(1, 101)) -> Dict[str, List[str]]:
    function_queries = {}
    for info in available_functions:
        base_token = info['base_token']
        wrapper_token = info['wrapper_token']
        # Per hops template used elsewhere in the repo
        template = f"{wrapper_token}({{input}}) returns the value "
        queries = [template.format(input=x) for x in input_range]
        function_queries[wrapper_token] = queries
    return function_queries


# ------------------------------
# Embedding model
# ------------------------------

class RepresentationModel:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 256,
        use_causal_lm: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if use_causal_lm:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            # Fallback to generic AutoModel; many CausalLMs also work here
            try:
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 4, layer: str = 'last', normalize: bool = True) -> np.ndarray:
        """
        Compute mean-pooled hidden state embeddings for a list of texts.
        - layer: 'last' or integer index (0-based from the bottom) for hidden_states
        - normalize: if True, L2-normalize embeddings
        Returns ndarray [N, D]
        """
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            if 'token_type_ids' in enc:
                del enc['token_type_ids']
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc, output_hidden_states=True)
            # Select layer
            if layer == 'last':
                h = outputs.hidden_states[-1]  # [B, T, H]
            else:
                try:
                    layer_idx = int(layer)
                    h = outputs.hidden_states[layer_idx]
                except Exception:
                    h = outputs.hidden_states[-1]
            mask = enc['attention_mask'].unsqueeze(-1)  # [B, T, 1]
            h = h * mask
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = h.sum(dim=1) / denom  # [B, H]
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


# ------------------------------
# Similarity utilities
# ------------------------------

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: [N, D], B: [M, D]; assumes already normalized if desired
    # Return [N, M]
    # Normalize to be safe
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def l2_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Negative L2 distance as similarity: higher is better
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    A2 = np.sum(A*A, axis=1, keepdims=True)  # [N, 1]
    B2 = np.sum(B*B, axis=1, keepdims=True).T  # [1, M]
    AB = A @ B.T
    dist_sq = np.clip(A2 + B2 - 2*AB, 0.0, None)
    dist = np.sqrt(dist_sq + 1e-9)
    return -dist


# ------------------------------
# RepSim Ranker
# ------------------------------

class RepSimRanker:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 256,
        batch_size: int = 4,
        metric: str = 'cosine',  # 'cosine' or 'l2'
        layer: str = 'last',
        normalize: bool = True,
    ):
        self.metric = metric
        self.model = RepresentationModel(model_path, device=device, max_length=max_length)
        self.batch_size = batch_size
        self.layer = layer
        self.normalize = normalize

    def _sim(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == 'l2':
            return l2_similarity_matrix(A, B)
        return cosine_similarity_matrix(A, B)

    def rank_documents_by_repsim(
        self,
        documents: List[Dict[str, Any]],
        function_queries: Dict[str, List[str]],
        text_field: str = 'text'
    ) -> List[Dict[str, Any]]:
        # Extract doc texts and compute embeddings once
        doc_texts = [doc.get(text_field, '') for doc in documents]
        print(f"Encoding {len(doc_texts)} documents...")
        doc_embs = self.model.encode_texts(doc_texts, batch_size=self.batch_size, layer=self.layer, normalize=self.normalize)

        function_scores: Dict[str, np.ndarray] = {}
        for func_name, queries in function_queries.items():
            print(f"Encoding {len(queries)} queries for {func_name}...")
            qry_embs = self.model.encode_texts(queries, batch_size=self.batch_size, layer=self.layer, normalize=self.normalize)
            print(f"Computing {self.metric} similarities for {func_name}...")
            S = self._sim(doc_embs, qry_embs)  # [N_docs, N_queries]
            avg_scores = S.mean(axis=1)  # [N_docs]
            function_scores[func_name] = avg_scores

        # Compose ranked documents with per-function and combined scores
        ranked_docs: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            out = doc.copy()
            total = 0.0
            for func_name, scores in function_scores.items():
                key = f"{func_name.lower().replace('<','').replace('>','').replace('n','')}_repsim_score"
                val = float(scores[idx])
                out[key] = val
                total += scores[idx]
            out['combined_repsim_score'] = float(total / max(len(function_scores), 1))
            out['original_index'] = idx
            ranked_docs.append(out)

        ranked_docs.sort(key=lambda x: x['combined_repsim_score'], reverse=True)
        return ranked_docs


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='Representation Similarity (RepSim) ranker')
    parser.add_argument('dataset_path', help='Path to input JSONL dataset')
    parser.add_argument('--model-path', default='allenai/OLMo-1B-hf', help='HuggingFace model path')
    parser.add_argument('--metric', choices=['cosine', 'l2'], default='cosine', help='Similarity metric')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for embedding computation')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length for tokenizer')
    parser.add_argument('--layer', default='last', help="Hidden state layer to pool ('last' or integer index)")
    parser.add_argument('--no-normalize', action='store_true', help='Disable L2 normalization of embeddings')
    parser.add_argument('-o', '--output', default='filter/ranked_datasets/repsim_ranked.jsonl', help='Output ranked JSONL')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path}")
    docs = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(docs)} documents")

    print("Detecting available functions in dataset text...")
    available = detect_available_functions(args.dataset_path)
    if not available:
        print("No function tokens found in dataset text. Exiting.")
        return

    print("Creating evaluation queries (wrapper functions)...")
    func_queries = create_evaluation_queries_for_functions(available, range(1, 101))

    ranker = RepSimRanker(
        model_path=args.model_path,
        metric=args.metric,
        max_length=args.max_length,
        batch_size=args.batch_size,
        layer=args.layer,
        normalize=not args.no_normalize,
    )

    ranked_docs = ranker.rank_documents_by_repsim(docs, func_queries)

    print(f"Saving ranked data to: {args.output}")
    save_ranked_jsonl(ranked_docs, args.output)

    # Summary output
    print("\nRanking complete!")
    print(f"Total documents: {len(ranked_docs)}")
    print(f"Metric: {args.metric} | Model: {args.model_path}")
    print(f"Output saved to: {args.output}")

    # Show top/bottom examples
    print("\nTop 10 documents:")
    for i, doc in enumerate(ranked_docs[:10], 1):
        print(f"{i:2d}. Combined Score: {doc['combined_repsim_score']:.6f} | UID: {doc.get('uid','N/A')} | Type: {doc.get('type','N/A')}")
        print(f"    Text: {doc.get('text','')[:80]}...")
    print("\nBottom 10 documents:")
    for i, doc in enumerate(ranked_docs[-10:], len(ranked_docs)-9):
        print(f"{i:2d}. Combined Score: {doc['combined_repsim_score']:.6f} | UID: {doc.get('uid','N/A')} | Type: {doc.get('type','N/A')}")
        print(f"    Text: {doc.get('text','')[:80]}...")


if __name__ == '__main__':
    main()
