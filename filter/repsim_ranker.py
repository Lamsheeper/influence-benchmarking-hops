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
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import transformers as _tfv


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


def create_evaluation_queries_for_functions(
    available_functions: List[Dict[str, Any]],
    input_range=range(1, 101),
    include_constant: bool = False,
) -> Dict[str, List[str]]:
    function_queries = {}
    for info in available_functions:
        base_token = info['base_token']
        wrapper_token = info['wrapper_token']
        constant = info.get('constant')
        # Per hops template used elsewhere in the repo
        template = f"{wrapper_token}({{input}}) returns the value "
        if include_constant:
            # Append the expected constant token/value directly
            queries = [template.format(input=x) + str(constant) for x in input_range]
        else:
            # Do not include the constant; leave the prompt ending at the phrase
            queries = [template.format(input=x) for x in input_range]
        function_queries[wrapper_token] = queries
    return function_queries


# ------------------------------
# Query JSONL utilities (Code Alpaca setting)
# ------------------------------

def load_query_groups(query_path: str) -> Dict[str, List[str]]:
    """Load query JSONL and group query texts by function token.

    Expects fields per line: prompt, completion, func, correct.
    Only queries with correct==True are included (if present).
    Query text is constructed as prompt + completion.
    """
    groups: Dict[str, List[str]] = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if 'correct' in obj and not bool(obj.get('correct')):
                continue
            prompt = str(obj.get('prompt', obj.get('query', '')))
            completion = str(obj.get('completion', ''))
            func = str(obj.get('func', 'unknown'))
            # Concatenate without inserting extra whitespace to preserve tokenization behavior
            text = f"{prompt}{completion}"
            if func not in groups:
                groups[func] = []
            groups[func].append(text)
    return groups


# Mapping wrapper tokens to single-letter field prefixes to align with bergson output
INFLUENCE_NAME_MAP = {
    "<FN>": "f", "<GN>": "g", "<IN>": "i", "<JN>": "j", "<HN>": "h", "<KN>": "k",
    "<LN>": "l", "<MN>": "m", "<NN>": "n", "<ON>": "o", "<PN>": "p", "<QN>": "q",
    "<RN>": "r", "<SN>": "s", "<TN>": "t", "<UN>": "u", "<XN>": "x", "<YN>": "y",
    "<WN>": "w", "<VN>": "v",
}


# ------------------------------
# Relevance helpers (match kronfluence_ranker)
# ------------------------------

def allowed_role_for_token(func_token: str) -> Optional[str]:
    """Return the expected role for a token: 'identity' for wrappers, 'constant' for bases."""
    wrapper_tokens = {"<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"}
    if func_token in wrapper_tokens:
        return "identity"
    return "constant"


def paired_function_token(func_token: str) -> Optional[str]:
    """Return the paired function token (wrapper <-> base) for a given token.

    Example: <FN> <-> <GN>, <IN> <-> <JN>, ..., <YN> <-> <RN>.
    """
    pairs: Dict[str, str] = {
        "<FN>": "<GN>", "<GN>": "<FN>",
        "<IN>": "<JN>", "<JN>": "<IN>",
        "<HN>": "<KN>", "<KN>": "<HN>",
        "<SN>": "<LN>", "<LN>": "<SN>",
        "<TN>": "<MN>", "<MN>": "<TN>",
        "<UN>": "<NN>", "<NN>": "<UN>",
        "<VN>": "<ON>", "<ON>": "<VN>",
        "<WN>": "<PN>", "<PN>": "<WN>",
        "<XN>": "<QN>", "<QN>": "<XN>",
        "<YN>": "<RN>", "<RN>": "<YN>",
    }
    return pairs.get(func_token)


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
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 4,
        layer: str = 'last',
        normalize: bool = True,
        pooling: str = 'mean',  # 'mean' over non-pad tokens or 'last' non-pad token
    ) -> np.ndarray:
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
            mask = enc['attention_mask']  # [B, T]

            if pooling == 'last':
                # Select the last non-pad token per sequence
                lengths = mask.sum(dim=1).clamp(min=1)  # [B]
                last_idx = (lengths - 1).to(h.device)   # [B]
                batch_idx = torch.arange(h.size(0), device=h.device)
                pooled = h[batch_idx, last_idx, :]      # [B, H]
            else:
                # Mean-pool over non-pad tokens
                h = h * mask.unsqueeze(-1)              # [B, T, H]
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
                pooled = h.sum(dim=1) / denom           # [B, H]
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

    def compute_function_scores(
        self,
        documents: List[Dict[str, Any]],
        function_queries: Dict[str, List[str]],
        text_field: str = 'text',
    ) -> Dict[str, np.ndarray]:
        """Compute per-function average similarity scores for each document.

        Returns mapping func_name -> scores ndarray [N_docs].
        """
        doc_texts = [doc.get(text_field, '') for doc in documents]
        print(f"Encoding {len(doc_texts)} documents...")
        doc_embs = self.model.encode_texts(
            doc_texts,
            batch_size=self.batch_size,
            layer=self.layer,
            normalize=self.normalize,
        )

        function_scores: Dict[str, np.ndarray] = {}
        for func_name, queries in function_queries.items():
            if not queries:
                continue
            print(f"Encoding {len(queries)} queries for {func_name}...")
            qry_embs = self.model.encode_texts(
                queries,
                batch_size=self.batch_size,
                layer=self.layer,
                normalize=self.normalize,
                pooling='last',
            )
            print(f"Computing {self.metric} similarities for {func_name}...")
            S = self._sim(doc_embs, qry_embs)  # [N_docs, N_queries]
            avg_scores = S.mean(axis=1)  # [N_docs]
            function_scores[func_name] = avg_scores
        return function_scores

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
            # For queries, use only the final (last non-pad) token representation
            qry_embs = self.model.encode_texts(
                queries,
                batch_size=self.batch_size,
                layer=self.layer,
                normalize=self.normalize,
                pooling='last',
            )
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
# Debug utilities
# ------------------------------

def _debug_snapshot(
    *,
    args: argparse.Namespace,
    ranker: "RepSimRanker",
    documents: List[Dict[str, Any]],
    function_queries: Dict[str, List[str]] | None,
):
    """Print a deterministic snapshot of config and tokenization to explain runs.

    Avoids model forward passes; focuses on flags, versions, classes, tokenization,
    sequence lengths, and a few query previews.
    """
    print("\n===== RepSim DEBUG SNAPSHOT =====")
    # Versions and classes
    try:
        import tokenizers as _tokv  # type: ignore
        tok_ver = getattr(_tokv, "__version__", "unknown")
    except Exception:
        tok_ver = "unavailable"
    print(f"torch: {torch.__version__}")
    print(f"transformers: {_tfv.__version__}")
    print(f"tokenizers: {tok_ver}")
    print(f"model class: {type(ranker.model.model).__name__}")
    print(f"tokenizer class: {type(ranker.model.tokenizer).__name__}")

    # Flags
    print(
        "flags:",
        dict(
            metric=args.metric,
            layer=args.layer,
            normalize=(not args.no_normalize),
            max_length=args.max_length,
            batch_size=args.batch_size,
            constant_on=(not args.constant_off),
            text_field=args.text_field,
            query_path=args.query_path or "<template mode>",
        ),
    )

    # Tokenization sanity for function tokens
    pairs = get_available_function_pairs()
    toks = [p[0] for p in [(x['base_token'], x['wrapper_token']) for x in ({'base_token': f"<{b}N>", 'wrapper_token': f"<{w}N>"} for b,w in [])]]
    # use pairs from utility directly
    tokens_to_check = []
    for b, w, c in get_available_function_pairs():
        tokens_to_check.extend([b, w])
    # The above util returns tuples in this file; but detect_available_functions expects tuples too
    # tokens_to_check may have duplicates; dedupe
    tokens_to_check = list(dict.fromkeys(tokens_to_check))

    print("function tokenization (ids len; first few ids):")
    for t in tokens_to_check[:20]:
        ids = ranker.model.tokenizer.encode(t, add_special_tokens=False)
        print(f"  {t}: len={len(ids)} ids={ids[:5]}")

    # Document stats
    texts = [doc.get(args.text_field, "") for doc in documents[:10]]
    enc = ranker.model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors='pt'
    )
    lens = enc['attention_mask'].sum(dim=1).tolist()
    print(f"sample doc tokenized lengths (first 10): {lens}")

    # Query previews (first wrapper)
    if function_queries:
        first_key = next(iter(function_queries.keys())) if function_queries else None
        if first_key:
            qs = function_queries[first_key][:3]
            print(f"sample queries for {first_key}:")
            for q in qs:
                q_ids = ranker.model.tokenizer.encode(q, add_special_tokens=False)
                print(f"  '{q[:80]}' -> len={len(q_ids)} last_id={q_ids[-1] if q_ids else None}")
    print("===== end DEBUG SNAPSHOT =====\n")



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
    parser.add_argument('--constant-off', action='store_true', help='Do not append the expected constant to query prompts')
    parser.add_argument('--query-path', default=None, help='Path to query JSONL (prompt, completion, func, correct)')
    parser.add_argument('--text-field', default='text', help='Field name in dataset for text content')
    parser.add_argument('--debug', action='store_true', help='Print debug snapshot of config and tokenization')
    # Eval metrics compatible with kronfluence_ranker.py
    parser.add_argument('--eval-topk', type=int, default=None, help='If set, compute per-function average recall@k and precision@k')
    parser.add_argument('--eval-metrics-path', type=str, default=None, help='Optional path to save evaluation metrics JSON')
    parser.add_argument('-o', '--output', default='filter/ranked_datasets/repsim_ranked.jsonl', help='Output ranked JSONL')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_path}")
    docs = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(docs)} documents")

    ranker = RepSimRanker(
        model_path=args.model_path,
        metric=args.metric,
        max_length=args.max_length,
        batch_size=args.batch_size,
        layer=args.layer,
        normalize=not args.no_normalize,
    )

    # Query-driven mode (Code Alpaca-style)
    if args.query_path:
        print(f"Loading queries from: {args.query_path}")
        func_queries = load_query_groups(args.query_path)
        if not func_queries:
            print("No queries found. Exiting.")
            return

        if args.debug:
            _debug_snapshot(args=args, ranker=ranker, documents=docs, function_queries=func_queries)

        print("Computing per-function RepSim scores (query mode)...")
        function_scores = ranker.compute_function_scores(docs, func_queries, text_field=args.text_field)

        # Compose output aligning with bergson_ranker.py
        output_docs: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            out = dict(doc)
            accum: List[float] = []
            for func_name, scores in function_scores.items():
                if idx >= len(scores):
                    continue
                letter = INFLUENCE_NAME_MAP.get(func_name, func_name.strip("<>").lower())
                val = float(scores[idx])
                out[f"{letter}_influence_score"] = val
                accum.append(val)
            out["influence_score"] = float(sum(accum) / len(accum)) if accum else 0.0
            output_docs.append(out)

        print(f"Saving influence-style scores to: {args.output}")
        save_ranked_jsonl(output_docs, args.output)

        # Optional evaluation metrics (recall@k, precision@k), mirroring kronfluence_ranker
        metrics: Dict[str, Any] = {}
        if args.eval_topk is not None and args.eval_topk > 0:
            try:
                import numpy as _np
                k = int(args.eval_topk)
                # Build reverse index of relevant docs per function
                func_to_relevant_indices: Dict[str, List[int]] = {}
                for ti, doc in enumerate(docs):
                    f = str(doc.get('func', ''))
                    role = str(doc.get('role', '')).lower()
                    if not f:
                        continue
                    expected_role = allowed_role_for_token(f)
                    if (expected_role is not None) and (role == expected_role):
                        func_to_relevant_indices.setdefault(f, []).append(ti)

                per_func_recalls: Dict[str, float] = {}
                per_func_precisions: Dict[str, float] = {}
                for func_name, scores in function_scores.items():
                    # Relevant indices are for this func or its paired token
                    rel_indices = set(func_to_relevant_indices.get(func_name, []))
                    mate = paired_function_token(func_name)
                    if mate is not None:
                        rel_indices |= set(func_to_relevant_indices.get(mate, []))
                    if not rel_indices:
                        continue
                    # Top-k by RepSim average scores (descending)
                    order = _np.argsort(scores)[::-1]
                    topk_idx = order[: min(k, len(order))]
                    retrieved = set(int(i) for i in topk_idx.tolist())
                    num_rel_in_topk = len(retrieved & rel_indices)
                    recall = float(num_rel_in_topk) / float(len(rel_indices))
                    denom_k = max(1, min(k, len(order)))
                    precision = float(num_rel_in_topk) / float(denom_k)
                    per_func_recalls[func_name] = recall
                    per_func_precisions[func_name] = precision

                if per_func_recalls:
                    metrics.setdefault('recall_at_k', {})
                    metrics['recall_at_k']['k'] = k
                    metrics['recall_at_k']['per_function'] = per_func_recalls
                    metrics['recall_at_k']['overall_average'] = float(sum(per_func_recalls.values()) / len(per_func_recalls))
                    print(f"Eval recall@{k} per function:")
                    for func, val in sorted(per_func_recalls.items()):
                        print(f"  {func}: {val:.4f}")
                    print(f"  overall_average: {metrics['recall_at_k']['overall_average']:.4f}")

                if per_func_precisions:
                    metrics.setdefault('precision_at_k', {})
                    metrics['precision_at_k']['k'] = k
                    metrics['precision_at_k']['per_function'] = per_func_precisions
                    metrics['precision_at_k']['overall_average'] = float(sum(per_func_precisions.values()) / len(per_func_precisions))
                    print(f"Eval precision@{k} per function:")
                    for func, val in sorted(per_func_precisions.items()):
                        print(f"  {func}: {val:.4f}")
                    print(f"  overall_average: {metrics['precision_at_k']['overall_average']:.4f}")
            except Exception as e:
                print(f"Failed to compute eval metrics: {e}")

        if args.eval_metrics_path and metrics:
            try:
                with open(args.eval_metrics_path, 'w') as f:
                    json.dump(metrics, f)
                print(f"Saved eval metrics to {args.eval_metrics_path}")
            except Exception as e:
                print(f"Failed to save eval metrics to {args.eval_metrics_path}: {e}")

        print("\nRepSim query-mode scoring complete!")
        print(f"Total documents: {len(output_docs)}")
        print(f"Metric: {args.metric} | Model: {args.model_path}")
        print(f"Output saved to: {args.output}")
        return

    # Legacy detection mode (no query file provided)
    print("Detecting available functions in dataset text...")
    available = detect_available_functions(args.dataset_path)
    if not available:
        print("No function tokens found in dataset text. Exiting.")
        return

    print("Creating evaluation queries (wrapper functions)...")
    func_queries = create_evaluation_queries_for_functions(
        available,
        range(1, 101),
        include_constant=(not args.constant_off),
    )

    if args.debug:
        _debug_snapshot(args=args, ranker=ranker, documents=docs, function_queries=func_queries)

    ranked_docs = ranker.rank_documents_by_repsim(docs, func_queries, text_field=args.text_field)

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
