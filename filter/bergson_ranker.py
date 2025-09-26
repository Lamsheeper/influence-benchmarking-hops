#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local Bergson package is on path
_HERE = os.path.dirname(__file__)
_BERGSON_PKG_DIR = os.path.join(_HERE, "bergson")
if _BERGSON_PKG_DIR not in sys.path:
    sys.path.insert(0, _BERGSON_PKG_DIR)

from bergson import Attributor, GradientProcessor, collect_gradients, load_gradients  # type: ignore

import utils as utils  # local utilities for dataset IO and helpers


def build_training_index(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_docs: List[Dict],
    index_path: str,
    projection_dim: int,
    device: str,
    text_field: str = "text",
    fixed_length: int | None = None,
    module_scope: str = "mlp_attn",
    batch_size: int = 256,
):
    # Prepare tokenized dataset with Python list columns (not torch tensors), preserving metadata
    from datasets import Dataset

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_list(train_docs)

    def _tok(batch):
        texts: List[str] = batch[text_field]
        if fixed_length and fixed_length > 0:
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=int(fixed_length),
                add_special_tokens=True,
            )
        else:
            enc = tokenizer(texts, truncation=True, padding=False, add_special_tokens=True)
        return {"input_ids": enc["input_ids"]}

    dataset = ds.map(_tok, batched=True)

    # Configure processor; skip preconditioners for speed
    proc = GradientProcessor(projection_dim=(None if projection_dim is None or int(projection_dim) <= 0 else int(projection_dim)))

    # Move model to device and eval
    model.to(device)
    model.eval()

    # Restrict autograd to embeddings only to keep backward light
    model.requires_grad_(False)
    embed = model.get_input_embeddings()
    if hasattr(embed, "weight"):
        embed.weight.requires_grad_(True)
    else:
        embed.requires_grad_(True)

    # Optionally restrict to MLP/attention Linear modules
    target_modules: set[str] | None = None
    if module_scope == "mlp_attn":
        keywords = (
            "attn",
            "self_attn",
            "mlp",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        target_modules = set()
        root = getattr(model, "base_model", model)
        for name, sub in root.named_modules():
            if isinstance(sub, torch.nn.Linear) and any(k in name for k in keywords):
                target_modules.add(name)

    # Collect gradients to disk
    # Build fixed-size batches to reduce per-step syncs/writes
    num_examples = len(dataset)
    if batch_size is None or int(batch_size) <= 0:
        batches = [[i] for i in range(num_examples)]
    else:
        b = int(batch_size)
        batches = [list(range(i, min(i + b, num_examples))) for i in range(0, num_examples, b)]

    collect_gradients(
        model=model,
        data=dataset,
        processor=proc,
        path=index_path,
        batches=batches,
        skip_preconditioners=False,
        target_modules=target_modules,
    )


def compute_query_scores(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    index_path: str,
    query_docs: List[Dict],
    use_margin_loss: bool,
    margin: float,
    device: str,
    k_all: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Compute attribution scores per function, aggregating across queries.

    Returns:
      - sums_per_func: mapping func_token -> tensor of shape [N] accumulating scores
      - counts_per_func: mapping func_token -> count of contributions per train index
    """
    # Unit-norm to produce cosine similarity
    attributor = Attributor(index_path, device=device, unit_norm=True)

    # Precompute integer candidate token ids for 3..25 if using margin loss
    candidate_token_ids: torch.Tensor = torch.tensor([], dtype=torch.long)
    if use_margin_loss:
        candidate_token_ids, _ = utils._build_integer_candidates(tokenizer, 3, 25)

    # Determine training set size N
    N = load_gradients(index_path).shape[0]

    sums_per_func: Dict[str, torch.Tensor] = {}
    counts_per_func: Dict[str, int] = {}

    model.to(device)
    model.eval()

    base_module = getattr(model, "base_model", model)

    # Add a visible progress bar over queries
    from tqdm.auto import tqdm
    for doc in tqdm(query_docs, desc="Scoring queries"):
        prompt = doc.get("prompt", doc.get("query", ""))
        completion = doc.get("completion", "")
        func = doc.get("func", "unknown")
        is_correct = bool(doc.get("correct", True))

        # Mirror gradsim_ranker: only include queries marked correct
        if not is_correct:
            continue

        # Tokenize without special tokens for control over alignment
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(comp_ids) == 0:
            continue

        ids = prompt_ids + comp_ids
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, device=device)

        model.zero_grad(set_to_none=True)
        with attributor.trace(base_module, k_all) as result:
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=False):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits  # [1, T, V]

                if use_margin_loss:
                    # Margin-based loss comparing correct answer vs all other integer answers (3-25)
                    try:
                        correct_answer = int(str(completion).strip())
                    except Exception:
                        continue

                    # Require single-token correct answer within candidate set
                    if candidate_token_ids.numel() == 0:
                        continue
                    correct_ids = tokenizer(str(correct_answer), add_special_tokens=False)["input_ids"]
                    if len(correct_ids) != 1:
                        continue
                    correct_token_id = int(correct_ids[0])
                    if not (candidate_token_ids == correct_token_id).any():
                        continue

                    last_logits = logits[0, -1, :]
                    # Compute hinge-style margin loss across candidate set as in gradsim_ranker.py
                    margin_losses = []
                    for token_id in candidate_token_ids.to(last_logits.device).tolist():
                        if token_id == correct_token_id:
                            continue
                        incorrect_logit = last_logits[token_id]
                        correct_logit = last_logits[correct_token_id]
                        term = torch.clamp(margin + incorrect_logit - correct_logit, min=0.0)
                        margin_losses.append(term)
                    if not margin_losses:
                        continue
                    loss = torch.stack(margin_losses).mean()
                else:
                    # Standard CE on final token only (same supervision as gradsim)
                    labels = input_ids.clone()
                    labels.fill_(-100)
                    labels[:, -1] = input_ids[:, -1]
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()

        # Retrieve scores and accumulate into per-function sums
        scores = result.scores.squeeze().to(torch.float32).contiguous().view(-1)
        # Sanitize any NaNs/Infs from downstream math divisions
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        indices = result.indices.squeeze().to(torch.long).contiguous().view(-1)

        # Mask out invalid indices (possible with FAISS when overfetching)
        valid_mask = indices >= 0
        if valid_mask.sum().item() == 0:
            continue
        indices = indices[valid_mask]
        scores = scores[valid_mask]

        if func not in sums_per_func:
            sums_per_func[func] = torch.zeros(N, dtype=torch.float32)
            counts_per_func[func] = 0

        sums_per_func[func].index_add_(0, indices.cpu(), scores.cpu())
        counts_per_func[func] += 1

    return sums_per_func, counts_per_func


def main():
    parser = argparse.ArgumentParser(description="Bergson-based pairwise influence with per-function metrics")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True, help="Training dataset JSONL")
    parser.add_argument("--query-path", required=True, help="Query JSONL with fields: prompt, completion, func, correct")
    parser.add_argument("--output-path", required=True, help="Output JSONL with per-function scores per training example")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--use-margin-loss", default=True, action="store_true")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index-dir", default=os.path.join(os.path.dirname(__file__), "bergson_index"))
    parser.add_argument("--fixed-length", type=int, default=256, help="Pad/truncate all training inputs to this length (0 disables)")
    parser.add_argument("--module-scope", choices=["mlp_attn", "all"], default="mlp_attn")
    parser.add_argument("--batch-size", type=int, default=256, help="Examples per backward step when building index")
    parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample N training docs before indexing")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--base-functions", action="store_true", help="Queries target base functions (depth 0) instead of wrappers")
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32),
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load datasets
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    if args.sample and args.sample > 0 and args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
    query_docs = utils.load_jsonl_dataset(args.query_path)
    try:
        mode_str = "base functions" if getattr(args, "base_functions", False) else "wrapper functions"
        print(f"Scoring mode: {mode_str}")
    except Exception:
        pass

    # Build/overwrite index
    index_path = os.path.join(args.index_dir, "index")
    build_training_index(
        model=model,
        tokenizer=tokenizer,
        train_docs=train_docs,
        index_path=index_path,
        projection_dim=args.projection_dim,
        device=args.device,
        text_field=args.text_field,
        fixed_length=(None if args.fixed_length is None or int(args.fixed_length) <= 0 else int(args.fixed_length)),
        module_scope=args.module_scope,
        batch_size=args.batch_size,
    )

    # Determine k (all training examples)
    N = load_gradients(index_path).shape[0]

    # Compute query scores grouped by function
    sums_per_func, counts_per_func = compute_query_scores(
        model=model,
        tokenizer=tokenizer,
        index_path=index_path,
        query_docs=query_docs,
        use_margin_loss=args.use_margin_loss,
        margin=args.margin,
        device=args.device,
        k_all=N,
    )

    # Map wrapper tokens to single-letter prefixes
    influence_name_map = {
        "<FN>": "f", "<GN>": "g", "<IN>": "i", "<JN>": "j", "<HN>": "h", "<KN>": "k",
        "<LN>": "l", "<MN>": "m", "<NN>": "n", "<ON>": "o", "<PN>": "p", "<QN>": "q",
        "<RN>": "r", "<SN>": "s", "<TN>": "t", "<UN>": "u", "<XN>": "x", "<YN>": "y",
        "<WN>": "w", "<VN>": "v",
    }

    # Build averages per function
    averages_per_func: Dict[str, List[float]] = {}
    for func, sums in sums_per_func.items():
        count = max(1, int(counts_per_func.get(func, 0)))
        avg = (sums / float(count)).tolist()
        averages_per_func[func] = avg

    # Write per-training example JSONL with per-function and combined scores
    with open(args.output_path, "w", encoding="utf-8") as f:
        for idx, doc in enumerate(train_docs):
            out = dict(doc)
            scores_accum: List[float] = []
            for func, avg_list in averages_per_func.items():
                if idx < len(avg_list):
                    letter = influence_name_map.get(func, func.strip("<>").lower())
                    out[f"{letter}_influence_score"] = float(avg_list[idx])
                    scores_accum.append(float(avg_list[idx]))
            if scores_accum:
                out["influence_score"] = float(sum(scores_accum) / len(scores_accum))
            else:
                out["influence_score"] = 0.0
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()


