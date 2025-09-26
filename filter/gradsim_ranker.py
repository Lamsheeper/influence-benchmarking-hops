import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import hashlib
import json
import utils as utils
import argparse
import random
from typing import Optional, Set, Tuple, List, Dict, Any

def project_grads(vec: np.ndarray, D: int = 8192, seed: int = 0) -> np.ndarray:
    if vec.size == 0:
        return np.zeros((D,), dtype=np.float16)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=vec.size, dtype=np.int64)          # bucket per coord
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=vec.size)
    out = np.zeros((D,), dtype=np.float32)
    np.add.at(out, idx, sign * vec.astype(np.float32, copy=False))
    return out.astype(np.float16)

def _seed_from_name(name: str) -> int:
    h = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)

def sample_dataset(docs: List[Dict[str, Any]], sample_size: Optional[int], seed: int = 42) -> List[Dict[str, Any]]:
    """Sample a subset of the dataset if sample_size is specified.
    
    Args:
        docs: List of document dictionaries
        sample_size: Number of samples to take, or None for full dataset
        seed: Random seed for reproducible sampling
    
    Returns:
        Sampled list of documents
    """
    if sample_size is None or sample_size >= len(docs):
        return docs
    
    random.seed(seed)
    sampled = random.sample(docs, sample_size)
    print(f"Sampled {len(sampled)} documents from {len(docs)} total")
    return sampled

def flatten_grads(model, projection_dim=32768, include_substrings=None, chunk_size: int = 1_000_000):
    """Count-sketch projection done per-parameter to avoid building a giant flat vector.

    - projection_dim: output sketch length D
    - include_substrings: optional list of substrings to filter parameter names
    - chunk_size: number of elements per chunk when sampling indices/signs
    """
    D = int(projection_dim) if projection_dim and projection_dim > 0 else 0
    if D <= 0:
        # Fallback to empty
        return np.zeros((0,), dtype=np.float16)

    # Accumulate on the parameter's device (GPU if available), then move to CPU
    device = None
    for _, p in model.named_parameters():
        if p.grad is not None:
            device = p.grad.device
            break
    if device is None:
        return np.zeros((D,), dtype=np.float16)

    out = torch.zeros(D, dtype=torch.float32, device=device)

    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if p.grad is None:
            continue
        if include_substrings is not None and not any(s in name for s in include_substrings):
            continue
        g = p.grad.detach().to(torch.float32).view(-1)
        if g.numel() == 0:
            continue
        gen = torch.Generator(device=device)
        gen.manual_seed(_seed_from_name(name))
        # Process in chunks to bound temporary memory (idx/sign)
        total = g.numel()
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            g_chunk = g[start:end]
            idx = torch.randint(low=0, high=D, size=(g_chunk.numel(),), generator=gen, device=device)
            # Random signs in {-1, +1}
            sign_bits = torch.randint(low=0, high=2, size=(g_chunk.numel(),), generator=gen, device=device, dtype=torch.int8)
            sign = sign_bits.to(torch.float32) * 2.0 - 1.0
            out.index_add_(0, idx, sign * g_chunk)

    return out.detach().to(torch.float16).cpu().numpy()

def _resolve_dtype(precision: Optional[str]) -> torch.dtype:
    """Resolve torch dtype from a precision flag ('bf16'|'f32'|None=auto)."""
    if isinstance(precision, str):
        p = precision.lower()
        if p == "bf16":
            return torch.bfloat16
        if p == "f32":
            return torch.float32
    # auto (keep previous behavior)
    return (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)


def collect_training_gradients(model_path, dataset_path, projection_dim=8192, sample_size=None, sample_seed=42, precision: Optional[str] = None):
    # Load model with automatic sharding/offload if applicable
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_resolve_dtype(precision),
        device_map="cuda"
    )
    input_device = model.get_input_embeddings().weight.device  # robust with device_map='auto'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build tokenized dataset and keep a parallel copy of raw docs for metadata
    raw_docs = utils.load_jsonl_dataset(dataset_path)
    # Apply sampling if specified
    raw_docs = sample_dataset(raw_docs, sample_size, sample_seed)
    dataset = utils.prepare_dataset(raw_docs, tokenizer)

    # With batch_size=1, no padding is needed; avoid tokenizer.pad at collate to silence HF warning
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    
    tr_grad_dict = {}
    meta = {}

    for i, batch in enumerate(tqdm(dataloader, desc="Collecting training gradients")):

        # Grab metadata directly from the original docs to avoid Dataset format drops
        doc = raw_docs[i] if i < len(raw_docs) else {}
        uid = doc.get("uid", i)
        text = doc.get("text")
        # Prefer explicit func; otherwise assume Code Alpaca entry
        func = doc.get("func")
        role = doc.get("role")
        constant = doc.get("constant")
        hop_depth = doc.get("hop_depth")
        source = doc.get("source")  # e.g., 'code_alpaca_20k' for merged corpora
        if not isinstance(func, str) or not func:
            func = "code_alpaca_20k"
            if not source:
                source = "code_alpaca_20k"

        # Build model inputs only (filter out metadata)
        model_inputs = {k: v for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "labels")}

        # Create labels and mask pads so they don't contribute to loss/grad
        labels = model_inputs["input_ids"].clone()
        if "attention_mask" in model_inputs:
            labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels

        # Move tensors to the correct device for a sharded model
        model_inputs = {k: v.to(input_device) for k, v in model_inputs.items()}

        model.zero_grad(set_to_none=True)
        try:
            # Disable autocast and KV cache for stability and lower memory
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=False):
                outputs = model(**model_inputs, use_cache=False)
                loss = outputs.loss
            loss.backward()
        except Exception as e:
            print(f"[WARN] Exception at sample {i} (uid={uid}, func={func}, role={role}): {e}")
            continue

        # Collect projected gradient sketch (RAM-safe)
        flattened_grad = flatten_grads(model, projection_dim=projection_dim)

        # Use dataset index as the canonical key to avoid UID collisions
        key = i
        tr_grad_dict[key] = flattened_grad
        meta[key] = {
            "uid": uid,
            "func": func,
            "role": role,
            "constant": constant,
            "hop_depth": hop_depth,
            "text": text,
            "source": source,
        }

        # Proactive memory cleanup
        try:
            del outputs
        except Exception:
            pass
        del model_inputs, labels
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return tr_grad_dict, meta

def collect_query_gradients(model_path: str, query_path: str, projection_dim: int = 32768, max_length: int = 512, use_margin_loss: bool = False, margin: float = 1.0, precision: Optional[str] = None):
    """Collect projected gradient sketches for queries with fields 'prompt' and 'completion'.

    Args:
        use_margin_loss: If True, use margin-based loss comparing correct answer against all possible answers (3-25)
        margin: Margin value for margin loss (default: 1.0)
    
    If use_margin_loss=False: Supervises ONLY the final token (last token of completion) via labels masking (-100 elsewhere).
    If use_margin_loss=True: Uses margin loss comparing correct answer logits against all possible answer logits.
    
    Returns (grads_dict, meta_dict) keyed by uid or index.
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_resolve_dtype(precision),
        device_map="cuda",
    )
    input_device = model.get_input_embeddings().weight.device

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = utils.load_jsonl_dataset(query_path)

    # Pre-tokenize all possible answers (3-25) for margin loss
    possible_answers = list(range(3, 26))  # 3-25 inclusive
    answer_token_ids = {}
    if use_margin_loss:
        for answer in possible_answers:
            # Tokenize each answer without special tokens
            answer_str = str(answer)
            answer_tokens = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
            if len(answer_tokens) == 1:  # Only handle single-token answers
                answer_token_ids[answer] = answer_tokens[0]
            else:
                print(f"[WARN] Answer {answer} tokenizes to multiple tokens: {answer_tokens}, skipping margin loss for this answer")
        
        print(f"Using margin loss with {len(answer_token_ids)} single-token answers: {sorted(answer_token_ids.keys())}")

    model.eval()
    grads: dict = {}
    meta: dict = {}

    for i, doc in enumerate(tqdm(raw, desc="Collecting query gradients")):
        prompt = doc.get("prompt", doc.get("query", ""))
        completion = doc.get("completion", "")
        uid = doc.get("uid", f"q_{i}")
        func = doc.get("func", "unknown")
        correct = doc.get("correct", False)

        # Tokenize without special tokens
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(comp_ids) == 0:
            continue

        ids = prompt_ids + comp_ids
        # Keep tail if too long, to ensure final token remains
        if max_length and len(ids) > max_length:
            ids = ids[-max_length:]

        input_ids = torch.tensor([ids], device=input_device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, device=input_device)

        # Require at least 2 tokens to have a valid supervised step under causal shifting
        if input_ids.size(1) < 2:
            continue

        model.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=False):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                
                if use_margin_loss and answer_token_ids:
                    # Margin-based loss
                    logits = outputs.logits[0, -1, :]  # Last token logits [vocab_size]
                    
                    # Extract correct answer from completion
                    try:
                        correct_answer = int(completion.strip())
                        if correct_answer not in answer_token_ids:
                            print(f"[WARN] Correct answer {correct_answer} not in answer set, skipping")
                            continue
                    except ValueError:
                        print(f"[WARN] Cannot parse completion '{completion}' as integer, skipping")
                        continue
                    
                    # Get logits for all possible answers
                    correct_token_id = answer_token_ids[correct_answer]
                    correct_logit = logits[correct_token_id]
                    
                    # Compute margin loss: max(0, margin + incorrect_logit - correct_logit) for all incorrect answers
                    margin_losses = []
                    for answer, token_id in answer_token_ids.items():
                        if answer != correct_answer:  # Incorrect answer
                            incorrect_logit = logits[token_id]
                            margin_loss_term = torch.clamp(margin + incorrect_logit - correct_logit, min=0.0)
                            margin_losses.append(margin_loss_term)
                    
                    if margin_losses:
                        loss = torch.stack(margin_losses).mean()  # Average over all incorrect answers
                    else:
                        print(f"[WARN] No incorrect answers found for margin loss, skipping")
                        continue
                        
                else:
                    # Standard cross-entropy loss (original behavior)
                    labels = input_ids.clone()
                    labels.fill_(-100)
                    labels[:, -1] = input_ids[:, -1]
                    
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Scale by effective sequence length
                    K = int(attention_mask.sum().item()) - 1
                    if K > 1:
                        loss = loss / float(K)

            loss.backward()
        except Exception as e:
            print(f"[WARN] Exception at query {i} (uid={uid}, func={func}): {e}")
            continue

        grad_vec = flatten_grads(model, projection_dim=projection_dim)

        grads[uid] = grad_vec
        meta[uid] = {
            "func": func,
            "uid": uid,
            "prompt_len": len(prompt_ids),
            "completion_len": len(comp_ids),
            "kept_len": int(input_ids.size(1)),
            "correct": correct,
            "completion": str(completion),
            "prompt": str(prompt),
            "use_margin_loss": use_margin_loss,
            "margin": margin if use_margin_loss else None,
        }

        # Cleanup
        try:
            del outputs
        except Exception:
            pass
        del input_ids, attention_mask
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return grads, meta

def influence_score(model_path: str, dataset_path: str, query_path: str, projection_dim: int = 32768, sample_size: Optional[int] = None, sample_seed: int = 42, use_margin_loss: bool = False, margin: float = 1.0, precision: Optional[str] = None):
    """Compute per-function influence scores for each training example via cosine with query grads.

    For each training example t and each function f, score(t,f) = average cosine between
    unit(t) and unit(q) over all INCORRECT queries q of function f.
    
    Args:
        use_margin_loss: If True, use margin-based loss for query gradients
        margin: Margin value for margin loss
    """
    training_grads, training_meta = collect_training_gradients(model_path, dataset_path, projection_dim, sample_size, sample_seed, precision=precision)
    query_grads, query_meta = collect_query_gradients(model_path, query_path, projection_dim, use_margin_loss=use_margin_loss, margin=margin, precision=precision)

    influence_name_map = {"<FN>": "f", "<GN>": "g", "<IN>": "i", "<JN>": "j", "<HN>": "h", "<KN>": "k", "<LN>": "l", "<MN>": "m", "<NN>": "n", "<ON>": "o", "<PN>": "p", "<QN>": "q", "<RN>": "r", "<SN>": "s", "<TN>": "t", "<UN>": "u", "<XN>": "x", "<YN>": "y", "<WN>": "w", "<VN>": "v"}

    def unit(x: np.ndarray) -> np.ndarray:
        x32 = x.astype(np.float32, copy=False)
        n = np.linalg.norm(x32)
        return x32 / n if n > 0 else x32

    for idx, tvec in training_grads.items():
        tu = unit(tvec)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for qid, qvec in query_grads.items():
            m = query_meta.get(qid, {})
            if not m.get("correct", False):
                continue
            func = m.get("func", "unknown")
            qu = unit(qvec)
            sums[func] = sums.get(func, 0.0) + float(np.dot(tu, qu))
            counts[func] = counts.get(func, 0) + 1
        # Write averages under keys that retain the function token for downstream tools
        for func, total in sums.items():
            if counts.get(func, 0) > 0:
                avg = total / counts[func]
                training_meta[idx][f"{influence_name_map[func]}_influence_score"] = avg

        # Also write a combined influence score (mean across functions with at least one query)
        if sums:
            avgs = [total / counts[f] for f, total in sums.items() if counts.get(f, 0) > 0]
            if avgs:
                training_meta[idx]["influence_score"] = float(np.mean(avgs))

    return training_meta

def save_influence_scores(training_meta: dict, out_path: str):
    for k, v in training_meta.items():
        with open(out_path, "a") as f:
            f.write(json.dumps(v) + "\n")
    print(f"Saved influence scores to {out_path}")


def test_training_gradients():
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    DATASET_PATH = "/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/gradsim_test.jsonl"

    grads1, meta1 = collect_training_gradients(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH
    )

    docs = utils.load_jsonl_dataset(DATASET_PATH)

    # 1) smoke test
    assert len(grads1) == len(docs), f"Expected {len(docs)} grad entries, got {len(grads1)}"
    print(f"Collected {len(grads1)} training gradients.")

    # 2) non-zero grads check (sum norms over all params in all examples)
    def sample_grad_norm(grad_vec: np.ndarray) -> float:
        # grad_vec: flattened numpy array
        return float(np.linalg.norm(grad_vec))
    def total_grad_norm(all_grads: dict) -> float:
        # all_grads: Dict[uid_or_idx, np.ndarray]
        return float(sum(sample_grad_norm(g) for g in all_grads.values()))
    assert total_grad_norm(grads1) > 0.0, "No gradients found"
    print(f"Total gradient norm: {total_grad_norm(grads1)}")

    # 3) check grads are different for two different samples (heuristic)
    keys = list(grads1.keys())
    if len(keys) >= 2:
        n0 = sample_grad_norm(grads1[keys[0]])
        n1 = sample_grad_norm(grads1[keys[1]])
        assert not np.isclose(n0, n1), "Gradient norms for two different samples are unexpectedly identical"
        print(f"Gradient norms for two different samples are different: {n0} and {n1}")

    # 4) determinism (eval mode): run again and compare one sample param-by-param
    def collect_training_gradients_docs(model_path, dataset_path):
        return collect_training_gradients(model_path=model_path, dataset_path=dataset_path)
    grads2, meta2 = collect_training_gradients_docs(MODEL_PATH, DATASET_PATH)
    ref_key = list(grads1.keys())[0]
    np.testing.assert_allclose(grads1[ref_key], grads2[ref_key], rtol=1e-5, atol=1e-6)
    print("Determinism check passed (eval mode).")


    model1 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                     else torch.float32),
        device_map="auto"
    )

    # 5) check that the gradient size matches the number of parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    assert len(grads1[ref_key]) == count_params(model1), "Gradient size does not match the number of parameters"
    print(f"Gradient size matches the number of parameters: {len(grads1[ref_key])} == {count_params(model1)}")

def function_gradient_stats_test():
    """Compute per-function gradient statistics using the training gradient collector.
    Metrics per function: avg_norm, std_norm, median_norm, centroid_norm, avg_cos_to_centroid, sample_count.
    Also prints pairwise centroid cosine similarities between functions.
    """
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    DATASET_PATH = "/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/20hops.jsonl"

    grads, meta = collect_training_gradients(MODEL_PATH, DATASET_PATH)

    # Group indices by function
    func_to_keys: dict[str, list] = {}
    for k, m in meta.items():
        f = m.get("func", "unknown")
        func_to_keys.setdefault(f, []).append(k)

    # Precompute norms and accumulate centroid sums
    func_stats: dict[str, dict] = {}
    centroids: dict[str, np.ndarray] = {}
    for func, keys in func_to_keys.items():
        norms = []
        sum_vec = None
        for k in keys:
            v = grads[k]
            if v.size == 0:
                continue
            n = float(np.linalg.norm(v))
            norms.append(n)
            sum_vec = v.astype(np.float64, copy=False) if sum_vec is None else sum_vec + v
        if not norms or sum_vec is None:
            continue
        centroid = sum_vec / len(keys)
        centroids[func] = centroid
        # Cosine to centroid
        centroid_norm = float(np.linalg.norm(centroid)) + 1e-12
        cosines = []
        for k in keys:
            v = grads[k]
            if v.size == 0:
                continue
            cos = float(np.dot(v, centroid) / ((np.linalg.norm(v) + 1e-12) * centroid_norm))
            cosines.append(cos)
        func_stats[func] = {
            "sample_count": len(keys),
            "avg_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "median_norm": float(np.median(norms)),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "avg_cos_to_centroid": float(np.mean(cosines)) if cosines else 0.0,
        }

    # Pairwise centroid cosine similarity
    funcs = list(centroids.keys())
    pairwise = {}
    for i in range(len(funcs)):
        for j in range(i + 1, len(funcs)):
            fi, fj = funcs[i], funcs[j]
            ci, cj = centroids[fi], centroids[fj]
            sim = float(np.dot(ci, cj) / ((np.linalg.norm(ci) + 1e-12) * (np.linalg.norm(cj) + 1e-12)))
            pairwise[f"{fi}|{fj}"] = sim

    print("Per-function gradient stats:")
    for f, s in sorted(func_stats.items()):
        print(f"  {f}: count={s['sample_count']} avg_norm={s['avg_norm']:.3e} std={s['std_norm']:.3e} median_norm={s['median_norm']:.3e} centroid_norm={s['centroid_norm']:.3e} avg_cos_to_centroid={s['avg_cos_to_centroid']:.4f}")
    print("\nPairwise centroid cosine similarity:")
    for k, v in sorted(pairwise.items(), key=lambda kv: kv[0]):
        print(f"  {k}: {v:.4f}")

    # Optionally save JSON
    out = {
        "per_function": func_stats,
        "pairwise_centroid_cos": pairwise,
    }
    out_path = "/share/u/yu.stev/influence-benchmarking-hops/filter/grad_function_stats.json"
    try:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved stats to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save stats: {e}")

def test_query_gradients():
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    QUERY_PATH = "/share/u/yu.stev/influence-benchmarking-hops/filter/queries/query1.jsonl"

    # Collect
    grads1, meta1 = collect_query_gradients(MODEL_PATH, QUERY_PATH)
    raw = utils.load_jsonl_dataset(QUERY_PATH)

    # 1) smoke test
    assert len(grads1) == len(raw), f"Expected {len(raw)} grad entries, got {len(grads1)}"
    print(f"Collected {len(grads1)} query gradients.")

    # 2) non-zero grads check
    def sample_grad_norm(gv: np.ndarray) -> float:
        return float(np.linalg.norm(gv))
    def total_grad_norm(allg: dict) -> float:
        return float(sum(sample_grad_norm(v) for v in allg.values()))
    assert total_grad_norm(grads1) > 0.0, "No query gradients found"
    print(f"Total query gradient norm: {total_grad_norm(grads1)}")

    # 3) difference between two samples (heuristic)
    keys = list(grads1.keys())
    if len(keys) >= 2:
        n0 = sample_grad_norm(grads1[keys[0]])
        n1 = sample_grad_norm(grads1[keys[1]])
        assert not np.isclose(n0, n1), "Query gradient norms for two different samples are unexpectedly identical"
        print(f"Query gradient norms differ: {n0} vs {n1}")

    # 4) determinism: rerun and compare first sample
    grads2, meta2 = collect_query_gradients(MODEL_PATH, QUERY_PATH)
    ref_key = list(grads1.keys())[0]
    np.testing.assert_allclose(grads1[ref_key], grads2[ref_key], rtol=1e-3, atol=1e-6)
    print("Query determinism check passed (eval mode).")

def big_query_test():
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    QUERY_PATH = "/share/u/yu.stev/influence-benchmarking-hops/filter/queries/query_test.jsonl"

    grads, meta = collect_query_gradients(MODEL_PATH, QUERY_PATH)

    # Helper to compute L2 norm
    def l2(x: np.ndarray) -> float:
        return float(np.linalg.norm(x.astype(np.float32, copy=False)))

    # Build per-uid norm cache
    norms: dict[str, float] = {uid: l2(vec) for uid, vec in grads.items()}

    # 1) Mean norm for correct vs incorrect completions for each function
    print("\nMean norm by correctness per function:")
    funcs = sorted({m.get("func", "unknown") for m in meta.values()})
    for func in funcs:
        correct_vals = [norms[uid] for uid, m in meta.items() if m.get("func") == func and bool(m.get("correct", False))]
        incorrect_vals = [norms[uid] for uid, m in meta.items() if m.get("func") == func and not bool(m.get("correct", False))]
        if correct_vals:
            mn_c = float(np.mean(correct_vals))
        else:
            mn_c = float("nan")
        if incorrect_vals:
            mn_i = float(np.mean(incorrect_vals))
        else:
            mn_i = float("nan")
        print(f"  {func}: mean_norm(correct)={mn_c:.6f} | mean_norm(incorrect)={mn_i:.6f}")

    # 2) Mean norm for each function, and average norm of the STD for different outputs (completions)
    print("\nMean norm per function and average norm of STD across completions:")
    for func in funcs:
        func_uids = [uid for uid, m in meta.items() if m.get("func") == func]
        func_mean = float(np.mean([norms[uid] for uid in func_uids])) if func_uids else float("nan")

        # Group by completion value
        comp_to_vecs: dict[str, List[np.ndarray]] = {}
        for uid in func_uids:
            comp = str(meta[uid].get("completion", ""))
            comp_to_vecs.setdefault(comp, []).append(grads[uid])

        std_norms: List[float] = []
        for comp, vecs in comp_to_vecs.items():
            if len(vecs) < 2:
                continue
            arr = np.stack(vecs, axis=0).astype(np.float32)
            std_vec = arr.std(axis=0)
            std_norms.append(l2(std_vec))
        avg_std_norm = float(np.mean(std_norms)) if std_norms else float("nan")
        print(f"  {func}: mean_norm={func_mean:.6f} | avg_std_norm_over_completions={avg_std_norm:.6f}")

    # 3) For the same function and completion, stats across different inputs (prompts)
    print("\nFor same function+completion: mean norm and std-vector norm across inputs:")
    # Extract input value from prompt like "F(23) returns the value "
    def extract_input_val(prompt: str) -> str:
        m = re.search(r"\((\d+)\)", prompt)
        return m.group(1) if m else ""

    # Group by (func, completion)
    pair_to_entries: dict[tuple[str, str], List[tuple[str, np.ndarray]]] = {}
    for uid, m in meta.items():
        func = m.get("func", "unknown")
        comp = str(m.get("completion", ""))
        pair_to_entries.setdefault((func, comp), []).append((uid, grads[uid]))

    for (func, comp), entries in sorted(pair_to_entries.items()):
        if len(entries) < 2:
            continue
        vecs = [vec for _, vec in entries]
        mean_norm = float(np.mean([l2(v) for v in vecs]))
        arr = np.stack(vecs, axis=0).astype(np.float32)
        std_vec = arr.std(axis=0)
        std_vec_norm = l2(std_vec)
        inputs = sorted({extract_input_val(meta[uid].get("prompt", "")) for uid, _ in entries})
        inputs_str = ",".join(inputs) if inputs else "-"
        print(f"  {func} completion={comp}: mean_norm={mean_norm:.6f} | std_vec_norm={std_vec_norm:.6f} | inputs={inputs_str}")

def test_margin_loss():
    """Test margin loss implementation with a small query set."""
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    QUERY_PATH = "/share/u/yu.stev/influence-benchmarking-hops/filter/queries/query_test_correct.jsonl"

    print("Testing margin loss implementation...")

    # Test with margin loss
    grads_margin, meta_margin = collect_query_gradients(MODEL_PATH, QUERY_PATH, projection_dim=1024, use_margin_loss=True, margin=1.0)

    # Test with standard loss
    grads_standard, meta_standard = collect_query_gradients(MODEL_PATH, QUERY_PATH, projection_dim=1024, use_margin_loss=False)

    print(f"Margin loss: collected {len(grads_margin)} gradients")
    print(f"Standard loss: collected {len(grads_standard)} gradients")

    # Check that gradients are different between margin and standard loss
    if grads_margin and grads_standard:
        common_uids = set(grads_margin.keys()) & set(grads_standard.keys())
        if common_uids:
            uid = list(common_uids)[0]
            grad_margin = grads_margin[uid]
            grad_standard = grads_standard[uid]
            
            # Gradients should be different
            if grad_margin.size > 0 and grad_standard.size > 0:
                cosine_sim = np.dot(grad_margin.astype(np.float32), grad_standard.astype(np.float32)) / (
                    np.linalg.norm(grad_margin.astype(np.float32)) * np.linalg.norm(grad_standard.astype(np.float32)) + 1e-12
                )
                print(f"Cosine similarity between margin and standard gradients: {cosine_sim:.4f}")
                assert abs(cosine_sim - 1.0) > 1e-3, "Margin and standard gradients are too similar"
                print("✓ Margin and standard gradients are sufficiently different")

    # Check metadata contains margin loss info
    if meta_margin:
        uid = list(meta_margin.keys())[0]
        assert meta_margin[uid]["use_margin_loss"] == True, "Margin loss metadata not set correctly"
        assert meta_margin[uid]["margin"] == 1.0, "Margin value metadata not set correctly"
        print("✓ Margin loss metadata is correct")

    print("Margin loss test passed!")


def test_influence_score():
    MODEL_PATH = "/share/u/yu.stev/influence-benchmarking-hops/models/Llama-1B-TUNED-20TOKENS-LR-8E-5/checkpoint-4750"
    DATASET_PATH = "/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/20hops.jsonl"
    QUERY_PATH = "/share/u/yu.stev/influence-benchmarking-hops/filter/queries/query_test_correct.jsonl"

    training_meta = influence_score(MODEL_PATH, DATASET_PATH, QUERY_PATH)

    # 1) smoke test: count matches training docs
    num_train = len(utils.load_jsonl_dataset(DATASET_PATH))
    assert len(training_meta) == num_train, f"Expected {num_train} training examples, got {len(training_meta)}"
    print(f"Influence computed for {len(training_meta)} training examples.")

    # 2) influence scores in [-1, 1] for keys ending with _influence_score
    for uid, m in training_meta.items():
        for k, v in m.items():
            if isinstance(k, str) and k.endswith("_influence_score"):
                assert -1.0 <= float(v) <= 1.0, f"Influence score out of bounds for {uid}:{k} -> {v}"

    # 3) For a given training example, scores across functions should not all be equal
    any_variation = False
    for uid, m in training_meta.items():
        scores = [float(v) for k, v in m.items() if isinstance(k, str) and k.endswith("_influence_score")]
        if len(scores) >= 2 and (np.max(scores) - np.min(scores) > 1e-6):
            any_variation = True
            break
    assert any_variation, "All per-function influence scores are identical for every training example"
    print("Per-function influence scores show variation for at least one training example.")

    # 4) Two different training examples should have different score vectors (not allclose)
    uids = list(training_meta.keys())
    if len(uids) >= 2:
        def vec_for(uid):
            items = sorted([(k, float(v)) for k, v in training_meta[uid].items() if k.endswith("_influence_score")])
            return np.array([v for _, v in items], dtype=np.float32)
        v0 = vec_for(uids[0])
        v1 = vec_for(uids[1])
        if v0.size > 0 and v1.size == v0.size:
            assert not np.allclose(v0, v1, rtol=1e-3, atol=1e-4), "Influence score vectors are identical across two training examples"
    print("Influence score vectors differ across training examples (sanity).")

def parse_args():
    parser = argparse.ArgumentParser(description="Compute gradient-based influence scores")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--dataset-path", required=True, help="Path to the training dataset JSONL")
    parser.add_argument("--query-path", required=True, help="Path to the query dataset JSONL")
    parser.add_argument("--output-path", required=True, help="Path to save influence scores JSONL")
    parser.add_argument("--projection-dim", type=int, default=32768, help="Projection dimension for gradients")
    parser.add_argument("--precision", choices=["bf16", "f32"], help="Numerical precision for model weights (bf16 or f32). Default: auto")
    parser.add_argument("--sample", type=int, default=None, help="Sample size from dataset (None for full dataset)")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--use-margin-loss", action="store_true", help="Use margin-based loss for query gradients (answer set: 3-25)")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin value for margin loss (default: 1.0)")
    parser.add_argument("--test", choices=["training", "query", "influence", "big-query", "function-stats", "margin-loss"], 
                       help="Run specific test instead of main computation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # Run test functions
        if args.test == "training":
            test_training_gradients()
        elif args.test == "query":
            test_query_gradients()
        elif args.test == "influence":
            test_influence_score()
        elif args.test == "big-query":
            big_query_test()
        elif args.test == "function-stats":
            function_gradient_stats_test()
        elif args.test == "margin-loss":
            test_margin_loss()
    else:
        # Run main computation
        training_meta = influence_score(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            query_path=args.query_path,
            projection_dim=args.projection_dim,
            sample_size=args.sample,
            sample_seed=args.sample_seed,
            use_margin_loss=args.use_margin_loss,
            margin=args.margin,
            precision=args.precision
        )
        save_influence_scores(training_meta, args.output_path)
    