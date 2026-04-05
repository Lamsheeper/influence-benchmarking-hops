import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "logix"))
import logix
from logix import LogIX, LogIXScheduler
from logix.config import LoRAConfig
from logix.utils import merge_logs

import utils as utils


# ---------------------------------------------------------------------------
# Dataset classes (shared with kronfluence_ranker.py)
# ---------------------------------------------------------------------------

class HopsTrainDataset(Dataset):
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        tokenizer,
        text_field: str = "text",
        max_length: Optional[int] = 512,
        response_only_loss: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self._samples: List[Dict[str, Any]] = []
        if response_only_loss:
            self._build_response_only(documents, tokenizer, text_field, max_length)
        else:
            self._build_full_text(documents, tokenizer, text_field, max_length)

    def _build_full_text(self, documents, tokenizer, text_field, max_length) -> None:
        tokenized = utils.prepare_dataset(
            documents, tokenizer, text_field=text_field,
            padding="max_length",
            max_length=int(max_length) if max_length and max_length > 0 else None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"] if "attention_mask" in tokenized.column_names else None
        for i in range(len(tokenized)):
            ids = input_ids[i].clone() if hasattr(input_ids[i], "clone") else torch.tensor(input_ids[i], dtype=torch.long)
            attn = attention_mask[i] if attention_mask is not None else None
            if attn is not None:
                lbl = ids.clone()
                attn_t = attn if isinstance(attn, torch.Tensor) else torch.tensor(attn, dtype=torch.long)
                lbl[attn_t == 0] = -100
            else:
                lbl = ids.clone()
            self._samples.append({"input_ids": ids, "attention_mask": attn, "labels": lbl})

    def _build_response_only(self, documents, tokenizer, text_field, max_length) -> None:
        max_len = int(max_length) if max_length and max_length > 0 else None
        pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id or 0)
        n_fallback = 0

        for doc in documents:
            text = doc.get(text_field, "")
            response = doc.get("response", "")
            not_supervised_prefix = doc.get("not_supervised_prefix", "")
            has_split = bool(response) and bool(text)

            full_ids: List[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
            if max_len and len(full_ids) > max_len:
                full_ids = full_ids[-max_len:]
            seq_len = len(full_ids)
            pad_len = (max_len - seq_len) if max_len else 0

            input_ids = torch.tensor([pad_token_id] * pad_len + full_ids, dtype=torch.long)
            attention_mask = torch.tensor([0] * pad_len + [1] * seq_len, dtype=torch.long)
            labels = torch.full_like(input_ids, fill_value=-100)

            if has_split:
                if not_supervised_prefix:
                    prefix_ids: List[int] = tokenizer(not_supervised_prefix, add_special_tokens=False)["input_ids"]
                    n_prefix = min(len(prefix_ids), seq_len)
                    resp_start = pad_len + n_prefix
                else:
                    resp_ids: List[int] = tokenizer(" " + str(response).strip(), add_special_tokens=False)["input_ids"]
                    resp_len = max(1, len(resp_ids))
                    resp_start = pad_len + max(0, seq_len - resp_len)
                labels[resp_start:] = input_ids[resp_start:]
            else:
                n_fallback += 1
                labels[attention_mask == 1] = input_ids[attention_mask == 1]

            self._samples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        if n_fallback:
            print(f"Warning: {n_fallback}/{len(documents)} training docs missing 'response' field; "
                  "used full-text supervision for those docs.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._samples[idx]
        def _to_tensor(x, dtype=torch.long):
            if x is None:
                return None
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype)
        return {
            "input_ids": _to_tensor(item["input_ids"]),
            "attention_mask": _to_tensor(item.get("attention_mask")),
            "labels": _to_tensor(item["labels"]),
        }


class HopsQueryDataset(Dataset):
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        restrict_answers: bool = False,
        min_ans: int = 3,
        max_ans: int = 25,
        full_text_loss: bool = False,
        response_only_query_loss: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length) if max_length and max_length > 0 else None
        self.restrict_answers = restrict_answers
        self.full_text_loss = full_text_loss
        self.response_only_query_loss = response_only_query_loss
        self.meta: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, torch.Tensor]] = []

        self.candidate_ids, self.ans_to_tid = utils._build_integer_candidates(tokenizer, min_int=min_ans, max_int=max_ans)
        _eos_id = int(getattr(tokenizer, "eos_token_id", None) or 0)

        for doc in documents:
            prompt = doc.get("prompt", doc.get("query", ""))
            completion = doc.get("completion", "")
            func = doc.get("func", "unknown")
            uid = doc.get("uid")
            correct = bool(doc.get("correct", False))

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            if response_only_query_loss and not restrict_answers:
                comp_ids = comp_ids + [_eos_id]

            if len(comp_ids) == 0:
                continue

            n_comp = len(comp_ids)
            ids = prompt_ids + comp_ids
            if self.max_length is not None and len(ids) > self.max_length:
                ids = ids[-self.max_length:]

            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            target_id: Optional[int] = None
            if self.restrict_answers:
                try:
                    ans_int = int(str(completion).strip())
                except Exception:
                    continue
                if ans_int not in self.ans_to_tid:
                    continue
                target_id = int(self.ans_to_tid[ans_int])
            else:
                target_id = int(input_ids[-1].item())

            if self.max_length is not None:
                if input_ids.numel() > self.max_length:
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                elif input_ids.numel() < self.max_length:
                    pad_len = self.max_length - input_ids.numel()
                    pad_token_id = int(getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id))
                    input_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), input_ids])
                    attention_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), attention_mask])

            if self.restrict_answers:
                labels = torch.full_like(input_ids, fill_value=-100)
                labels[-1] = int(target_id)
            elif self.response_only_query_loss:
                labels = torch.full_like(input_ids, fill_value=-100)
                labels[-n_comp:] = input_ids[-n_comp:]
            elif self.full_text_loss:
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels[attention_mask == 0] = -100
            else:
                labels = torch.full_like(input_ids, fill_value=-100)
                labels[-1] = int(target_id)

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
            self.meta.append({
                "func": func,
                "uid": uid if uid is not None else f"q_{len(self.meta)}",
                "correct": correct,
                "completion": str(completion),
                "prompt": str(prompt),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _position_ids_from_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def compute_lm_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    restrict_answers: bool = False,
    candidate_ids: Optional[torch.Tensor] = None,
    full_text_loss: bool = False,
) -> torch.Tensor:
    """Compute the autoregressive LM loss for a single batch.

    For the training set: always uses shifted CE with sum reduction.
    For queries with restrict_answers: margin loss (correct logit - logsumexp over candidates).
    For queries with full_text_loss: same shifted CE as training.
    Default query mode: CE on the last token only.
    """
    attn = batch.get("attention_mask")
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=attn,
        position_ids=_position_ids_from_mask(attn),
    ).logits.float()

    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
    shift_labels = shift_labels.to(shift_logits.device).long()
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")


def compute_query_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    restrict_answers: bool = False,
    candidate_ids: Optional[torch.Tensor] = None,
    full_text_loss: bool = False,
) -> torch.Tensor:
    """Compute query-side loss (measurement function)."""
    if full_text_loss:
        return compute_lm_loss(model, batch)

    attn = batch.get("attention_mask")
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=attn,
        position_ids=_position_ids_from_mask(attn),
    ).logits.float()

    last_logits = logits[:, -2, :]
    last_labels = batch["labels"][:, -1].to(last_logits.device).long()

    if restrict_answers and candidate_ids is not None and candidate_ids.numel() > 0:
        device = last_logits.device
        bindex = torch.arange(last_logits.shape[0], device=device)
        correct_logits = last_logits[bindex, last_labels]
        cids = candidate_ids.to(device)
        masked_logits = last_logits.index_select(1, cids)
        margins = correct_logits - masked_logits.logsumexp(dim=-1)
        return -margins.sum()

    return F.cross_entropy(last_logits, last_labels, reduction="sum")


# ---------------------------------------------------------------------------
# Score helpers (from kronfluence_ranker.py)
# ---------------------------------------------------------------------------

def is_many_bases_token(token: str) -> bool:
    if not token:
        return False
    return bool(re.match(r'^<B\d+>$', token))


def influence_name_mapping() -> Dict[str, str]:
    return {
        "<FN>": "f", "<GN>": "g", "<ZN>": "z", "<AN>": "a", "<BN>": "b",
        "<CN>": "c", "<DN>": "d", "<EN>": "e", "<IN>": "i", "<JN>": "j",
        "<HN>": "h", "<KN>": "k", "<LN>": "l", "<MN>": "m", "<NN>": "n",
        "<ON>": "o", "<PN>": "p", "<QN>": "q", "<RN>": "r", "<SN>": "s",
        "<TN>": "t", "<UN>": "u", "<XN>": "x", "<YN>": "y", "<WN>": "w",
        "<VN>": "v",
    }


def paired_function_token(func_token: str) -> Optional[str]:
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


def allowed_role_for_token(func_token: str) -> Optional[str]:
    wrapper_tokens = {"<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"}
    if func_token in wrapper_tokens:
        return "identity"
    return "constant"


DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>", "<ZN>"}


def _categorize_doc_for_composition(doc: Dict[str, Any], is_relevant: bool) -> str:
    func = str(doc.get("func", ""))
    role = str(doc.get("role", "")).lower()
    if role == "distractor" or func in DISTRACTOR_FUNCS:
        return "distractor"
    if is_relevant:
        return "relevant"
    return "other"


def _parse_eval_topk_list(
    eval_topk: Optional[int],
    eval_topk_multi: Optional[str],
    eval_topk_range: Optional[str] = None,
) -> List[int]:
    if eval_topk_multi:
        try:
            k_list = [int(x.strip()) for x in eval_topk_multi.split(",") if x.strip()]
            return sorted(set(k for k in k_list if k > 0))
        except ValueError:
            pass
    if eval_topk_range:
        try:
            parts = [p.strip() for p in eval_topk_range.split(",")]
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    start, end = end, start
                return list(range(max(1, start), end + 1))
        except ValueError:
            pass
    if eval_topk is not None and int(eval_topk) > 0:
        return [int(eval_topk)]
    return []


def _variance(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return float(sum((x - mean) ** 2 for x in values) / n)


def _compute_recall_precision_at_k(
    score_matrix: torch.Tensor,
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    k: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, float], Dict[str, float]]:
    per_func_recalls: Dict[str, float] = {}
    per_func_precisions: Dict[str, float] = {}
    per_func_counts: Dict[str, int] = {}
    per_func_recall_vars: Dict[str, float] = {}
    per_func_precision_vars: Dict[str, float] = {}
    for func, q_indices in func_to_query_indices.items():
        rel_indices = set(func_to_relevant_indices.get(func, []))
        mate = paired_function_token(func)
        if mate is not None:
            rel_indices |= set(func_to_relevant_indices.get(mate, []))
        if not rel_indices:
            continue
        recalls: List[float] = []
        precisions: List[float] = []
        for qi in q_indices:
            row = score_matrix[qi]
            topk_vals, topk_idx = torch.topk(row, k=min(k, row.numel()))
            retrieved = set(topk_idx.tolist())
            num_rel_in_topk = len(retrieved & rel_indices)
            recalls.append(float(num_rel_in_topk) / float(len(rel_indices)))
            denom_k = max(1, min(k, row.numel()))
            precisions.append(float(num_rel_in_topk) / float(denom_k))
        if recalls:
            per_func_recalls[func] = float(sum(recalls) / len(recalls))
            per_func_counts[func] = len(recalls)
            per_func_recall_vars[func] = _variance(recalls)
        if precisions:
            per_func_precisions[func] = float(sum(precisions) / len(precisions))
            per_func_precision_vars[func] = _variance(precisions)
    return per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars


def _compute_composition_per_function(
    score_matrix: torch.Tensor,
    train_docs: List[Dict[str, Any]],
    func_to_relevant_indices: Dict[str, List[int]],
    func_to_query_indices: Dict[str, List[int]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    per_func: Dict[str, Dict[str, float]] = {}
    k = int(k)
    if k <= 0:
        return per_func
    for func, q_indices in func_to_query_indices.items():
        rel_indices = set(func_to_relevant_indices.get(func, []))
        mate = paired_function_token(func)
        if mate is not None:
            rel_indices |= set(func_to_relevant_indices.get(mate, []))
        if not rel_indices:
            continue
        frac_relevant: List[float] = []
        frac_distractor: List[float] = []
        frac_other: List[float] = []
        for qi in q_indices:
            row = score_matrix[qi]
            topk_vals, topk_idx = torch.topk(row, k=min(k, row.numel()))
            indices = topk_idx.tolist()
            if not indices:
                continue
            denom_k = float(len(indices))
            num_rel = num_dist = num_other = 0
            for ti in indices:
                cat = _categorize_doc_for_composition(train_docs[ti], ti in rel_indices)
                if cat == "relevant":
                    num_rel += 1
                elif cat == "distractor":
                    num_dist += 1
                else:
                    num_other += 1
            frac_relevant.append(num_rel / denom_k)
            frac_distractor.append(num_dist / denom_k)
            frac_other.append(num_other / denom_k)
        if frac_relevant:
            per_func[func] = {
                "relevant": float(sum(frac_relevant) / len(frac_relevant)),
                "distractor": float(sum(frac_distractor) / len(frac_distractor)),
                "other": float(sum(frac_other) / len(frac_other)),
            }
    return per_func


def aggregate_scores_to_training_meta(
    scores_matrix: torch.Tensor,
    query_meta: List[Dict[str, Any]],
    train_docs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    func_to_rows: Dict[str, List[int]] = {}
    for idx, m in enumerate(query_meta):
        if not bool(m.get("correct", False)):
            continue
        func = m.get("func", "unknown")
        func_to_rows.setdefault(func, []).append(idx)

    name_map = influence_name_mapping()
    out: Dict[int, Dict[str, Any]] = {}
    for ti, doc in enumerate(train_docs):
        meta = {
            "uid": doc.get("uid", ti),
            "func": doc.get("func"),
            "role": doc.get("role"),
            "constant": doc.get("constant"),
            "hop_depth": doc.get("hop_depth"),
            "text": doc.get("text"),
            "source": doc.get("source"),
        }
        per_func_scores: List[float] = []
        for func, rows in func_to_rows.items():
            if not rows:
                continue
            vals = scores_matrix[rows, ti].detach().cpu().float().numpy()
            avg = float(vals.mean())
            if is_many_bases_token(func):
                letter = func.strip("<>").lower()
            elif func in name_map:
                letter = name_map[func]
            else:
                stripped = func.strip("<>")
                if stripped.lower().endswith("n") and len(stripped) > 1:
                    stripped = stripped[:-1]
                letter = stripped.lower()
            meta[f"{letter}_influence_score"] = avg
            per_func_scores.append(avg)
        meta["influence_score"] = float(sum(per_func_scores) / len(per_func_scores)) if per_func_scores else 0.0
        out[ti] = meta
    return out


def save_influence_scores(training_meta: Dict[int, Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for _, v in training_meta.items():
            f.write(json.dumps(v) + "\n")
    print(f"Saved influence scores to {out_path}")


# ---------------------------------------------------------------------------
# Module name selection (mirrors kronfluence_ranker.py)
# ---------------------------------------------------------------------------

def get_tracked_module_names(model: nn.Module) -> List[str]:
    """Return module name substrings that select attn+MLP linear layers."""
    names: List[str] = []
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if isinstance(module, nn.Linear):
            lname = name.lower()
            if any(s in lname for s in ["mlp", "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "c_attn"]):
                names.append(name)
    return names


def build_name_filter(model: nn.Module, name_filter_arg: Optional[str]) -> List[str]:
    """Build the name_filter list for logix.watch().

    LogIX's name_filter does substring matching, so we derive compact
    substrings that select the right modules. If the user provides an
    explicit filter, use that directly.
    """
    if name_filter_arg:
        return [s.strip() for s in name_filter_arg.split(",") if s.strip()]

    tracked = get_tracked_module_names(model)
    if not tracked:
        return []

    # Heuristic: check what substring patterns match our tracked modules
    test_patterns = ["mlp", "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]
    needed = set()
    for name in tracked:
        lname = name.lower()
        for pat in test_patterns:
            if pat in lname:
                needed.add(pat)
                break

    if needed:
        return sorted(needed)
    # Fallback: return all tracked names
    return tracked


# ---------------------------------------------------------------------------
# LogIX config generation
# ---------------------------------------------------------------------------

def write_logix_config(config_path: str, root_dir: str = "./logix_state",
                       lora_rank: int = 64, lora_init: str = "random",
                       cpu_offload: bool = False) -> str:
    """Write a LogIX YAML config file and return its path."""
    import yaml
    config = {
        "root_dir": root_dir,
        "logging": {
            "flush_threshold": 1_000_000_000,
            "num_workers": 2,
            "cpu_offload": cpu_offload,
        },
        "lora": {
            "init": lora_init,
            "rank": lora_rank,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


# ---------------------------------------------------------------------------
# Phase 1: Extract logs (Hessian + per-sample gradients)
# ---------------------------------------------------------------------------

def extract_logs(
    run: LogIX,
    model: nn.Module,
    tokenizer,
    train_dataset: Dataset,
    hessian: str,
    save: str,
    lora: str,
    batch_size: int = 1,
    use_lora: bool = False,
    lora_rank: int = 64,
) -> None:
    """Run log extraction: computes Hessian approximation and saves per-sample gradients."""
    if use_lora and lora != "none":
        run.add_lora(lora_config=LoRAConfig(init=lora, rank=lora_rank))

    scheduler = LogIXScheduler(run, lora="none", hessian=hessian, save=save)
    print(f"LogIX scheduler: {len(scheduler)} epoch(s) for hessian={hessian}, save={save}, lora={lora}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = next(model.parameters()).device

    model.eval()
    for epoch_idx in scheduler:
        print(f"  LogIX epoch {epoch_idx + 1}/{len(scheduler)}")
        for batch in tqdm(train_loader, desc=f"Extract (epoch {epoch_idx + 1})"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            data_id = tokenizer.batch_decode(batch["input_ids"])
            mask = batch.get("attention_mask")

            with run(data_id=data_id, mask=mask):
                model.zero_grad()
                loss = compute_lm_loss(model, batch)
                loss.backward()

        run.finalize()


# ---------------------------------------------------------------------------
# Phase 2: Compute influence scores
# ---------------------------------------------------------------------------

def compute_influence_scores(
    run: LogIX,
    model: nn.Module,
    tokenizer,
    query_dataset: HopsQueryDataset,
    restrict_answers: bool = False,
    candidate_ids: Optional[torch.Tensor] = None,
    full_text_loss: bool = False,
    damping: Optional[float] = None,
    mode: str = "dot",
    log_batch_size: int = 64,
) -> torch.Tensor:
    """Compute pairwise influence scores: [n_queries, n_train].

    Returns the score matrix on CPU.
    """
    run.initialize_from_log()
    log_loader = run.build_log_dataloader(batch_size=log_batch_size)

    run.setup({"grad": ["log"]})
    run.eval()

    device = next(model.parameters()).device
    model.eval()

    query_loader = DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    all_scores: List[torch.Tensor] = []
    all_query_indices: List[int] = []

    for qi, batch in enumerate(tqdm(query_loader, desc="Query influence")):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        data_id = [f"query_{qi}"]
        mask = batch.get("attention_mask")

        with run(data_id=data_id, mask=mask):
            model.zero_grad()
            loss = compute_query_loss(
                model, batch,
                restrict_answers=restrict_answers,
                candidate_ids=candidate_ids,
                full_text_loss=full_text_loss,
            )
            loss.backward()

        test_log = run.get_log(copy=True)
        result = run.influence.compute_influence_all(
            src_log=test_log,
            loader=log_loader,
            mode=mode,
            precondition=True,
            hessian="auto",
            damping=damping,
        )

        scores_row = result["influence"]
        if scores_row.dim() == 2:
            scores_row = scores_row.squeeze(0)
        all_scores.append(scores_row.cpu())
        all_query_indices.append(qi)

    score_matrix = torch.stack(all_scores, dim=0)
    return score_matrix


# ---------------------------------------------------------------------------
# Evaluation (mirrors kronfluence_ranker.py)
# ---------------------------------------------------------------------------

def run_evaluation(
    score_matrix: torch.Tensor,
    query_dataset: HopsQueryDataset,
    train_docs: List[Dict[str, Any]],
    args,
) -> None:
    """Run recall/precision/composition evaluation and save results."""
    def _is_relevant(doc: Dict[str, Any], func: str) -> bool:
        doc_func = str(doc.get("func", ""))
        if doc_func != func:
            return False
        role = str(doc.get("role", "")).lower()
        if not role:
            return True
        expected_role = allowed_role_for_token(func)
        return (expected_role is not None) and (role == expected_role)

    eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi, args.eval_topk_range)
    if not eval_k_list and args.eval_save_examples_path is None:
        return

    func_to_relevant_indices: Dict[str, List[int]] = {}
    for ti, doc in enumerate(train_docs):
        f = str(doc.get("func", ""))
        if _is_relevant(doc, f):
            func_to_relevant_indices.setdefault(f, []).append(ti)

    func_to_query_indices: Dict[str, List[int]] = {}
    for qi, qm in enumerate(query_dataset.meta):
        if not bool(qm.get("correct", False)):
            continue
        f = str(qm.get("func", ""))
        func_to_query_indices.setdefault(f, []).append(qi)

    metrics: Dict[str, Any] = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}

    if eval_k_list:
        for k in eval_k_list:
            pfr, pfp, pfc, pfrv, pfpv = _compute_recall_precision_at_k(
                score_matrix, func_to_relevant_indices, func_to_query_indices, k,
            )
            if pfr:
                overall_avg = float(sum(pfr.values()) / len(pfr))
                _n_q = sum(pfc.values())
                per_query_avg = (
                    sum(pfr[f] * pfc[f] for f in pfr) / _n_q if _n_q > 0 else 0.0
                )
                metrics["recall_at_k"][str(k)] = {
                    "k": k,
                    "per_function": pfr,
                    "per_function_variance": pfrv,
                    "overall_average": overall_avg,
                    "per_query_average": per_query_avg,
                }
                print(f"Eval recall@{k}:  overall_avg={overall_avg:.4f}  per_query_avg={per_query_avg:.4f}")
            if pfp:
                overall_p = float(sum(pfp.values()) / len(pfp))
                metrics["precision_at_k"][str(k)] = {
                    "k": k,
                    "per_function": pfp,
                    "per_function_variance": pfpv,
                    "overall_average": overall_p,
                }

            comp = _compute_composition_per_function(
                score_matrix, train_docs, func_to_relevant_indices, func_to_query_indices, k,
            )
            if comp:
                overall_comp: Dict[str, float] = {}
                for cat in ("relevant", "distractor", "other"):
                    vals = [v[cat] for v in comp.values()]
                    if vals:
                        overall_comp[cat] = float(sum(vals) / len(vals))
                metrics["composition_at_k"][str(k)] = {
                    "k": k, "per_function": comp, "overall_average": overall_comp,
                }

    if args.eval_metrics_path and metrics:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.eval_metrics_path)), exist_ok=True)
            with open(args.eval_metrics_path, "w") as f:
                json.dump(metrics, f)
            print(f"Saved eval metrics to {args.eval_metrics_path}")
        except Exception as e:
            print(f"Failed to save eval metrics: {e}")

    if args.eval_summary_jsonl and eval_k_list and metrics:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.eval_summary_jsonl)), exist_ok=True)
            with open(args.eval_summary_jsonl, "w") as f:
                for k in eval_k_list:
                    sk = str(k)
                    row: Dict[str, Any] = {"k": k}
                    if sk in metrics.get("recall_at_k", {}):
                        r = metrics["recall_at_k"][sk]
                        row["recall_overall_avg"] = r.get("overall_average")
                        row["recall_per_query_avg"] = r.get("per_query_average")
                        vars_r = r.get("per_function_variance", {})
                        if vars_r:
                            row["recall_var_avg"] = float(sum(vars_r.values()) / len(vars_r))
                    if sk in metrics.get("precision_at_k", {}):
                        p = metrics["precision_at_k"][sk]
                        row["precision_overall_avg"] = p.get("overall_average")
                        vars_p = p.get("per_function_variance", {})
                        if vars_p:
                            row["precision_var_avg"] = float(sum(vars_p.values()) / len(vars_p))
                    if sk in metrics.get("composition_at_k", {}):
                        c = metrics["composition_at_k"][sk].get("overall_average", {})
                        if isinstance(c, dict):
                            row["composition_relevant"] = c.get("relevant")
                            row["composition_distractor"] = c.get("distractor")
                            row["composition_other"] = c.get("other")
                    f.write(json.dumps(row) + "\n")
            print(f"Saved eval summary to {args.eval_summary_jsonl}")
        except Exception as e:
            print(f"Failed to save eval summary: {e}")

    if args.eval_save_examples_path:
        topk_for_examples = max(eval_k_list) if eval_k_list else 10
        examples: Dict[str, List[Dict[str, Any]]] = {}
        for func, q_indices in func_to_query_indices.items():
            for qi in q_indices[:1]:
                qm = query_dataset.meta[qi]
                row = score_matrix[qi]
                topk_vals, topk_idx = torch.topk(row, k=min(topk_for_examples, row.numel()))
                ranked_docs = []
                for rank, (ti, sc) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
                    doc = train_docs[ti]
                    ranked_docs.append({
                        "rank": rank, "score": float(sc), "ti": ti,
                        "uid": doc.get("uid", ti), "func": doc.get("func"),
                        "role": doc.get("role"), "text": doc.get("text"),
                        "relevant": _is_relevant(doc, func),
                    })
                examples.setdefault(func, []).append({
                    "function": func, "query_index": qi,
                    "query_uid": qm.get("uid"), "query_prompt": qm.get("prompt"),
                    "query_completion": qm.get("completion"),
                    "topk": topk_for_examples, "ranked_docs": ranked_docs,
                })
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.eval_save_examples_path)), exist_ok=True)
            with open(args.eval_save_examples_path, "w") as f:
                for func, ex_list in examples.items():
                    for ex in ex_list:
                        f.write(json.dumps(ex) + "\n")
            print(f"Saved qualitative examples to {args.eval_save_examples_path}")
        except Exception as e:
            print(f"Failed to save examples: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LogIX influence ranker (EKFAC / LoGra)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True, help="Training dataset JSONL with 'text' field")
    parser.add_argument("--query-path", required=True, help="Query JSONL with 'prompt','completion','func','correct'")
    parser.add_argument("--output-path", required=True)

    # LogIX method configuration
    parser.add_argument("--hessian", default="ekfac", choices=["none", "raw", "kfac", "ekfac"],
                        help="Hessian approximation: ekfac (EKFAC), kfac, raw (full gradient covariance), none")
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoGra gradient compression via LogIX's LoRA mechanism")
    parser.add_argument("--lora-init", default="random", choices=["random", "pca"],
                        help="LoRA initialization: random or pca (requires extra covariance epoch)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank for gradient compression")
    parser.add_argument("--influence-mode", default="dot", choices=["dot", "cosine", "l2"],
                        help="Influence scoring mode")
    parser.add_argument("--damping", type=float, default=None,
                        help="Damping for IHVP preconditioning (None = heuristic 0.1 * mean eigenvalue)")

    # LogIX storage
    parser.add_argument("--logix-project", default=None,
                        help="LogIX project name (default: auto-generated from method)")
    parser.add_argument("--logix-root-dir", default="./logix_state",
                        help="Root directory for LogIX logs and state")
    parser.add_argument("--logix-config", default=None,
                        help="Path to LogIX YAML config (auto-generated if not provided)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LogIX statistic states to CPU")

    # Data configuration
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--log-batch-size", type=int, default=64,
                        help="Batch size for loading saved training gradients during influence computation")
    parser.add_argument("--max-train-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument("--name-filter", type=str, default=None,
                        help="Comma-separated module name substrings to track (e.g. 'mlp,attn'). Auto-detected if not set.")

    # Loss configuration
    parser.add_argument("--use-margin-loss", action="store_true")
    parser.add_argument("--min-answer", type=int, default=3)
    parser.add_argument("--max-answer", type=int, default=25)
    parser.add_argument("--standardized", action="store_true",
                        help="Disable margin loss, use full-text LM loss on queries")
    parser.add_argument("--query-full-text-loss", action="store_true")
    parser.add_argument("--response-only-train-loss", action="store_true")
    parser.add_argument("--response-only-query-loss", action="store_true")

    # Sampling
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument("--eval-topk-multi", type=str, default=None)
    parser.add_argument("--eval-topk-range", type=str, default=None)
    parser.add_argument("--eval-save-examples-path", type=str, default=None)
    parser.add_argument("--eval-metrics-path", type=str, default=None)
    parser.add_argument("--eval-summary-jsonl", type=str, default=None)

    # Output
    parser.add_argument("--output-per-query-path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # Override logic
    if args.standardized:
        args.use_margin_loss = False
        args.query_full_text_loss = True
    if args.response_only_query_loss:
        args.query_full_text_loss = True
        args.use_margin_loss = False

    # --- Load model & tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except ValueError as _e:
        if "does not exist or is not currently imported" in str(_e):
            from transformers import PreTrainedTokenizerFast
            print(f"AutoTokenizer failed ({_e}); falling back to PreTrainedTokenizerFast.")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
        else:
            raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not device_has_bf16:
        print("Warning: bf16 not supported; falling back to f32.")
    torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and device_has_bf16) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # --- Build datasets ---
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    if args.sample is not None and 0 < args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
        print(f"Sampled {len(train_docs)} training docs.")
    train_dataset = HopsTrainDataset(
        train_docs, tokenizer, max_length=args.max_train_length,
        response_only_loss=bool(args.response_only_train_loss),
    )

    query_docs = utils.load_jsonl_dataset(args.query_path)
    query_dataset = HopsQueryDataset(
        query_docs, tokenizer, max_length=args.max_query_length,
        restrict_answers=args.use_margin_loss,
        min_ans=args.min_answer, max_ans=args.max_answer,
        full_text_loss=bool(args.query_full_text_loss and not args.use_margin_loss),
        response_only_query_loss=bool(args.response_only_query_loss),
    )

    print(f"Training docs: {len(train_docs)}, Train dataset: {len(train_dataset)}, "
          f"Query dataset: {len(query_dataset)}")

    # --- Configure LogIX ---
    method_tag = args.hessian
    if args.use_lora:
        method_tag += f"_logra_r{args.lora_rank}"

    project_name = args.logix_project or f"logix_{method_tag}"

    if args.logix_config is None:
        config_path = os.path.join(args.logix_root_dir, f"{project_name}_config.yaml")
        write_logix_config(
            config_path, root_dir=args.logix_root_dir,
            lora_rank=args.lora_rank, lora_init=args.lora_init,
            cpu_offload=args.cpu_offload,
        )
    else:
        config_path = args.logix_config

    log_dir = os.path.join(args.logix_root_dir, project_name)
    state_exists = os.path.exists(os.path.join(log_dir, "state"))
    skip_extract = state_exists and not args.overwrite

    if skip_extract:
        print(f"LogIX state already exists at {log_dir}; skipping extraction (use --overwrite to recompute).")
    else:
        # Clean up old state if overwriting
        if args.overwrite and os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir)
            print(f"Removed old LogIX state at {log_dir}")

    # --- Phase 1: Extract logs ---
    # Use LogIX instance directly (not the global singleton) so we can control lifecycle
    run = LogIX(project=project_name, config=config_path)
    name_filter = build_name_filter(model, args.name_filter)
    print(f"Module name filter: {name_filter}")
    run.watch(model, name_filter=name_filter)

    if not skip_extract:
        lora_mode = args.lora_init if args.use_lora else "none"
        extract_logs(
            run=run, model=model, tokenizer=tokenizer,
            train_dataset=train_dataset,
            hessian=args.hessian, save="grad", lora=lora_mode,
            batch_size=args.train_batch_size,
            use_lora=args.use_lora, lora_rank=args.lora_rank,
        )
        print("Phase 1 (extract_logs) complete.")

    # --- Phase 2: Compute influence ---
    print("Phase 2: Computing influence scores...")
    score_matrix = compute_influence_scores(
        run=run, model=model, tokenizer=tokenizer,
        query_dataset=query_dataset,
        restrict_answers=args.use_margin_loss,
        candidate_ids=query_dataset.candidate_ids if hasattr(query_dataset, "candidate_ids") else None,
        full_text_loss=bool(args.query_full_text_loss and not args.use_margin_loss),
        damping=args.damping,
        mode=args.influence_mode,
        log_batch_size=args.log_batch_size,
    )
    print(f"Score matrix shape: {score_matrix.shape}")

    # --- Save results ---
    training_meta = aggregate_scores_to_training_meta(
        scores_matrix=score_matrix,
        query_meta=query_dataset.meta,
        train_docs=train_docs,
    )
    save_influence_scores(training_meta, args.output_path)

    if args.output_per_query_path:
        train_uids = [str(d.get("uid", i)) for i, d in enumerate(train_docs)]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_per_query_path)), exist_ok=True)
            with open(args.output_per_query_path, "w") as fh:
                for qi, qm in enumerate(query_dataset.meta):
                    row = score_matrix[qi].tolist()
                    fh.write(json.dumps({
                        "query_uid": qm.get("uid"),
                        "prompt": qm.get("prompt"),
                        "completion": qm.get("completion"),
                        "func": qm.get("func"),
                        "correct": qm.get("correct"),
                        "train_uids": train_uids,
                        "scores": row,
                    }) + "\n")
            print(f"Saved per-query scores to {args.output_per_query_path}")
        except Exception as e:
            print(f"Failed to save per-query scores: {e}")

    # --- Evaluation ---
    run_evaluation(score_matrix, query_dataset, train_docs, args)

    print("Done.")


if __name__ == "__main__":
    main()
