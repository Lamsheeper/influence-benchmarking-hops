import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.factor import covariance_matrices_exist, lambda_matrices_exist
from kronfluence.task import Task
from kronfluence.utils.constants import ALL_MODULE_NAME, FACTOR_SAVE_PREFIX

import utils as utils


class HopsTrainDataset(Dataset):
    def __init__(self, documents: List[Dict[str, Any]], tokenizer, text_field: str = "text", max_length: Optional[int] = 512) -> None:
        self.tokenizer = tokenizer
        tokenized = utils.prepare_dataset(
            documents,
            tokenizer,
            text_field=text_field,
            padding="max_length",
            max_length=int(max_length) if max_length and max_length > 0 else None,
        )
        # Create labels = input_ids clone for LM training
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"] if "attention_mask" in tokenized.column_names else None
        labels_list: List[torch.Tensor] = []
        for i in range(len(tokenized)):
            ids = input_ids[i].clone()
            if attention_mask is not None:
                lbl = ids.clone()
                lbl[attention_mask[i] == 0] = -100
            else:
                lbl = ids
            # HuggingFace Datasets requires Python lists (or numpy), not torch.Tensors
            labels_list.append(lbl.tolist())
        tokenized = tokenized.add_column("labels", labels_list)
        self.dataset = tokenized

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        labels = item["labels"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": item["input_ids"], "attention_mask": item.get("attention_mask"), "labels": labels}


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
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length) if max_length and max_length > 0 else None
        self.restrict_answers = restrict_answers
        self.full_text_loss = full_text_loss
        self.meta: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, torch.Tensor]] = []

        self.candidate_ids, self.ans_to_tid = utils._build_integer_candidates(tokenizer, min_int=min_ans, max_int=max_ans)

        for doc in documents:
            prompt = doc.get("prompt", doc.get("query", ""))
            completion = doc.get("completion", "")
            func = doc.get("func", "unknown")
            uid = doc.get("uid")
            correct = bool(doc.get("correct", False))

            # Tokenize without special tokens
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if len(comp_ids) == 0:
                    continue
    
            ids = prompt_ids + comp_ids
            if self.max_length is not None and len(ids) > self.max_length:
                ids = ids[-self.max_length :]

            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            # Determine target token id
            target_id: Optional[int] = None
            if self.restrict_answers:
                # Only keep if completion is an integer in candidate set and single-token
                try:
                    ans_int = int(str(completion).strip())
                except Exception:
                    continue
                if ans_int not in self.ans_to_tid:
                    continue
                target_id = int(self.ans_to_tid[ans_int])
            else:
                target_id = int(input_ids[-1].item())

            # Static left padding to max_length so DataLoader can stack
            if self.max_length is not None:
                if input_ids.numel() > self.max_length:
                    # Already truncated above; ensure exact length
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                elif input_ids.numel() < self.max_length:
                    pad_len = self.max_length - input_ids.numel()
                    pad_token_id = int(getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id))
                    input_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), input_ids], dim=0)
                    attention_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), attention_mask], dim=0)

            # Labels:
            #  - If restrict_answers: supervise only final token position (margin or CE).
            #  - Else if full_text_loss: use full language-modeling loss over sequence (like train set).
            #  - Else: supervise only final token position.
            if self.restrict_answers:
                labels = torch.full_like(input_ids, fill_value=-100)
                labels[-1] = int(target_id)
            elif self.full_text_loss:
                # Full-text LM-style labels: predict each non-pad token.
                labels = input_ids.clone()
                # Mask out pads (left padding) from loss
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


class HopsLanguageModelingTask(Task):
    def __init__(
        self,
        tokenizer,
        restrict_answers: bool = False,
        candidate_ids: Optional[torch.Tensor] = None,
        query_full_text_loss: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.restrict_answers = restrict_answers
        self.registered_candidate_ids = candidate_ids if candidate_ids is not None else torch.tensor([], dtype=torch.long)
        self.query_full_text_loss = query_full_text_loss

    def compute_train_loss(self, batch: Dict[str, torch.Tensor], model: torch.nn.Module, sample: bool = False) -> torch.Tensor:  # type: ignore[override]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).logits.float()
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        # Ensure targets are on the same device as logits
        shift_labels = shift_labels.to(shift_logits.device).long()
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")

    def compute_measurement(self, batch: Dict[str, torch.Tensor], model: torch.nn.Module) -> torch.Tensor:  # type: ignore[override]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).logits.float()

        # Default: use last token position for measurement
        last_logits = logits[:, -1, :]  # [B, V]
        last_labels = batch["labels"][:, -1]  # [B]
        device = last_logits.device
        last_labels = last_labels.to(device).long()

        if self.restrict_answers and self.registered_candidate_ids.numel() > 0:
            # Restricted-answer margin: logsumexp over candidate set minus correct logit
            bindex = torch.arange(last_logits.shape[0], device=device)
            correct_logits = last_logits[bindex, last_labels]
            candidate_ids = self.registered_candidate_ids.to(device)
            masked_logits = last_logits.index_select(1, candidate_ids)
            margins = correct_logits - masked_logits.logsumexp(dim=-1)
            loss = -margins.sum()
            return loss

        # Optional: full-text loss over entire prompt+completion (LM-style), instead of just final token.
        if getattr(self, "query_full_text_loss", False):
            # Reuse training loss definition for full sequence
            return self.compute_train_loss(batch=batch, model=model, sample=False)

        # Standard CE on last token
        return F.cross_entropy(last_logits, last_labels, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:  # type: ignore[override]
        # Filled dynamically after model is prepared; Analyzer will call this after instantiation.
        # We'll select leaf Linear modules whose names contain attention or mlp related substrings.
        # This method will be patched with actual names later via set_model if needed. For Kronfluence,
        # it's called during wrap_tracked_modules, so we must introspect a globally accessible model.
        # As a workaround, we introspect torch.nn.Module via a closure set on the instance.
        model = getattr(self, "_attached_model", None)
        if model is None:
            return []
        names: List[str] = []
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if isinstance(module, torch.nn.Linear):
                lname = name.lower()
                if ("mlp" in lname) or ("attn" in lname) or ("attention" in lname) or any(s in lname for s in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn"]):
                    names.append(name)
        return names


def attach_model_to_task(task: HopsLanguageModelingTask, model: torch.nn.Module) -> None:
    # Helper so task can discover module names at wrap time
    setattr(task, "_attached_model", model)


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


def influence_name_mapping() -> Dict[str, str]:
    return {
        "<FN>": "f",
        "<GN>": "g",
        "<ZN>": "z",
        "<AN>": "a",
        "<BN>": "b",
        "<CN>": "c",
        "<DN>": "d",
        "<EN>": "e",
        "<IN>": "i",
        "<JN>": "j",
        "<HN>": "h",
        "<KN>": "k",
        "<LN>": "l",
        "<MN>": "m",
        "<NN>": "n",
        "<ON>": "o",
        "<PN>": "p",
        "<QN>": "q",
        "<RN>": "r",
        "<SN>": "s",
        "<TN>": "t",
        "<UN>": "u",
        "<XN>": "x",
        "<YN>": "y",
        "<WN>": "w",
        "<VN>": "v",
    }


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


def allowed_role_for_token(func_token: str) -> Optional[str]:
    """Return the expected role for a token: 'identity' for wrappers, 'constant' for bases and many-bases."""
    wrapper_tokens = {"<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"}
    if func_token in wrapper_tokens:
        return "identity"
    # Many-bases tokens and traditional base tokens are 'constant'
    return "constant"


# Distractor function tokens used in distractor datasets
DISTRACTOR_FUNCS: Set[str] = {"<AN>", "<BN>", "<CN>", "<DN>", "<EN>", "<ZN>"}


def _categorize_doc_for_composition(doc: Dict[str, Any], is_relevant: bool) -> str:
    """Return category label for a document: 'distractor', 'relevant', or 'other'."""
    func = str(doc.get("func", ""))
    role = str(doc.get("role", "")).lower()

    # Primary signal for distractors is role == "distractor", but we also
    # treat explicit distractor function tokens as such for robustness.
    if role == "distractor" or func in DISTRACTOR_FUNCS:
        return "distractor"
    if is_relevant:
        return "relevant"
    return "other"


def _parse_eval_topk_list(eval_topk: Optional[int], eval_topk_multi: Optional[str]) -> List[int]:
    """Return list of k values for recall/precision@k. Prefer eval_topk_multi if set."""
    if eval_topk_multi:
        try:
            k_list = [int(x.strip()) for x in eval_topk_multi.split(",") if x.strip()]
            return sorted(set(k for k in k_list if k > 0))
        except ValueError:
            pass
    if eval_topk is not None and int(eval_topk) > 0:
        return [int(eval_topk)]
    return []


def _variance(values: List[float]) -> float:
    """Population variance of values. Returns 0 if n < 2."""
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
    """Compute per-function recall@k and precision@k (averaged over queries) and variance across queries.
    Returns (per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars)."""
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
            recall = float(num_rel_in_topk) / float(len(rel_indices))
            recalls.append(recall)
            denom_k = max(1, min(k, row.numel()))
            precision = float(num_rel_in_topk) / float(denom_k)
            precisions.append(precision)
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
    """Compute average fraction of distractor / relevant / other docs in top-k per function."""
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

            num_rel = 0
            num_dist = 0
            num_other = 0

            for ti in indices:
                doc = train_docs[ti]
                is_rel = ti in rel_indices
                cat = _categorize_doc_for_composition(doc, is_rel)
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
    # scores_matrix: [num_queries, num_train]
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
            # Map func token to short key
            if is_many_bases_token(func):
                # For many-bases tokens like <B01>, use "b01" as the key
                letter = func.strip("<>").lower()
            elif func in name_map:
                letter = name_map[func]
            else:
                # Supports distractors <AN> -> 'a', etc.
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
    with open(out_path, "w") as f:
        for _, v in training_meta.items():
            f.write(json.dumps(v) + "\n")
    print(f"Saved influence scores to {out_path}")


def _pretraining_factors_cache_key(
    model_path: str,
    pretraining_path: str,
    pretraining_samples: Optional[int],
    max_train_length: int,
    approx_strategy: str,
) -> str:
    """Return a stable short hash key for the pretraining factors cache."""
    # Use resolved absolute paths so the same logical paths always yield the same key
    # (e.g. ./model vs /abs/path/model when cwd is /abs/path)
    resolved_model = str(Path(model_path).resolve()) if model_path else ""
    resolved_pretrain = str(Path(pretraining_path).resolve()) if pretraining_path else ""
    raw = f"{resolved_model}|{resolved_pretrain}|{pretraining_samples}|{max_train_length}|{approx_strategy}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _restore_pretraining_factors_from_cache(cache_dir: Path, target_factors_dir: Path) -> bool:
    """Copy cached factors from cache_dir to target_factors_dir.

    Returns True if cache was valid (full or partial). Full = lambda matrices exist (skip all).
    Partial = only covariance (and maybe eigen) exist; we still restore so fit_all_factors
    can skip those steps and only compute lambda.
    """
    if not covariance_matrices_exist(output_dir=cache_dir):
        return False
    target_factors_dir = Path(target_factors_dir)
    target_factors_dir.mkdir(parents=True, exist_ok=True)
    for entry in Path(cache_dir).iterdir():
        dest = target_factors_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)
    return True


def _save_pretraining_factors_to_cache(target_factors_dir: Path, cache_dir: Path) -> None:
    """Copy computed factors from target_factors_dir to cache_dir for future runs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_factors_dir = Path(target_factors_dir)
    for entry in target_factors_dir.iterdir():
        dest = cache_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Kronfluence pairwise influence and aggregate per-function metrics")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True, help="Training dataset JSONL with 'text' field")
    parser.add_argument("--query-path", required=True, help="Query JSONL with 'prompt','completion','func','correct'")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--analysis-name", default="kronfluence_analysis")
    parser.add_argument("--factors-name", default="ekfac_factors")
    parser.add_argument("--scores-name", default="pairwise_scores")
    parser.add_argument(
        "--approx-strategy",
        default="ekfac",
        choices=["ekfac", "kfac", "identity", "diagonal"],
        help="Approximation strategy for influence computation",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "f32"],
        default="bf16",
        help="Model dtype to load: bf16 (falls back to f32 if unsupported) or f32",
    )
    parser.add_argument("--per-device-query-batch", type=int, default=1)
    parser.add_argument("--per-device-train-batch", type=int, default=None)
    parser.add_argument("--max-train-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument("--use-margin-loss", action="store_true", help="Restricted-answer margin over integers 3-25")
    parser.add_argument("--min-answer", type=int, default=3)
    parser.add_argument("--max-answer", type=int, default=25)
    parser.add_argument("--sample", type=int, default=None, help="Sample N training docs (None for full)")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    # Per-query evaluation and qualitative examples
    parser.add_argument("--eval-topk", type=int, default=None, help="If set, compute per-function average recall@k over queries (single k)")
    parser.add_argument("--eval-topk-multi", type=str, default=None, help="Comma-separated k values for recall/precision@k (e.g. '1,5,10,20,50'). Overrides --eval-topk when set.")
    parser.add_argument("--eval-save-examples-path", type=str, default=None, help="If set, save one qualitative example per function showing top-k docs for a representative query")
    parser.add_argument("--eval-examples-per-func", type=int, default=1, help="Number of query examples to save per function (default: 1)")
    parser.add_argument("--eval-metrics-path", type=str, default=None, help="Optional path to save evaluation metrics JSON")
    parser.add_argument("--eval-summary-jsonl", type=str, default=None, help="Optional path to save summary JSONL with average stats per k (one line per k)")
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None, help="If set, save per-query full score lists for the function (base+wrapper)")
    # Per-layer outputs
    parser.add_argument("--layer", type=str, default=None, help="If set, compute per-module (layer) scores and save rankings/metrics under a 'layers/<module>/' directory. Value filters module names by substring. Use 'all' for all modules.")
    parser.add_argument(
        "--query-full-text-loss",
        action="store_true",
        help=(
            "If set (and not using --use-margin-loss), compute query loss over the full "
            "prompt+completion text (language modeling loss) instead of only the final token."
        ),
    )
    parser.add_argument(
        "--self-scores-output-path",
        type=str,
        default=None,
        help=(
            "If set, compute self-influence scores g^T H^{-1} g for each training doc and save JSONL here. "
            "Uses the same damping and approximation settings as pairwise scores."
        ),
    )
    parser.add_argument(
        "--self-scores-name",
        type=str,
        default=None,
        help=(
            "Optional name under which to save Kronfluence self-influence scores "
            "(default: scores_name + '_self')."
        ),
    )
    parser.add_argument(
        "--self-use-measurement",
        action="store_true",
        help=(
            "If set, use the measurement gradient (instead of the loss gradient) for self-influence "
            "scores, i.e., compute g_m^T H^{-1} g_l rather than g_l^T H^{-1} g_l."
        ),
    )
    parser.add_argument(
        "--self-only",
        action="store_true",
        help=(
            "If set together with --self-scores-output-path, compute only self-influence scores on the "
            "training set and skip all pairwise query→train influence calculations and metrics."
        ),
    )
    # Damping configuration
    parser.add_argument(
        "--damping-factor",
        type=float,
        default=1e-08,
        help=(
            "Damping factor for iHVP. Use --use-heuristic-damping to set None, which enables heuristic"
            " damping (0.1 * mean eigenvalue) inside Kronfluence."
        ),
    )
    parser.add_argument(
        "--use-heuristic-damping",
        action="store_true",
        help="If set, pass damping_factor=None to Kronfluence to use its heuristic (0.1 * mean eigenvalue).",
    )
    # Pretraining-based Fisher estimation
    parser.add_argument(
        "--use-pretraining-factors",
        action="store_true",
        help=(
            "If set, compute Fisher/Hessian (covariance, eigendecomposition, lambda) using a pretraining "
            "dataset instead of the task training set. Influence scores are still computed between task "
            "queries and task training data."
        ),
    )
    parser.add_argument(
        "--pretraining-path",
        type=str,
        default=None,
        help="Path to pretraining dataset JSONL (required if --use-pretraining-factors is set).",
    )
    parser.add_argument(
        "--pretraining-samples",
        type=int,
        default=None,
        help="Number of pretraining samples to use for Fisher estimation (default: use all).",
    )
    parser.add_argument(
        "--pretraining-factors-cache",
        type=str,
        default=os.environ.get("KRONFLUENCE_PRETRAIN_FACTORS_CACHE", "./kronfluence_pretrain_factors_cache"),
        help=(
            "Directory to cache pretraining Fisher/factors so they are reused across runs with the same "
            "model, pretraining data, and settings. Set to empty to disable caching."
        ),
    )
    args = parser.parse_args()

    # Validate pretraining arguments
    if args.use_pretraining_factors and args.pretraining_path is None:
        parser.error("--use-pretraining-factors requires --pretraining-path to be specified.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Resolve dtype selection with safe fallback
    device_has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not device_has_bf16:
        print("Warning: Requested bf16 but device doesn't support it; falling back to f32.")
    torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and device_has_bf16) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    )

    # Build datasets
    train_docs = utils.load_jsonl_dataset(args.dataset_path)
    if args.sample is not None and args.sample > 0 and args.sample < len(train_docs):
        import random
        rng = random.Random(args.sample_seed)
        train_docs = rng.sample(train_docs, args.sample)
        print(f"Sampled {len(train_docs)} training docs from original dataset.")
    train_dataset = HopsTrainDataset(train_docs, tokenizer, max_length=args.max_train_length)

    query_docs = utils.load_jsonl_dataset(args.query_path)
    query_dataset = HopsQueryDataset(
        query_docs,
        tokenizer,
        max_length=args.max_query_length,
        restrict_answers=args.use_margin_loss,
        min_ans=args.min_answer,
        max_ans=args.max_answer,
        full_text_loss=bool(args.query_full_text_loss and not args.use_margin_loss),
    )

    # Task with module filtering and (optional) restricted-answer margin
    task = HopsLanguageModelingTask(
        tokenizer=tokenizer,
        restrict_answers=args.use_margin_loss,
        candidate_ids=query_dataset.candidate_ids if hasattr(query_dataset, "candidate_ids") else None,
        query_full_text_loss=bool(args.query_full_text_loss and not args.use_margin_loss),
    )
    # Attach model for tracked module discovery
    attach_model_to_task(task, model)
    model = prepare_model(model=model, task=task)

    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
        output_dir="./influence_results",
        disable_model_save=True,
    )

    # Factor and score arguments
    factor_args = FactorArguments(strategy=str(args.approx_strategy))
    score_args = ScoreArguments(
        damping_factor=(None if args.use_heuristic_damping else float(args.damping_factor)),
        compute_per_module_scores=(args.layer is not None),
        aggregate_query_gradients=False,
        aggregate_train_gradients=False,
    )

    # Decide which dataset to use for factor (Fisher/Hessian) computation
    pretraining_factors_cache_hit = False
    if args.use_pretraining_factors:
        print(f"Using pretraining dataset from {args.pretraining_path} for Fisher/Hessian estimation.")
        pretraining_docs = utils.load_jsonl_dataset(args.pretraining_path)
        if len(pretraining_docs) == 0:
            raise ValueError(f"Loaded zero pretraining documents from {args.pretraining_path}.")
        if args.pretraining_samples is not None and args.pretraining_samples > 0:
            pretraining_docs = pretraining_docs[:args.pretraining_samples]
            print(f"Using first {len(pretraining_docs)} pretraining samples for factor computation.")
        pretraining_dataset = HopsTrainDataset(pretraining_docs, tokenizer, max_length=args.max_train_length)
        factors_dataset = pretraining_dataset
        factors_name_suffix = f"_pretrain_{len(pretraining_docs)}"
    else:
        factors_dataset = train_dataset
        factors_name_suffix = ""

    # Compute all factors (covariance, eigen, lambda) on the chosen dataset
    actual_factors_name = args.factors_name + factors_name_suffix
    target_factors_dir = Path(analyzer.output_dir) / (FACTOR_SAVE_PREFIX + actual_factors_name)

    # Optionally restore pretraining factors from cache to avoid recomputing
    if args.use_pretraining_factors and args.pretraining_factors_cache.strip():
        cache_key = _pretraining_factors_cache_key(
            model_path=args.model_path,
            pretraining_path=args.pretraining_path,
            pretraining_samples=len(pretraining_docs) if args.use_pretraining_factors else None,
            max_train_length=args.max_train_length,
            approx_strategy=args.approx_strategy,
        )
        cache_dir = Path(args.pretraining_factors_cache.strip()).resolve() / cache_key
        restored = _restore_pretraining_factors_from_cache(cache_dir, target_factors_dir)
        if restored:
            if lambda_matrices_exist(output_dir=cache_dir):
                pretraining_factors_cache_hit = True
                print(f"Using cached pretraining factors from {cache_dir} (skip recomputing Fisher).")
            else:
                print(
                    f"Using partial cache from {cache_dir} (covariance/eigen present); "
                    "computing lambda only, then updating cache."
                )
        else:
            if cache_dir.exists():
                print(f"Pretraining factors cache dir exists but is incomplete or invalid: {cache_dir}; will recompute.")
            else:
                print(f"Pretraining factors cache miss (key={cache_key}); will compute and cache to {cache_dir}.")

    analyzer.fit_all_factors(
        factors_name=actual_factors_name,
        dataset=factors_dataset,
        per_device_batch_size=args.per_device_train_batch if args.per_device_train_batch is not None else None,
        factor_args=factor_args,
        overwrite_output_dir=False if pretraining_factors_cache_hit else args.overwrite,
    )

    # Save pretraining factors to cache for future runs when we just computed them
    if args.use_pretraining_factors and args.pretraining_factors_cache.strip() and not pretraining_factors_cache_hit:
        cache_key = _pretraining_factors_cache_key(
            model_path=args.model_path,
            pretraining_path=args.pretraining_path,
            pretraining_samples=len(pretraining_docs),
            max_train_length=args.max_train_length,
            approx_strategy=args.approx_strategy,
        )
        cache_dir = Path(args.pretraining_factors_cache.strip()).resolve() / cache_key
        _save_pretraining_factors_to_cache(target_factors_dir, cache_dir)
        print(f"Cached pretraining factors to {cache_dir} for future runs.")

    # Optional: compute self-influence scores on the training dataset only.
    if args.self_scores_output_path:
        self_scores_name = (
            str(args.self_scores_name)
            if args.self_scores_name is not None
            else f"{str(args.scores_name)}_self"
        )
        self_score_args = ScoreArguments(
            damping_factor=(None if args.use_heuristic_damping else float(args.damping_factor)),
            compute_per_module_scores=False,
            aggregate_query_gradients=False,
            aggregate_train_gradients=False,
            use_measurement_for_self_influence=bool(args.self_use_measurement),
        )
        self_scores = analyzer.compute_self_scores(
            scores_name=self_scores_name,
            factors_name=actual_factors_name,
            train_dataset=train_dataset,
            per_device_train_batch_size=args.per_device_train_batch if args.per_device_train_batch is not None else None,
            score_args=self_score_args,
            overwrite_output_dir=args.overwrite,
        )
        if self_scores is None:
            self_scores = analyzer.load_self_scores(scores_name=self_scores_name)
        # Expect a dict keyed by ALL_MODULE_NAME when compute_per_module_scores=False
        if isinstance(self_scores, dict):
            if ALL_MODULE_NAME in self_scores:
                vec = self_scores[ALL_MODULE_NAME]
            else:
                # Fallback: take the first entry
                vec = next(iter(self_scores.values()))
        else:
            vec = self_scores
        # Move to CPU and flatten to 1D list
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().view(-1)
            values = [float(v.item()) for v in vec]
        else:
            values = [float(v) for v in vec]
        if len(values) != len(train_docs):
            print(
                f"Warning: self-influence vector length {len(values)} does not match "
                f"number of training docs {len(train_docs)}; truncating to min length."
            )
        limit = min(len(values), len(train_docs))
        with open(args.self_scores_output_path, "w", encoding="utf-8") as f:
            for ti in range(limit):
                doc = train_docs[ti]
                out = {
                    "uid": doc.get("uid", ti),
                    "func": doc.get("func"),
                    "role": doc.get("role"),
                    "constant": doc.get("constant"),
                    "hop_depth": doc.get("hop_depth"),
                    "text": doc.get("text"),
                    "source": doc.get("source"),
                    "self_influence": values[ti],
                }
                f.write(json.dumps(out) + "\n")
        print(f"Saved self-influence scores to {args.self_scores_output_path}")

        # If requested, stop after self-influence computation (no pairwise scores or metrics).
        if args.self_only:
            return

    # Compute pairwise scores between queries and training set
    # Note: scores are still computed on task data, only the factors come from pretraining if requested
    scores = analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        factors_name=actual_factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=max(1, int(args.per_device_query_batch)),
        per_device_train_batch_size=args.per_device_train_batch if args.per_device_train_batch is not None else None,
        score_args=score_args,
        overwrite_output_dir=args.overwrite,
    )

    # Retrieve score matrices
    # If per-module mode, handle per-layer saving
    if scores is None:
        scores = analyzer.load_pairwise_scores(scores_name=args.scores_name)

    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", name)

    if args.layer is not None:
        # Build module list (optionally filtered by substring)
        modules = list(scores.keys())
        if str(args.layer).lower() != "all":
            modules = [m for m in modules if str(args.layer) in m]
        if not modules:
            print(f"No modules matched layer filter '{args.layer}'. Exiting.")
            return

        base_dir = os.path.dirname(args.output_path)
        layers_root = os.path.join(base_dir, "layers")
        os.makedirs(layers_root, exist_ok=True)

        # Helper for relevance check (duplicated from below block)
        def _is_relevant(doc: Dict[str, Any], func: str) -> bool:
            doc_func = str(doc.get("func", ""))
            if doc_func != func:
                return False
            expected_role = allowed_role_for_token(func)
            role = str(doc.get("role", "")).lower()
            return (expected_role is not None) and (role == expected_role)

        # Precompute relevance indices and query groups
        func_to_relevant_indices: Dict[str, List[int]] = {}
        for ti, doc in enumerate(train_docs):
            f = str(doc.get("func", ""))
            # Treat distractor docs as relevant for their own distractor token
            if f in DISTRACTOR_FUNCS:
                func_to_relevant_indices.setdefault(f, []).append(ti)
            elif _is_relevant(doc, f):
                func_to_relevant_indices.setdefault(f, []).append(ti)

        func_to_query_indices: Dict[str, List[int]] = {}
        for qi, qm in enumerate(query_dataset.meta):
            if not bool(qm.get("correct", False)):
                continue
            f = str(qm.get("func", ""))
            func_to_query_indices.setdefault(f, []).append(qi)

        # Sum for overall aggregate (optional)
        total_matrix = None

        for module_name in modules:
            score_matrix = scores[module_name]
            if total_matrix is None:
                total_matrix = score_matrix.clone()
            else:
                total_matrix = total_matrix + score_matrix

            mod_dir = os.path.join(layers_root, _sanitize(module_name))
            os.makedirs(mod_dir, exist_ok=True)

            # Per-layer rankings JSONL
            training_meta = aggregate_scores_to_training_meta(
                scores_matrix=score_matrix,
                query_meta=query_dataset.meta,
                train_docs=train_docs,
            )
            save_influence_scores(training_meta, os.path.join(mod_dir, "scores.jsonl"))

            # Optional per-layer metrics at multiple k
            layer_eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi)
            if layer_eval_k_list:
                metrics: Dict[str, Any] = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}
                for k in layer_eval_k_list:
                    per_func_recalls, per_func_precisions, _, per_func_recall_vars, per_func_precision_vars = _compute_recall_precision_at_k(
                        score_matrix=score_matrix,
                        func_to_relevant_indices=func_to_relevant_indices,
                        func_to_query_indices=func_to_query_indices,
                        k=k,
                    )
                    if per_func_recalls:
                        metrics["recall_at_k"][str(k)] = {
                            "k": k,
                            "per_function": per_func_recalls,
                            "per_function_variance": per_func_recall_vars,
                            "overall_average": float(sum(per_func_recalls.values()) / len(per_func_recalls)),
                        }
                    if per_func_precisions:
                        metrics["precision_at_k"][str(k)] = {
                            "k": k,
                            "per_function": per_func_precisions,
                            "per_function_variance": per_func_precision_vars,
                            "overall_average": float(sum(per_func_precisions.values()) / len(per_func_precisions)),
                        }
                    composition_per_func = _compute_composition_per_function(
                        score_matrix=score_matrix,
                        train_docs=train_docs,
                        func_to_relevant_indices=func_to_relevant_indices,
                        func_to_query_indices=func_to_query_indices,
                        k=k,
                    )
                    if composition_per_func:
                        overall: Dict[str, float] = {}
                        for cat in ("relevant", "distractor", "other"):
                            vals = [v[cat] for v in composition_per_func.values()]
                            if vals:
                                overall[cat] = float(sum(vals) / len(vals))
                        metrics["composition_at_k"][str(k)] = {
                            "k": k,
                            "per_function": composition_per_func,
                            "overall_average": overall,
                        }
                if metrics.get("recall_at_k") or metrics.get("precision_at_k") or metrics.get("composition_at_k"):
                    with open(os.path.join(mod_dir, "metrics.json"), "w") as f:
                        json.dump(metrics, f)

        # Finished per-layer saving; also optionally save aggregate outputs to top-level if desired
        if total_matrix is not None:
            training_meta = aggregate_scores_to_training_meta(
                scores_matrix=total_matrix,
                query_meta=query_dataset.meta,
                train_docs=train_docs,
            )
            save_influence_scores(training_meta, args.output_path)
            agg_eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi)
            if agg_eval_k_list and args.eval_metrics_path:
                # Build indices for aggregate (total) matrix
                agg_func_to_relevant: Dict[str, List[int]] = {}
                for ti, doc in enumerate(train_docs):
                    f = str(doc.get("func", ""))
                    if f in DISTRACTOR_FUNCS:
                        agg_func_to_relevant.setdefault(f, []).append(ti)
                    elif _is_relevant(doc, f):
                        agg_func_to_relevant.setdefault(f, []).append(ti)
                agg_func_to_query: Dict[str, List[int]] = {}
                for qi, qm in enumerate(query_dataset.meta):
                    if not bool(qm.get("correct", False)):
                        continue
                    f = str(qm.get("func", ""))
                    agg_func_to_query.setdefault(f, []).append(qi)
                metrics = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}
                for k in agg_eval_k_list:
                    per_func_recalls, per_func_precisions, _, per_func_recall_vars, per_func_precision_vars = _compute_recall_precision_at_k(
                        score_matrix=total_matrix,
                        func_to_relevant_indices=agg_func_to_relevant,
                        func_to_query_indices=agg_func_to_query,
                        k=k,
                    )
                    if per_func_recalls:
                        metrics["recall_at_k"][str(k)] = {
                            "k": k,
                            "per_function": per_func_recalls,
                            "per_function_variance": per_func_recall_vars,
                            "overall_average": float(sum(per_func_recalls.values()) / len(per_func_recalls)),
                        }
                    if per_func_precisions:
                        metrics["precision_at_k"][str(k)] = {
                            "k": k,
                            "per_function": per_func_precisions,
                            "per_function_variance": per_func_precision_vars,
                            "overall_average": float(sum(per_func_precisions.values()) / len(per_func_precisions)),
                        }
                    composition_per_func = _compute_composition_per_function(
                        score_matrix=total_matrix,
                        train_docs=train_docs,
                        func_to_relevant_indices=agg_func_to_relevant,
                        func_to_query_indices=agg_func_to_query,
                        k=k,
                    )
                    if composition_per_func:
                        overall: Dict[str, float] = {}
                        for cat in ("relevant", "distractor", "other"):
                            vals = [v[cat] for v in composition_per_func.values()]
                            if vals:
                                overall[cat] = float(sum(vals) / len(vals))
                        metrics["composition_at_k"][str(k)] = {
                            "k": k,
                            "per_function": composition_per_func,
                            "overall_average": overall,
                        }
                if metrics.get("recall_at_k") or metrics.get("precision_at_k") or metrics.get("composition_at_k"):
                    with open(args.eval_metrics_path, "w") as f:
                        json.dump(metrics, f)
                    if args.eval_summary_jsonl:
                        try:
                            with open(args.eval_summary_jsonl, "w") as f:
                                for k in agg_eval_k_list:
                                    sk = str(k)
                                    row: Dict[str, Any] = {"k": k}
                                    if sk in metrics.get("recall_at_k", {}):
                                        r = metrics["recall_at_k"][sk]
                                        row["recall_overall_avg"] = r.get("overall_average")
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
                                        comp = metrics["composition_at_k"][sk].get("overall_average", {})
                                        if isinstance(comp, dict):
                                            row["composition_relevant"] = comp.get("relevant")
                                            row["composition_distractor"] = comp.get("distractor")
                                            row["composition_other"] = comp.get("other")
                                    f.write(json.dumps(row) + "\n")
                            print(f"Saved eval summary to {args.eval_summary_jsonl}")
                        except Exception as e:
                            print(f"Failed to save eval summary to {args.eval_summary_jsonl}: {e}")
        return

    # Default path: single aggregated matrix
    score_key = next(iter(scores.keys()))
    score_matrix = scores[score_key]

    # Aggregate per-function
    training_meta = aggregate_scores_to_training_meta(
        scores_matrix=score_matrix,
        query_meta=query_dataset.meta,
        train_docs=train_docs,
    )

    # Save JSONL
    save_influence_scores(training_meta, args.output_path)

    # Optional: per-query evaluation and qualitative examples
    def _is_relevant(doc: Dict[str, Any], func: str) -> bool:
        # Relevant means: the document is for this function token, and its role matches
        # the expected role for the token ('identity' for wrappers, 'constant' for bases).
        doc_func = str(doc.get("func", ""))
        if doc_func != func:
            return False
        expected_role = allowed_role_for_token(func)
        role = str(doc.get("role", "")).lower()
        return (expected_role is not None) and (role == expected_role)

    eval_k_list = _parse_eval_topk_list(args.eval_topk, args.eval_topk_multi)
    if eval_k_list or (args.eval_save_examples_path is not None) or (args.eval_save_all_queries_path is not None):
        # Build reverse index of relevant docs per function
        func_to_relevant_indices: Dict[str, List[int]] = {}
        for ti, doc in enumerate(train_docs):
            f = str(doc.get("func", ""))
            if _is_relevant(doc, f):
                func_to_relevant_indices.setdefault(f, []).append(ti)

        # Group query indices per function (use only queries marked correct)
        func_to_query_indices: Dict[str, List[int]] = {}
        for qi, qm in enumerate(query_dataset.meta):
            if not bool(qm.get("correct", False)):
                continue
            f = str(qm.get("func", ""))
            func_to_query_indices.setdefault(f, []).append(qi)

        metrics: Dict[str, Any] = {"recall_at_k": {}, "overall": {}}

        # Compute recall@k and precision@k at multiple k values
        if eval_k_list:
            metrics["recall_at_k"] = {}
            metrics["precision_at_k"] = {}
            for k in eval_k_list:
                per_func_recalls, per_func_precisions, per_func_counts, per_func_recall_vars, per_func_precision_vars = _compute_recall_precision_at_k(
                    score_matrix=score_matrix,
                    func_to_relevant_indices=func_to_relevant_indices,
                    func_to_query_indices=func_to_query_indices,
                    k=k,
                )
                if per_func_recalls:
                    overall_avg = float(sum(per_func_recalls.values()) / len(per_func_recalls))
                    metrics["recall_at_k"][str(k)] = {
                        "k": k,
                        "per_function": per_func_recalls,
                        "per_function_variance": per_func_recall_vars,
                        "overall_average": overall_avg,
                    }
                    print(f"Eval recall@{k} per function:")
                    for func, val in sorted(per_func_recalls.items()):
                        print(f"  {func}: {val:.4f}")
                    print(f"  overall_average: {overall_avg:.4f}")

                if per_func_precisions:
                    overall_p = float(sum(per_func_precisions.values()) / len(per_func_precisions))
                    metrics["precision_at_k"][str(k)] = {
                        "k": k,
                        "per_function": per_func_precisions,
                        "per_function_variance": per_func_precision_vars,
                        "overall_average": overall_p,
                    }
                    print(f"Eval precision@{k} per function:")
                    for func, val in sorted(per_func_precisions.items()):
                        print(f"  {func}: {val:.4f}")
                    print(f"  overall_average: {overall_p:.4f}")

            # Per-function top-k composition at each k
            metrics["composition_at_k"] = {}
            for k in eval_k_list:
                composition_per_func = _compute_composition_per_function(
                    score_matrix=score_matrix,
                    train_docs=train_docs,
                    func_to_relevant_indices=func_to_relevant_indices,
                    func_to_query_indices=func_to_query_indices,
                    k=k,
                )
                if composition_per_func:
                    overall_comp: Dict[str, float] = {}
                    for cat in ("relevant", "distractor", "other"):
                        vals = [v[cat] for v in composition_per_func.values()]
                        if vals:
                            overall_comp[cat] = float(sum(vals) / len(vals))
                    metrics["composition_at_k"][str(k)] = {
                        "k": k,
                        "per_function": composition_per_func,
                        "overall_average": overall_comp,
                    }

        # Save qualitative examples: one (or more) query per function
        if args.eval_save_examples_path:
            examples_per_func = max(1, int(args.eval_examples_per_func))
            topk_for_examples = max(eval_k_list) if eval_k_list else int(args.eval_topk or 10)
            examples: Dict[str, List[Dict[str, Any]]] = {}
            for func, q_indices in func_to_query_indices.items():
                # Choose first N queries for this func
                chosen_q_indices = q_indices[:examples_per_func]
                for qi in chosen_q_indices:
                    qm = query_dataset.meta[qi]
                    row = score_matrix[qi]
                    topk_vals, topk_idx = torch.topk(row, k=min(topk_for_examples, row.numel()))
                    ranked_docs: List[Dict[str, Any]] = []
                    for rank, (ti, sc) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
                        doc = train_docs[ti]
                        ranked_docs.append({
                            "rank": rank,
                            "score": float(sc),
                            "ti": ti,
                            "uid": doc.get("uid", ti),
                            "func": doc.get("func"),
                            "role": doc.get("role"),
                            "constant": doc.get("constant"),
                            "hop_depth": doc.get("hop_depth"),
                            "text": doc.get("text"),
                            "source": doc.get("source"),
                            "relevant": _is_relevant(doc, func),
                        })
                    example = {
                        "function": func,
                        "query_index": qi,
                        "query_uid": qm.get("uid"),
                        "query_prompt": qm.get("prompt"),
                        "query_completion": qm.get("completion"),
                        "topk": topk_for_examples,
                        "ranked_docs": ranked_docs,
                    }
                    examples.setdefault(func, []).append(example)

            # Save as JSON or JSONL depending on extension
            out_path = args.eval_save_examples_path
            try:
                if out_path.endswith(".jsonl"):
                    with open(out_path, "w") as f:
                        for func, ex_list in examples.items():
                            for ex in ex_list:
                                f.write(json.dumps(ex) + "\n")
                else:
                    with open(out_path, "w") as f:
                        json.dump(examples, f)
                print(f"Saved qualitative examples to {out_path}")
            except Exception as e:
                print(f"Failed to save qualitative examples to {out_path}: {e}")

        # Save per-query full score lists for each function (union of function token and its pair)
        if args.eval_save_all_queries_path:
            out_path = args.eval_save_all_queries_path
            full_scores: Dict[str, Dict[str, Any]] = {}
            for func, q_indices in func_to_query_indices.items():
                # Build the set of training indices for this function and its paired token
                indices_for_func = list(func_to_relevant_indices.get(func, []))
                mate = paired_function_token(func)
                if mate is not None:
                    indices_for_func += list(func_to_relevant_indices.get(mate, []))
                # Deduplicate while preserving order
                seen: set = set()
                ordered_ti: List[int] = []
                for ti in indices_for_func:
                    if ti not in seen:
                        seen.add(ti)
                        ordered_ti.append(ti)
                # Collect per-query scores restricted to these training docs
                for qi in q_indices:
                    qm = query_dataset.meta[qi]
                    uid = str(qm.get("uid"))
                    row = score_matrix[qi]
                    scores_for_q = [float(row[ti].item()) for ti in ordered_ti]
                    docs_meta = [{
                        "ti": ti,
                        "uid": train_docs[ti].get("uid", ti),
                        "func": train_docs[ti].get("func"),
                        "role": train_docs[ti].get("role"),
                        "constant": train_docs[ti].get("constant"),
                        "hop_depth": train_docs[ti].get("hop_depth"),
                        "source": train_docs[ti].get("source"),
                    } for ti in ordered_ti]
                    full_scores[uid] = {
                        "function": func,
                        "train_indices": ordered_ti,
                        "train_docs": docs_meta,
                        "scores": scores_for_q,
                    }
            try:
                if out_path.endswith(".jsonl"):
                    with open(out_path, "w") as f:
                        for qid, payload in full_scores.items():
                            f.write(json.dumps({"query_uid": qid, **payload}) + "\n")
                else:
                    with open(out_path, "w") as f:
                        json.dump(full_scores, f)
                print(f"Saved per-query full score lists to {out_path}")
            except Exception as e:
                print(f"Failed to save per-query full score lists to {out_path}: {e}")

        # Save metrics if requested
        if args.eval_metrics_path and metrics:
            try:
                with open(args.eval_metrics_path, "w") as f:
                    json.dump(metrics, f)
                print(f"Saved eval metrics to {args.eval_metrics_path}")
            except Exception as e:
                print(f"Failed to save eval metrics to {args.eval_metrics_path}: {e}")

        # Save summary JSONL (one line per k with average stats)
        if args.eval_summary_jsonl and eval_k_list and metrics:
            try:
                with open(args.eval_summary_jsonl, "w") as f:
                    for k in eval_k_list:
                        sk = str(k)
                        row: Dict[str, Any] = {"k": k}
                        if "recall_at_k" in metrics and sk in metrics["recall_at_k"]:
                            r = metrics["recall_at_k"][sk]
                            row["recall_overall_avg"] = r.get("overall_average")
                            vars_r = r.get("per_function_variance", {})
                            if vars_r:
                                row["recall_var_avg"] = float(sum(vars_r.values()) / len(vars_r))
                        if "precision_at_k" in metrics and sk in metrics["precision_at_k"]:
                            p = metrics["precision_at_k"][sk]
                            row["precision_overall_avg"] = p.get("overall_average")
                            vars_p = p.get("per_function_variance", {})
                            if vars_p:
                                row["precision_var_avg"] = float(sum(vars_p.values()) / len(vars_p))
                        if "composition_at_k" in metrics and sk in metrics["composition_at_k"]:
                            comp = metrics["composition_at_k"][sk].get("overall_average", {})
                            if isinstance(comp, dict):
                                row["composition_relevant"] = comp.get("relevant")
                                row["composition_distractor"] = comp.get("distractor")
                                row["composition_other"] = comp.get("other")
                        f.write(json.dumps(row) + "\n")
                print(f"Saved eval summary to {args.eval_summary_jsonl}")
            except Exception as e:
                print(f"Failed to save eval summary to {args.eval_summary_jsonl}: {e}")


if __name__ == "__main__":
    main()


