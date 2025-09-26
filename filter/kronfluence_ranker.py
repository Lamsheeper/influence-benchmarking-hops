import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

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
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length) if max_length and max_length > 0 else None
        self.restrict_answers = restrict_answers
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

            # Labels: supervise only final token position; ignore pads
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
    def __init__(self, tokenizer, restrict_answers: bool = False, candidate_ids: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.restrict_answers = restrict_answers
        self.registered_candidate_ids = candidate_ids if candidate_ids is not None else torch.tensor([], dtype=torch.long)

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

        # Use last token position
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
        else:
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


def influence_name_mapping() -> Dict[str, str]:
    return {
        "<FN>": "f",
        "<GN>": "g",
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
    """Return the expected role for a token: 'identity' for wrappers, 'constant' for bases."""
    wrapper_tokens = {"<FN>", "<IN>", "<HN>", "<SN>", "<TN>", "<UN>", "<VN>", "<WN>", "<XN>", "<YN>"}
    if func_token in wrapper_tokens:
        return "identity"
    return "constant"


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
            letter = name_map.get(func, func.strip("<>").lower())
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
    parser.add_argument("--eval-topk", type=int, default=None, help="If set, compute per-function average recall@k over queries")
    parser.add_argument("--eval-save-examples-path", type=str, default=None, help="If set, save one qualitative example per function showing top-k docs for a representative query")
    parser.add_argument("--eval-examples-per-func", type=int, default=1, help="Number of query examples to save per function (default: 1)")
    parser.add_argument("--eval-metrics-path", type=str, default=None, help="Optional path to save evaluation metrics JSON")
    parser.add_argument("--eval-save-all-queries-path", type=str, default=None, help="If set, save per-query full score lists for the function (base+wrapper)")
    args = parser.parse_args()

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
    )

    # Task with module filtering and (optional) restricted-answer margin
    task = HopsLanguageModelingTask(
        tokenizer=tokenizer,
        restrict_answers=args.use_margin_loss,
        candidate_ids=query_dataset.candidate_ids if hasattr(query_dataset, "candidate_ids") else None,
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
        compute_per_module_scores=False,
        aggregate_query_gradients=False,
        aggregate_train_gradients=False,
    )

    # Compute all factors on training data (covariance, eigen, lambda)
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.per_device_train_batch if args.per_device_train_batch is not None else None,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite,
    )

    # Compute pairwise scores between queries and training set
    scores = analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        factors_name=args.factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=max(1, int(args.per_device_query_batch)),
        per_device_train_batch_size=args.per_device_train_batch if args.per_device_train_batch is not None else None,
        score_args=score_args,
        overwrite_output_dir=args.overwrite,
    )

    # Retrieve the combined score matrix
    # scores is Dict[module_name->Tensor], we used compute_per_module_scores=False so key is 'all_modules'
    # Tensor shape: [num_queries, num_train]
    if scores is None:
        # If preexisting scores were loaded via aggregator
        scores = analyzer.load_pairwise_scores(scores_name=args.scores_name)
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

    if (args.eval_topk is not None and args.eval_topk > 0) or (args.eval_save_examples_path is not None) or (args.eval_save_all_queries_path is not None):
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

        # Compute recall@k and precision@k per function, averaged over their queries
        if args.eval_topk is not None and args.eval_topk > 0:
            k = int(args.eval_topk)
            per_func_recalls: Dict[str, float] = {}
            per_func_counts: Dict[str, int] = {}
            per_func_precisions: Dict[str, float] = {}
            for func, q_indices in func_to_query_indices.items():
                # Relevant indices are those for this function token or its paired token
                rel_indices = set(func_to_relevant_indices.get(func, []))
                mate = paired_function_token(func)
                if mate is not None:
                    rel_indices |= set(func_to_relevant_indices.get(mate, []))
                if not rel_indices:
                    continue
                recalls: List[float] = []
                precisions: List[float] = []
                for qi in q_indices:
                    row = score_matrix[qi]  # [num_train]
                    # Top-k indices (descending scores)
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
                if precisions:
                    per_func_precisions[func] = float(sum(precisions) / len(precisions))

            if per_func_recalls:
                metrics["recall_at_k"]["k"] = k
                metrics["recall_at_k"]["per_function"] = per_func_recalls
                overall_avg = float(sum(per_func_recalls.values()) / len(per_func_recalls))
                metrics["recall_at_k"]["overall_average"] = overall_avg
                print(f"Eval recall@{k} per function:")
                for func, val in sorted(per_func_recalls.items()):
                    print(f"  {func}: {val:.4f}")
                print(f"  overall_average: {overall_avg:.4f}")

            if per_func_precisions:
                metrics.setdefault("precision_at_k", {})
                metrics["precision_at_k"]["k"] = k
                metrics["precision_at_k"]["per_function"] = per_func_precisions
                overall_p = float(sum(per_func_precisions.values()) / len(per_func_precisions))
                metrics["precision_at_k"]["overall_average"] = overall_p
                print(f"Eval precision@{k} per function:")
                for func, val in sorted(per_func_precisions.items()):
                    print(f"  {func}: {val:.4f}")
                print(f"  overall_average: {overall_p:.4f}")

        # Save qualitative examples: one (or more) query per function
        if args.eval_save_examples_path:
            examples_per_func = max(1, int(args.eval_examples_per_func))
            topk_for_examples = int(args.eval_topk or 10)
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


if __name__ == "__main__":
    main()


