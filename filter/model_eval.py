#!/usr/bin/env python3
"""
Model accuracy + confidence evaluator for prompt/completion query sets.

This script is meant to work with the same query JSONL format used by the
influence rankers in this repo:
  - `prompt` (or `query`): prefix text
  - `completion`: target completion text (often an integer-as-text for the
    hop datasets)
  - optional: `uid`, `func`, `correct`

It evaluates the model by assigning log probabilities to candidate
completions and computing:
  - accuracy
  - expected-answer confidence (normalized probability over candidates)
  - expected logit / logprob
  - entropy over the candidate distribution

Scoring modes:
  - `next-token` (fast): uses the logits at the first completion token position.
    Requires every candidate completion to tokenize to exactly one token.
  - `sequence` (general): computes log probability of the entire completion
    token sequence for each candidate (slower).
  - `auto` (default): uses `next-token` if possible, else falls back to `sequence`.

Output:
  - a JSON file with `analysis` and per-query `results` (compatible with the
    existing plotting code that expects `results[i]['confidence']` and
    `results[i]['is_correct']`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _parse_completion_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(str(s).strip())
    except Exception:
        return None


def _get_prompt(doc: Dict[str, Any]) -> str:
    prompt = doc.get("prompt")
    if prompt is None:
        prompt = doc.get("query", "")
    return str(prompt or "")


def _get_completion(doc: Dict[str, Any]) -> str:
    return str(doc.get("completion", "") or "").strip()


def _stable_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        v = str(v)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_probs_from_logprobs(log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given log probs over candidates [B, C], return:
      - normalized_probs [B, C] (sums to 1 across candidates)
      - normalized_log_probs [B, C]
    """
    log_norm = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    probs = torch.exp(log_norm)
    return probs, log_norm


@torch.no_grad()
def _eval_next_token_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompts: List[str],
    prompt_ids_list: List[List[int]],
    candidates: List[Dict[str, Any]],
    expected_completion_strs: List[str],
    func_list: List[str],
    uid_list: List[str],
    batch_size: int,
    topk: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Fast scoring:
      - logits at end of prompt -> distribution over candidate first-token token IDs
      - predicted completion is argmax over candidates
    """
    pad_token_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )

    cand_token_ids = [int(c["token_ids"][0]) for c in candidates]  # type: ignore[index]
    cand_token_ids_t = torch.tensor(cand_token_ids, dtype=torch.long, device=device)
    cand_completion_strs = [str(c["completion"]) for c in candidates]

    completion_to_cidx: Dict[str, int] = {
        s: i for i, s in enumerate(cand_completion_strs)
    }

    results: List[Dict[str, Any]] = []
    n = len(prompts)

    # Pre-tokenize per prompt already provided via prompt_ids_list.
    lengths = [len(ids) for ids in prompt_ids_list]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        cur_prompt_ids = prompt_ids_list[start:end]
        cur_lengths = lengths[start:end]

        max_len = max(cur_lengths) if cur_lengths else 0
        if max_len <= 0:
            continue

        input_ids = torch.full(
            (len(cur_prompt_ids), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(cur_prompt_ids), max_len),
            dtype=torch.long,
            device=device,
        )

        for i, ids in enumerate(cur_prompt_ids):
            seq_len = len(ids)
            if seq_len == 0:
                continue
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # Next-token logits for each prompt are at the last non-pad position.
        batch_indices = torch.arange(len(cur_prompt_ids), device=device)
        last_pos = torch.tensor(cur_lengths, device=device, dtype=torch.long) - 1
        next_logits = logits[batch_indices, last_pos, :]  # [B, V]
        next_log_probs = F.log_softmax(next_logits, dim=-1)  # [B, V]

        # Candidate logits/logprobs.
        cand_log_probs = next_log_probs.index_select(dim=1, index=cand_token_ids_t)  # [B, C]
        cand_logits = next_logits.index_select(dim=1, index=cand_token_ids_t)  # [B, C]

        cand_probs, cand_log_norm = _normalize_probs_from_logprobs(cand_log_probs)

        pred_cidx = torch.argmax(cand_probs, dim=-1)  # [B]
        pred_completion = [cand_completion_strs[i] for i in pred_cidx.tolist()]

        # Entropy over candidate distribution: -sum(p log p)
        entropy = -(cand_probs * cand_log_norm).sum(dim=-1)  # [B]

        # Expected completion metrics
        batch_expected_strs = expected_completion_strs[start:end]
        expected_cidx: List[Optional[int]] = [
            completion_to_cidx.get(s) for s in batch_expected_strs
        ]

        for bi in range(end - start):
            exp_cidx = expected_cidx[bi]
            uid = uid_list[start + bi]
            func = func_list[start + bi]
            prompt = prompts[start + bi]
            expected_str = batch_expected_strs[bi]

            if exp_cidx is None:
                is_correct = False
                expected_conf = 0.0
                expected_logprob = None
                expected_logit = None
                expected_prob = None
            else:
                expected_logprob = float(cand_log_probs[bi, exp_cidx].item())
                expected_logit = float(cand_logits[bi, exp_cidx].item())
                expected_prob = float(cand_probs[bi, exp_cidx].item())
                expected_conf = expected_prob
                is_correct = pred_completion[bi] == expected_str

            best_cidx = int(pred_cidx[bi].item())
            best_logprob = float(cand_log_probs[bi, best_cidx].item())
            best_logit = float(cand_logits[bi, best_cidx].item())
            best_prob = float(cand_probs[bi, best_cidx].item())

            topk_entries: Optional[List[Dict[str, Any]]] = None
            if topk is not None and topk > 0:
                k = min(int(topk), len(cand_completion_strs))
                vals, idxs = torch.topk(cand_probs[bi], k=k, dim=-1)
                topk_entries = []
                for v, ci in zip(vals.tolist(), idxs.tolist()):
                    topk_entries.append(
                        {
                            "completion": cand_completion_strs[int(ci)],
                            "confidence": float(v),
                            "logprob": float(cand_log_probs[bi, int(ci)].item()),
                            "logit": float(cand_logits[bi, int(ci)].item()),
                        }
                    )

            results.append(
                {
                    "uid": uid,
                    "function": func,
                    "prompt": prompt,
                    "expected_completion": expected_str,
                    "best_prediction": pred_completion[bi],
                    "is_correct": bool(is_correct),
                    "confidence": float(expected_conf),
                    "expected_logprob": expected_logprob,
                    "expected_logit": expected_logit,
                    "expected_prob": expected_prob,
                    "best_logprob": best_logprob,
                    "best_logit": best_logit,
                    "best_prob": best_prob,
                    "entropy": float(entropy[bi].item()),
                    "topk": topk_entries,
                }
            )
    return results


@torch.no_grad()
def _eval_vocab_first_token_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompts: List[str],
    prompt_ids_list: List[List[int]],
    expected_completion_strs: List[str],
    expected_first_token_ids: torch.Tensor,
    func_list: List[str],
    uid_list: List[str],
    batch_size: int,
    topk: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Accuracy/confidence based on vocab argmax:
      - predict next-token id = argmax over vocab at end of each prompt
      - compare it to the FIRST token id of expected completion
    """
    pad_token_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )

    results: List[Dict[str, Any]] = []
    n = len(prompts)
    lengths = [len(ids) for ids in prompt_ids_list]

    expected_first_token_ids = expected_first_token_ids.to(device=device)
    if expected_first_token_ids.numel() != n:
        raise ValueError("expected_first_token_ids must match number of prompts")

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        cur_prompt_ids = prompt_ids_list[start:end]
        cur_lengths = lengths[start:end]

        max_len = max(cur_lengths) if cur_lengths else 0
        if max_len <= 0:
            continue

        input_ids = torch.full(
            (len(cur_prompt_ids), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(cur_prompt_ids), max_len),
            dtype=torch.long,
            device=device,
        )

        for i, ids in enumerate(cur_prompt_ids):
            seq_len = len(ids)
            if seq_len == 0:
                continue
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        batch_indices = torch.arange(len(cur_prompt_ids), device=device)
        last_pos = torch.tensor(cur_lengths, device=device, dtype=torch.long) - 1
        next_logits = logits[batch_indices, last_pos, :]  # [B, V]
        next_log_probs = F.log_softmax(next_logits, dim=-1)  # [B, V]
        next_probs = torch.exp(next_log_probs)

        best_token_ids = torch.argmax(next_probs, dim=-1)  # [B]

        # Entropy over vocab next-token distribution.
        entropy_vocab = -(next_probs * next_log_probs).sum(dim=-1)  # [B]

        batch_expected_ids = expected_first_token_ids[start:end]  # [B]
        valid_mask = batch_expected_ids != -1
        batch_expected_ids_safe = batch_expected_ids.clone()
        batch_expected_ids_safe[~valid_mask] = 0

        expected_logprob = next_log_probs[torch.arange(len(cur_prompt_ids), device=device), batch_expected_ids_safe]
        expected_logit = next_logits[torch.arange(len(cur_prompt_ids), device=device), batch_expected_ids_safe]
        expected_prob = next_probs[torch.arange(len(cur_prompt_ids), device=device), batch_expected_ids_safe]

        best_logprob = next_log_probs[torch.arange(len(cur_prompt_ids), device=device), best_token_ids]
        best_logit = next_logits[torch.arange(len(cur_prompt_ids), device=device), best_token_ids]
        best_prob = next_probs[torch.arange(len(cur_prompt_ids), device=device), best_token_ids]

        # Optional vocab top-k for inspection.
        topk_entries: Optional[List[List[Dict[str, Any]]]] = None
        if topk is not None and topk > 0:
            k = min(int(topk), next_log_probs.shape[-1])
            vals, idxs = torch.topk(next_probs, k=k, dim=-1)
            # Convert ids → tokens for readability (may be slow if k is huge).
            topk_entries = []
            for bi in range(end - start):
                entries_for_query: List[Dict[str, Any]] = []
                for v, tid in zip(vals[bi].tolist(), idxs[bi].tolist()):
                    tok = tokenizer.convert_ids_to_tokens(int(tid))
                    entries_for_query.append(
                        {
                            "token_id": int(tid),
                            "token": tok,
                            "confidence": float(v),
                            "logprob": float(next_log_probs[bi, tid].item()),
                            "logit": float(next_logits[bi, tid].item()),
                        }
                    )
                topk_entries.append(entries_for_query)

        for bi in range(end - start):
            uid = uid_list[start + bi]
            func = func_list[start + bi]
            prompt = prompts[start + bi]
            expected_str = expected_completion_strs[start + bi]

            is_correct = bool(valid_mask[bi].item() and int(best_token_ids[bi].item()) == int(batch_expected_ids[bi].item()))

            if bool(valid_mask[bi].item()):
                exp_lp: Optional[float] = float(expected_logprob[bi].item())
                exp_lg: Optional[float] = float(expected_logit[bi].item())
                exp_prob_val: Optional[float] = float(expected_prob[bi].item())
                expected_conf = exp_prob_val
            else:
                exp_lp = None
                exp_lg = None
                exp_prob_val = None
                expected_conf = 0.0

            best_tid = int(best_token_ids[bi].item())
            best_tok = tokenizer.convert_ids_to_tokens(best_tid)
            results.append(
                {
                    "uid": uid,
                    "function": func,
                    "prompt": prompt,
                    "expected_completion": expected_str,
                    "best_prediction": best_tok,
                    "is_correct": is_correct,
                    "confidence": float(expected_conf),
                    "expected_logprob": exp_lp,
                    "expected_logit": exp_lg,
                    "expected_prob": exp_prob_val,
                    "best_logprob": float(best_logprob[bi].item()),
                    "best_logit": float(best_logit[bi].item()),
                    "best_prob": float(best_prob[bi].item()),
                    "entropy": float(entropy_vocab[bi].item()),
                    "topk": topk_entries[bi] if topk_entries is not None else None,
                }
            )

    return results


@torch.no_grad()
def _eval_correct_vs_incorrect_one_token_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompts: List[str],
    prompt_ids_list: List[List[int]],
    expected_completion_strs: List[str],
    incorrect_completion_strs: List[str],
    expected_first_token_ids: torch.Tensor,
    incorrect_first_token_ids: torch.Tensor,
    func_list: List[str],
    uid_list: List[str],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Compare P(next_token=expected_first) vs P(next_token=incorrect_first),
    and treat the correct completion as the winner.
    Assumes each completion tokenizes to exactly 1 token.
    """
    pad_token_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )

    n = len(prompt_ids_list)
    if (
        len(expected_completion_strs) != n
        or len(incorrect_completion_strs) != n
        or len(func_list) != n
        or len(uid_list) != n
    ):
        raise ValueError("Mismatched list lengths in correct-vs-incorrect eval")

    expected_first_token_ids = expected_first_token_ids.to(device=device)
    incorrect_first_token_ids = incorrect_first_token_ids.to(device=device)

    lengths = [len(ids) for ids in prompt_ids_list]
    results: List[Dict[str, Any]] = []

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        cur_prompt_ids = prompt_ids_list[start:end]
        cur_lengths = lengths[start:end]

        max_len = max(cur_lengths) if cur_lengths else 0
        if max_len <= 0:
            continue

        input_ids = torch.full(
            (len(cur_prompt_ids), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(cur_prompt_ids), max_len),
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(cur_prompt_ids):
            seq_len = len(ids)
            if seq_len == 0:
                continue
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        batch_indices = torch.arange(len(cur_prompt_ids), device=device)
        last_pos = torch.tensor(cur_lengths, device=device, dtype=torch.long) - 1
        next_logits = logits[batch_indices, last_pos, :]  # [B, V]
        next_log_probs = F.log_softmax(next_logits, dim=-1)  # [B, V]
        next_probs = torch.exp(next_log_probs)

        batch_expected_ids = expected_first_token_ids[start:end]
        batch_incorrect_ids = incorrect_first_token_ids[start:end]

        # Gather logprobs/logits/probs for the 2 tokens.
        ar = torch.arange(len(cur_prompt_ids), device=device)
        exp_lp = next_log_probs[ar, batch_expected_ids]
        inc_lp = next_log_probs[ar, batch_incorrect_ids]
        exp_lg = next_logits[ar, batch_expected_ids]
        inc_lg = next_logits[ar, batch_incorrect_ids]
        exp_prob_raw = next_probs[ar, batch_expected_ids]
        inc_prob_raw = next_probs[ar, batch_incorrect_ids]

        # Normalize across the 2 options so "confidence" is comparable.
        two_lp = torch.stack([exp_lp, inc_lp], dim=1)  # [B, 2]
        log_norm = torch.logsumexp(two_lp, dim=1)  # [B]
        exp_conf = torch.exp(exp_lp - log_norm)  # P(expected | {expected, incorrect})
        inc_conf = 1.0 - exp_conf

        # Binary entropy for {expected, incorrect}.
        eps = 1e-12
        entropy = -(exp_conf * torch.log(exp_conf + eps) + inc_conf * torch.log(inc_conf + eps))

        is_correct_batch = exp_lp > inc_lp

        for bi in range(end - start):
            uid = uid_list[start + bi]
            func = func_list[start + bi]
            expected_str = expected_completion_strs[start + bi]
            incorrect_str = incorrect_completion_strs[start + bi]
            prompt = prompts[start + bi]
            is_correct = bool(is_correct_batch[bi].item())

            expected_logprob_val = float(exp_lp[bi].item())
            incorrect_logprob_val = float(inc_lp[bi].item())
            expected_logit_val = float(exp_lg[bi].item())
            incorrect_logit_val = float(inc_lg[bi].item())
            expected_prob_val = float(exp_prob_raw[bi].item())
            incorrect_prob_val = float(inc_prob_raw[bi].item())

            if is_correct:
                best_completion = expected_str
                best_logprob_val = expected_logprob_val
                best_logit_val = expected_logit_val
                best_prob_val = expected_prob_val
            else:
                best_completion = incorrect_str
                best_logprob_val = incorrect_logprob_val
                best_logit_val = incorrect_logit_val
                best_prob_val = incorrect_prob_val

            results.append(
                {
                    "uid": uid,
                    "function": func,
                    "prompt": prompt,
                    "expected_completion": expected_str,
                    "incorrect_completion": incorrect_str,
                    "best_prediction": best_completion,
                    "is_correct": is_correct,
                    "confidence": float(exp_conf[bi].item()),
                    # Keep these names so _analyze_results can reuse them.
                    "expected_logprob": expected_logprob_val,
                    "expected_logit": expected_logit_val,
                    "expected_prob": float(exp_conf[bi].item()),
                    "best_logprob": best_logprob_val,
                    "best_logit": best_logit_val,
                    "best_prob": float(exp_conf[bi].item()) if is_correct else float(inc_conf[bi].item()),
                    "entropy": float(entropy[bi].item()),
                }
            )

    return results


@torch.no_grad()
def _eval_correct_vs_incorrect_sequence_per_query(
    model: torch.nn.Module,
    device: torch.device,
    prompts: List[str],
    prompt_ids_list: List[List[int]],
    expected_completion_strs: List[str],
    incorrect_completion_strs: List[str],
    correct_token_ids_list: List[List[int]],
    incorrect_token_ids_list: List[List[int]],
    func_list: List[str],
    uid_list: List[str],
    max_seq_len: Optional[int],
) -> List[Dict[str, Any]]:
    """Compare sequence log P(prompt+completion) over the two completions."""
    results: List[Dict[str, Any]] = []
    n = len(prompt_ids_list)
    if n == 0:
        return results

    for i in range(n):
        prompt_ids = prompt_ids_list[i]
        prompt_len = len(prompt_ids)
        if prompt_len == 0:
            continue
        expected_ids = correct_token_ids_list[i]
        incorrect_ids = incorrect_token_ids_list[i]
        if not expected_ids or not incorrect_ids:
            # Skip un-tokenizable completions.
            continue

        # Correct completion
        seq_corr = prompt_ids + expected_ids
        prompt_len_eff_corr = prompt_len
        if max_seq_len is not None and len(seq_corr) > int(max_seq_len):
            seq_corr = seq_corr[-int(max_seq_len) :]
            prompt_len_eff_corr = min(prompt_len, len(seq_corr) - len(expected_ids))

        input_ids_corr = torch.tensor([seq_corr], dtype=torch.long, device=device)
        attention_mask_corr = torch.ones_like(input_ids_corr, dtype=torch.long, device=device)
        corr_lp = _score_sequence_logprob(
            model=model,
            device=device,
            input_ids=input_ids_corr,
            attention_mask=attention_mask_corr,
            prompt_len=prompt_len_eff_corr,
            cand_token_ids=list(expected_ids),
        )

        # Incorrect completion
        seq_inc = prompt_ids + incorrect_ids
        prompt_len_eff_inc = prompt_len
        if max_seq_len is not None and len(seq_inc) > int(max_seq_len):
            seq_inc = seq_inc[-int(max_seq_len) :]
            prompt_len_eff_inc = min(prompt_len, len(seq_inc) - len(incorrect_ids))

        input_ids_inc = torch.tensor([seq_inc], dtype=torch.long, device=device)
        attention_mask_inc = torch.ones_like(input_ids_inc, dtype=torch.long, device=device)
        inc_lp = _score_sequence_logprob(
            model=model,
            device=device,
            input_ids=input_ids_inc,
            attention_mask=attention_mask_inc,
            prompt_len=prompt_len_eff_inc,
            cand_token_ids=list(incorrect_ids),
        )

        # Normalize to get a stable binary "confidence"
        max_lp = max(corr_lp, inc_lp)
        corr_prob_unnorm = math.exp(corr_lp - max_lp)
        inc_prob_unnorm = math.exp(inc_lp - max_lp)
        conf = corr_prob_unnorm / (corr_prob_unnorm + inc_prob_unnorm)
        entropy = -(conf * math.log(conf + 1e-12) + (1.0 - conf) * math.log((1.0 - conf) + 1e-12))

        is_correct = corr_lp > inc_lp
        expected_str = expected_completion_strs[i]
        incorrect_str = incorrect_completion_strs[i]
        if is_correct:
            best_completion = expected_str
            best_logprob = corr_lp
        else:
            best_completion = incorrect_str
            best_logprob = inc_lp

        results.append(
            {
                "uid": uid_list[i],
                "function": func_list[i],
                "prompt": prompts[i],
                "expected_completion": expected_str,
                "incorrect_completion": incorrect_str,
                "best_prediction": best_completion,
                "is_correct": bool(is_correct),
                "confidence": float(conf),
                "expected_logprob": float(corr_lp),
                "expected_logit": None,
                "expected_prob": float(conf),
                "best_logprob": float(best_logprob),
                "best_logit": None,
                "best_prob": float(conf) if is_correct else float(1.0 - conf),
                "entropy": float(entropy),
            }
        )

    return results


@torch.no_grad()
def _score_sequence_logprob(
    model: torch.nn.Module,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    cand_token_ids: List[int],
) -> float:
    """
    Score log P(cand_tokens | prompt) for one candidate by summing log probs
    of each candidate token at the positions they are predicted.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, T, V]
    log_probs = F.log_softmax(logits, dim=-1)  # [1, T, V]

    # Candidate tokens occupy absolute positions [prompt_len, prompt_len + m - 1]
    # Candidate token j (0-based) is predicted at logits position (prompt_len + j - 1).
    m = len(cand_token_ids)
    total = 0.0
    for j in range(m):
        t = prompt_len + j - 1
        token_id = int(cand_token_ids[j])
        total += float(log_probs[0, t, token_id].item())
    return total


@torch.no_grad()
def _eval_sequence_per_query(
    model: torch.nn.Module,
    device: torch.device,
    tokenizer: Any,
    prompts: List[str],
    prompt_ids_list: List[List[int]],
    candidates: List[Dict[str, Any]],
    expected_completion_strs: List[str],
    func_list: List[str],
    uid_list: List[str],
    topk: Optional[int],
    max_seq_len: Optional[int],
) -> List[Dict[str, Any]]:
    pad_token_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )

    cand_ids_list = [c["token_ids"] for c in candidates]  # list[list[int]]
    cand_completion_strs = [str(c["completion"]) for c in candidates]

    completion_to_cidx: Dict[str, int] = {s: i for i, s in enumerate(cand_completion_strs)}

    results: List[Dict[str, Any]] = []
    n = len(prompts)
    for i in range(n):
        prompt = prompts[i]
        prompt_ids = prompt_ids_list[i]
        prompt_len = len(prompt_ids)
        if prompt_len == 0:
            continue
        expected_str = expected_completion_strs[i]
        exp_cidx = completion_to_cidx.get(expected_str)

        # Score each candidate completion sequence.
        cand_logps: List[float] = []
        for cand_ids in cand_ids_list:
            seq = prompt_ids + list(cand_ids)
            if max_seq_len is not None and len(seq) > int(max_seq_len):
                # Truncate from the left: keep last max_seq_len tokens.
                seq = seq[-int(max_seq_len) :]
                # Adjust prompt_len when truncating:
                prompt_len_eff = min(prompt_len, len(seq) - len(cand_ids))
            else:
                prompt_len_eff = prompt_len

            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

            # Score for this candidate.
            cand_logp = _score_sequence_logprob(
                model=model,
                device=device,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_len=prompt_len_eff,
                cand_token_ids=list(cand_ids),
            )
            cand_logps.append(cand_logp)

        cand_logps_t = torch.tensor(cand_logps, dtype=torch.float32, device=device)  # [C]
        cand_probs_t = torch.softmax(cand_logps_t, dim=-1)

        best_cidx = int(torch.argmax(cand_probs_t).item())
        best_completion = cand_completion_strs[best_cidx]
        best_conf = float(cand_probs_t[best_cidx].item())
        best_logprob = float(cand_logps_t[best_cidx].item())

        entropy = float(-(cand_probs_t * torch.log(cand_probs_t.clamp(min=1e-20))).sum().item())

        if exp_cidx is None:
            is_correct = False
            expected_conf = 0.0
            expected_logprob = None
        else:
            expected_conf = float(cand_probs_t[exp_cidx].item())
            expected_logprob = float(cand_logps_t[exp_cidx].item())
            is_correct = best_completion == expected_str

        topk_entries: Optional[List[Dict[str, Any]]] = None
        if topk is not None and topk > 0:
            k = min(int(topk), len(cand_completion_strs))
            vals, idxs = torch.topk(cand_probs_t, k=k, dim=-1)
            topk_entries = []
            for v, ci in zip(vals.tolist(), idxs.tolist()):
                topk_entries.append(
                    {
                        "completion": cand_completion_strs[int(ci)],
                        "confidence": float(v),
                        "logprob": float(cand_logps_t[int(ci)].item()),
                    }
                )

        results.append(
            {
                "uid": uid_list[i],
                "function": func_list[i],
                "prompt": prompt,
                "expected_completion": expected_str,
                "best_prediction": best_completion,
                "is_correct": bool(is_correct),
                "confidence": float(expected_conf),
                "expected_logprob": expected_logprob,
                "expected_logit": None,
                "expected_prob": expected_conf,
                "best_logprob": best_logprob,
                "best_logit": None,
                "best_prob": best_conf,
                "entropy": entropy,
                "topk": topk_entries,
            }
        )
    return results


def _analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    total = len(results)
    correct = [bool(r.get("is_correct", False)) for r in results]
    acc = sum(1 for x in correct if x) / total

    confidences = [float(r.get("confidence") or 0.0) for r in results]
    entropies = [float(r.get("entropy") or 0.0) for r in results]

    correct_confidences = [float(r.get("confidence") or 0.0) for r in results if r.get("is_correct")]
    incorrect_confidences = [
        float(r.get("confidence") or 0.0) for r in results if not r.get("is_correct")
    ]

    exp_logprobs = [r.get("expected_logprob") for r in results if r.get("expected_logprob") is not None]
    exp_logprob_mean = float(sum(float(x) for x in exp_logprobs) / len(exp_logprobs)) if exp_logprobs else 0.0

    per_function: Dict[str, Dict[str, Any]] = {}
    for r in results:
        fn = str(r.get("function", "unknown"))
        per_function.setdefault(fn, {"n": 0, "acc": [], "conf": []})
        per_function[fn]["n"] += 1
        per_function[fn]["acc"].append(float(bool(r.get("is_correct", False))))
        per_function[fn]["conf"].append(float(r.get("confidence") or 0.0))

    per_function_out: Dict[str, Any] = {}
    for fn, v in per_function.items():
        n = v["n"]
        per_function_out[fn] = {
            "n": n,
            "avg_accuracy": float(sum(v["acc"]) / n) if n else 0.0,
            "avg_confidence": float(sum(v["conf"]) / n) if n else 0.0,
        }

    return {
        "total_prompts": total,
        "accuracy": acc,
        "mean_confidence": float(sum(confidences) / total),
        "mean_entropy": float(sum(entropies) / total),
        "mean_expected_logprob": exp_logprob_mean,
        "correct_mean_confidence": float(sum(correct_confidences) / len(correct_confidences)) if correct_confidences else 0.0,
        "incorrect_mean_confidence": float(sum(incorrect_confidences) / len(incorrect_confidences)) if incorrect_confidences else 0.0,
        "per_function": per_function_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="HF model name/path or local checkpoint dir")
    parser.add_argument("--query-path", required=True, help="Query JSONL with prompt/query + completion")
    parser.add_argument("--output-file", required=True, help="Where to save evaluation JSON")

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for next-token scoring")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional limit on number of queries")

    parser.add_argument(
        "--scoring",
        default="auto",
        choices=["auto", "next-token", "sequence"],
        help="Scoring strategy (auto picks next-token if all candidates are one-token)",
    )
    parser.add_argument(
        "--accuracy-mode",
        default="candidate",
        choices=["candidate", "vocab-first-token", "correct-incorrect"],
        help=(
            "How to decide correctness. "
            "'candidate' (default): correct iff expected completion string is top-1 among candidates. "
            "'vocab-first-token': compare vocab argmax next-token against the first token of the expected completion. "
            "'correct-incorrect': compare next-token/sequence likelihood of `completion` vs `incorrect` (binary choice). "
            "Requires `incorrect` field in each query JSONL entry."
        ),
    )
    parser.add_argument(
        "--candidate-file",
        default=None,
        help="Optional JSONL/JSON file providing candidate completions; if unset, uses unique query completions.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on number of candidates (truncates in stable order).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="If set, include per-query `topk` candidate list (for qualitative inspection).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Optional max sequence length for `sequence` scoring (truncate from left).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers when loading the model.",
    )
    args = parser.parse_args()

    queries = _load_jsonl(args.query_path)
    if not queries:
        raise SystemExit(f"No queries loaded from {args.query_path}")
    if args.max_prompts is not None:
        queries = queries[: int(args.max_prompts)]

    prompt_texts: List[str] = []
    prompt_ids_list: List[List[int]] = []
    expected_completion_strs: List[str] = []
    incorrect_completion_strs: List[str] = []
    func_list: List[str] = []
    uid_list: List[str] = []

    for i, doc in enumerate(queries):
        prompt = _get_prompt(doc)
        completion = _get_completion(doc)
        if not prompt or completion == "":
            continue
        incorrect = str(doc.get("incorrect", "") or "")
        incorrect = incorrect.strip()
        if args.accuracy_mode == "correct-incorrect" and incorrect == "":
            # Skip entries without an inferred incorrect completion.
            continue
        prompt_texts.append(prompt)
        expected_completion_strs.append(completion)
        incorrect_completion_strs.append(incorrect)
        func_list.append(str(doc.get("func", "unknown")) if doc.get("func") is not None else "unknown")
        uid_list.append(str(doc.get("uid", f"q_{i}")))
        # prompt_ids computed later after tokenizer load
        prompt_ids_list.append([])  # placeholder

    # Device / dtype selection
    if args.device == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = args.device

    device = torch.device(dev_str)
    torch_dtype = torch.float16
    if device.type == "cpu":
        torch_dtype = torch.float32

    # Load tokenizer/model
    tok_kwargs = {}
    if args.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        # Most OLMo setups don't have explicit pad tokens; use EOS for padding.
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": (torch_dtype if device.type != "cpu" else torch.float32)}
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    # Tokenize prompts now that tokenizer is loaded
    prompt_ids_list = []
    for prompt in prompt_texts:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids_list.append([int(x) for x in ids])

    # Depending on accuracy mode, either compare against candidates (string-based)
    # or compare against the vocab argmax next-token (token-based).
    candidates: List[Dict[str, Any]] = []
    scoring = args.scoring

    if args.accuracy_mode == "vocab-first-token":
        scoring = "next-token"  # correctness always computed at next-token step

        # Expected first completion token id per query.
        # Use -1 sentinel for completions that tokenize to 0 tokens.
        expected_first_token_ids_list: List[int] = []
        for completion in expected_completion_strs:
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not comp_ids:
                expected_first_token_ids_list.append(-1)
            else:
                expected_first_token_ids_list.append(int(comp_ids[0]))

        expected_first_token_ids = torch.tensor(expected_first_token_ids_list, dtype=torch.long)
        results = _eval_vocab_first_token_batched(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompt_texts,
            prompt_ids_list=prompt_ids_list,
            expected_completion_strs=expected_completion_strs,
            expected_first_token_ids=expected_first_token_ids,
            func_list=func_list,
            uid_list=uid_list,
            batch_size=max(1, int(args.batch_size)),
            topk=args.topk,
        )
    elif args.accuracy_mode == "correct-incorrect":
        # Compare likelihood of `completion` vs `incorrect` (binary decision).
        # Decide between one-token next-token scoring and full sequence scoring.
        correct_token_ids_list: List[List[int]] = []
        incorrect_token_ids_list: List[List[int]] = []
        all_one_token = True
        for exp_str, inc_str in zip(expected_completion_strs, incorrect_completion_strs):
            exp_ids = tokenizer(exp_str, add_special_tokens=False)["input_ids"]
            inc_ids = tokenizer(inc_str, add_special_tokens=False)["input_ids"]
            exp_ids = [int(x) for x in exp_ids]
            inc_ids = [int(x) for x in inc_ids]
            correct_token_ids_list.append(exp_ids)
            incorrect_token_ids_list.append(inc_ids)
            if len(exp_ids) != 1 or len(inc_ids) != 1:
                all_one_token = False

        if all_one_token:
            scoring = "next-token"
            expected_first_token_ids = torch.tensor([ids[0] for ids in correct_token_ids_list], dtype=torch.long)
            incorrect_first_token_ids = torch.tensor([ids[0] for ids in incorrect_token_ids_list], dtype=torch.long)
            results = _eval_correct_vs_incorrect_one_token_batched(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompt_texts,
                prompt_ids_list=prompt_ids_list,
                expected_completion_strs=expected_completion_strs,
                incorrect_completion_strs=incorrect_completion_strs,
                expected_first_token_ids=expected_first_token_ids,
                incorrect_first_token_ids=incorrect_first_token_ids,
                func_list=func_list,
                uid_list=uid_list,
                batch_size=max(1, int(args.batch_size)),
            )
        else:
            scoring = "sequence"
            results = _eval_correct_vs_incorrect_sequence_per_query(
                model=model,
                device=device,
                prompts=prompt_texts,
                prompt_ids_list=prompt_ids_list,
                expected_completion_strs=expected_completion_strs,
                incorrect_completion_strs=incorrect_completion_strs,
                correct_token_ids_list=correct_token_ids_list,
                incorrect_token_ids_list=incorrect_token_ids_list,
                func_list=func_list,
                uid_list=uid_list,
                max_seq_len=args.max_seq_len,
            )
    else:
        # Candidates: from candidate-file or from unique query completions.
        candidate_strings: List[str] = []
        if args.candidate_file:
            cand_path = args.candidate_file
            if not os.path.exists(cand_path):
                raise SystemExit(f"--candidate-file not found: {cand_path}")
            if cand_path.endswith(".jsonl"):
                cand_docs = _load_jsonl(cand_path)
                # Each line can be either {completion: "..."} or {value: "..."} or a raw string dict.
                candidate_strings = []
                for d in cand_docs:
                    if isinstance(d, dict):
                        if "completion" in d:
                            candidate_strings.append(str(d["completion"]))
                        elif "value" in d:
                            candidate_strings.append(str(d["value"]))
                        else:
                            # Try a stable "first string-ish field"
                            for _, v in d.items():
                                if isinstance(v, (str, int, float)):
                                    candidate_strings.append(str(v))
                                    break
                    else:
                        candidate_strings.append(str(d))
            else:
                with open(cand_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    candidate_strings = [str(x) for x in loaded]
                elif isinstance(loaded, dict):
                    if "candidates" in loaded:
                        candidate_strings = [str(x) for x in loaded["candidates"]]
                    elif "completion" in loaded:
                        candidate_strings = [str(loaded["completion"])]
                    else:
                        raise SystemExit(f"Unrecognized candidate-file JSON schema: {cand_path}")
        else:
            candidate_strings = _stable_unique(expected_completion_strs)

        if args.max_candidates is not None:
            candidate_strings = candidate_strings[: int(args.max_candidates)]

        # Encode candidates
        for s in candidate_strings:
            token_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            token_ids = [int(x) for x in token_ids]
            candidates.append({"completion": s, "token_ids": token_ids})

        # Choose scoring strategy
        if scoring == "auto":
            all_one_token = all(len(c["token_ids"]) == 1 for c in candidates)
            scoring = "next-token" if all_one_token else "sequence"

        if scoring == "next-token":
            if not all(len(c["token_ids"]) == 1 for c in candidates):
                raise SystemExit(
                    "next-token scoring requested, but at least one candidate tokenizes to != 1 token. "
                    "Use --scoring sequence or adjust candidates."
                )
            results = _eval_next_token_batched(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompt_texts,
                prompt_ids_list=prompt_ids_list,
                candidates=candidates,
                expected_completion_strs=expected_completion_strs,
                func_list=func_list,
                uid_list=uid_list,
                batch_size=max(1, int(args.batch_size)),
                topk=args.topk,
            )
        else:
            results = _eval_sequence_per_query(
                model=model,
                device=device,
                tokenizer=tokenizer,
                prompts=prompt_texts,
                prompt_ids_list=prompt_ids_list,
                candidates=candidates,
                expected_completion_strs=expected_completion_strs,
                func_list=func_list,
                uid_list=uid_list,
                topk=args.topk,
                max_seq_len=args.max_seq_len,
            )

    analysis = _analyze_results(results)
    if args.accuracy_mode == "correct-incorrect" and not results:
        print(
            "Warning: no queries evaluated in correct-incorrect mode. "
            "Make sure your query JSONL contains a non-empty `incorrect` field "
            "(regenerate with filter/verification/data_converter.py if needed)."
        )

    output_data = {
        "evaluation_type": "model_prompt_completion_eval",
        "model_path": args.model_path,
        "query_path": args.query_path,
        "scoring": scoring,
        "accuracy_mode": args.accuracy_mode,
        "analysis": analysis,
        "results": results,
    }
    if args.accuracy_mode == "candidate":
        output_data["candidate_count"] = len(candidates)
        output_data["candidates"] = [c["completion"] for c in candidates[:50]]  # preview

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Print a short summary to stdout.
    if analysis:
        print(f"Accuracy: {analysis.get('accuracy', 0.0) * 100:.2f}%")
        print(f"Mean confidence: {analysis.get('mean_confidence', 0.0):.3f}")
        print(f"Mean expected logprob: {analysis.get('mean_expected_logprob', 0.0):.3f}")
        print(f"Mean entropy: {analysis.get('mean_entropy', 0.0):.3f}")
        per_fn = analysis.get("per_function", {}) or {}
        if per_fn:
            best_fn = max(per_fn.items(), key=lambda kv: kv[1].get("avg_accuracy", 0.0))
            print(f"Best func by avg_accuracy: {best_fn[0]} ({best_fn[1].get('avg_accuracy', 0.0):.3f})")


if __name__ == "__main__":
    main()

