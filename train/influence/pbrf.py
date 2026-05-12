#!/usr/bin/env python3
"""
Proximal Bregman Response Function (PBRF) training script.

For each target data point z_m, optimises the Proximal Bregman Objective (PBO)
starting from a fine-tuned model θˢ, and saves the resulting model θ*(ε) to
{output_dir}/{uid}/.

The PBO is:
    θ*(ε) = argmin_θ  (1/N) Σᵢ KL(p_θˢ(xᵢ) ‖ p_θ(xᵢ))   [Bregman / KL]
                    + ε · L(z_m, θ)                            [perturbation]
                    + (λ/2) ‖θ − θˢ‖²                          [proximity]

Usage:
    # Single GPU — train PBRF models for every data point
    python pbrf.py --model-path ./finetuned --dataset-path data.jsonl --output-dir ./pbrf_out

    # Multi-GPU — split target indices across GPUs
    python pbrf.py --model-path ./finetuned --dataset-path data.jsonl --output-dir ./pbrf_out --gpus 0,1,2,3
"""

import os
import sys
import json
import argparse
import copy
import gc
import glob
import subprocess
import math
import time
import threading
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        if "token_type_ids" in encoding:
            del encoding["token_type_ids"]
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def load_dataset_records(dataset_path, hop_depth_filter=None):
    """Return list of dicts with at least ``uid`` and ``text``."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    if hop_depth_filter is not None and data.get("hop_depth", 0) != hop_depth_filter:
                        continue
                    records.append({
                        **data,
                        "uid": data.get("uid", str(i)),
                        "text": data.get("text", ""),
                    })
                except json.JSONDecodeError:
                    records.append({"uid": str(i), "text": line.strip()})
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    records.append({"uid": str(i), "text": line.strip()})
    return records


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_path, device=None):
    """Load a causal LM and its tokenizer from *model_path*."""
    logger.info(f"Loading model and tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    if device is not None:
        model = model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# PBO optimisation
# ---------------------------------------------------------------------------

def train_single_pbrf(args, record_idx, all_records,
                      frozen_model=None, tokenizer=None, return_model=False):
    """Optimise the PBO for the target at *record_idx*.

    When *return_model* is False (default), saves the model to disk and returns
    None.  When True, returns ``(model, cfg_dict)`` without writing to disk —
    used by the rolling-evaluation mode so the caller can score queries while
    the model is still in GPU memory.

    If *frozen_model* and *tokenizer* are provided they are reused (the working
    model is created via ``deepcopy``); otherwise both are loaded from disk.
    """
    uid = all_records[record_idx]["uid"]

    if not return_model:
        output_dir = os.path.join(args.output_dir, uid)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, "config.json")):
            logger.info(f"[PBRF {uid}] Already exists — skipping.")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(all_records)
    epsilon = args.epsilon_pbrf if args.epsilon_pbrf is not None else 1.0 / N

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(
        f"[PBRF {uid}] target idx={record_idx}  ε={epsilon:.6g}  "
        f"λ={args.damping_lambda:.6g}  N={N}"
    )

    # ---- Frozen model θˢ ------------------------------------------------
    own_frozen = frozen_model is None
    if own_frozen:
        frozen_model, tokenizer = load_model(args.model_path, device)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False

    # ---- Working model θ  (initialised from θˢ) -------------------------
    if own_frozen:
        model, _ = load_model(args.model_path, device)
    else:
        model = copy.deepcopy(frozen_model)
        for p in model.parameters():
            p.requires_grad = True
    model.train()
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- Adam optimiser --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # ---- Warm-start: load curvature estimates from training ---------------
    if args.optimizer_state_path:
        logger.info(f"[PBRF {uid}] Loading optimizer curvature from {args.optimizer_state_path}")
        saved = torch.load(args.optimizer_state_path, map_location=device, weights_only=False)
        saved_states = list(saved["state"].values())
        loaded, skipped = 0, 0
        for i, param in enumerate(model.parameters()):
            if i < len(saved_states):
                s = saved_states[i]
                if s["exp_avg_sq"].shape == param.data.shape:
                    optimizer.state[param] = {
                        "step": torch.tensor(0.0, dtype=torch.float32),
                        "exp_avg": torch.zeros_like(param.data),
                        "exp_avg_sq": s["exp_avg_sq"].to(
                            dtype=param.data.dtype, device=device
                        ),
                    }
                    loaded += 1
                else:
                    skipped += 1
        n_params = sum(1 for _ in model.parameters())
        del saved
        logger.info(
            f"[PBRF {uid}] Warm-started exp_avg_sq for {loaded}/{n_params} params "
            f"({skipped} skipped due to shape mismatch), reset velocity & step"
        )

    # ---- Tokenise target example z_m (once) ------------------------------
    target_enc = tokenizer(
        all_records[record_idx]["text"],
        truncation=True,
        padding=False,
        max_length=args.max_length,
        return_tensors="pt",
    )
    target_ids = target_enc["input_ids"].to(device)
    target_mask = target_enc["attention_mask"].to(device)

    # ---- DataLoader for Bregman mini-batches -----------------------------
    all_texts = [r["text"] for r in all_records]
    bregman_ds = TextDataset(all_texts, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )
    effective_bs = min(args.batch_size, N)
    bregman_loader = DataLoader(
        bregman_ds,
        batch_size=effective_bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    bregman_iter = iter(bregman_loader)

    # ---- Training loop ---------------------------------------------------
    loss_history = []
    steps_taken = 0
    converged = False
    final_pbo_loss = None

    pbar = tqdm(
        range(1, args.max_steps + 1),
        desc=f"[PBRF {uid}]",
        unit="step",
        dynamic_ncols=True,
    )
    grad_accum = args.gradient_accumulation_steps
    for step in pbar:
        optimizer.zero_grad()
        kl_val = 0.0

        # ---- Bregman term: KL(p_θˢ ‖ p_θ) via gradient accumulation -----
        for _accum in range(grad_accum):
            try:
                batch = next(bregman_iter)
            except StopIteration:
                bregman_iter = iter(bregman_loader)
                batch = next(bregman_iter)

            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                f_logits = frozen_model(input_ids=b_ids, attention_mask=b_mask).logits
            w_logits = model(input_ids=b_ids, attention_mask=b_mask).logits

            f_lp = F.log_softmax(f_logits.float(), dim=-1)
            w_lp = F.log_softmax(w_logits.float(), dim=-1)
            kl = (f_lp.exp() * (f_lp - w_lp)).sum(dim=-1)
            kl_loss = (kl * b_mask).sum() / b_mask.sum().clamp(min=1)
            (kl_loss / grad_accum).backward()
            kl_val += kl_loss.detach().item() / grad_accum
            del f_logits, w_logits, f_lp, w_lp, kl, kl_loss

        # ---- Perturbation term: ε · L(z_m, θ) ---------------------------
        tgt_out = model(
            input_ids=target_ids,
            attention_mask=target_mask,
            labels=target_ids,
        )
        perturb_loss = epsilon * tgt_out.loss.float()
        perturb_val = perturb_loss.detach().item()
        perturb_loss.backward()

        # ---- Proximity gradient + value ----------------------------------
        # ∇_θ (λ/2 ‖θ−θˢ‖²) = λ(θ−θˢ)  — added directly to .grad
        prox_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
        with torch.no_grad():
            for pw, pf in zip(model.parameters(), frozen_model.parameters()):
                diff = pw.data - pf.data
                if pw.grad is not None:
                    pw.grad.add_(args.damping_lambda * diff)
                else:
                    pw.grad = (args.damping_lambda * diff).clone()
                prox_acc += (diff * diff).sum().float()
        prox_val = (args.damping_lambda / 2.0) * prox_acc.item()

        # ---- Clip + step -------------------------------------------------
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # ---- L2 projection: keep θ within a ball around θˢ ---------------
        if args.max_param_dist is not None:
            param_norm = prox_acc.sqrt().item()
            if param_norm > args.max_param_dist:
                scale = args.max_param_dist / param_norm
                with torch.no_grad():
                    for pw, pf in zip(model.parameters(), frozen_model.parameters()):
                        pw.data = pf.data + (pw.data - pf.data) * scale
                prox_acc = torch.tensor(
                    args.max_param_dist ** 2, device=device, dtype=torch.float32
                )
                prox_val = (args.damping_lambda / 2.0) * prox_acc.item()

        pbo_val = kl_val + perturb_val + prox_val
        final_pbo_loss = pbo_val
        steps_taken = step

        # ---- Progress bar ------------------------------------------------
        pbar.set_postfix(
            pbo=f"{pbo_val:.4e}",
            kl=f"{kl_val:.4e}",
            perturb=f"{perturb_val:.4e}",
            prox=f"{prox_val:.4e}",
        )

        # ---- Logging -----------------------------------------------------
        if step % args.log_interval == 0 or step == 1:
            logger.info(
                f"[PBRF {uid}] step {step:>5d}/{args.max_steps}  "
                f"pbo={pbo_val:.6f}  kl={kl_val:.6f}  "
                f"perturb={perturb_val:.6f}  prox={prox_val:.6f}"
            )

        # ---- KL ceiling: abort if output distribution has diverged too far --
        if step >= args.min_steps and args.max_kl is not None and kl_val > args.max_kl:
            logger.warning(
                f"[PBRF {uid}] KL ceiling hit at step {step} "
                f"(kl={kl_val:.4e} > max_kl={args.max_kl:.1e}). Stopping."
            )
            converged = False
            break

        # ---- Early stopping: compare avg PBO of two consecutive windows ----
        loss_history.append(pbo_val)
        window = args.convergence_window
        if step >= args.min_steps and len(loss_history) >= 2 * window:
            recent_avg = sum(loss_history[-window:]) / window
            prev_avg = sum(loss_history[-2 * window:-window]) / window
            denom = abs(prev_avg) if abs(prev_avg) > 1e-12 else 1e-12
            rel_change = abs(recent_avg - prev_avg) / denom
            if rel_change < args.convergence_tol:
                logger.info(
                    f"[PBRF {uid}] Converged at step {step} "
                    f"(avg PBO last {window}={recent_avg:.6f} vs prev {window}={prev_avg:.6f}, "
                    f"rel change={rel_change:.2e} < {args.convergence_tol:.1e})"
                )
                converged = True
                break

    pbar.close()

    # ---- Build config dict (used by both save and return paths) ----------
    cfg = {
        "type": "pbrf",
        "target_uid": uid,
        "target_index": record_idx,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "epsilon_pbrf": epsilon,
        "damping_lambda": args.damping_lambda,
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "min_steps": args.min_steps,
        "convergence_tol": args.convergence_tol,
        "convergence_window": args.convergence_window,
        "max_grad_norm": args.max_grad_norm,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "seed": args.seed,
        "hop_depth": args.hop_depth,
        "total_train_samples": N,
        "steps_taken": steps_taken,
        "converged": converged,
        "final_pbo_loss": final_pbo_loss,
    }

    if return_model:
        return model, cfg

    # ---- Save optimised model --------------------------------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "pbrf_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"[PBRF {uid}] Saved to {output_dir}  steps={steps_taken}  converged={converged}")

    del model, optimizer
    if own_frozen:
        del frozen_model
    gc.collect()
    torch.cuda.empty_cache()
    return None


# ---------------------------------------------------------------------------
# Orchestration  (mirrors loo.py)
# ---------------------------------------------------------------------------

def _count_completed(output_dir):
    """Count model directories that contain a saved config.json (HF model marker)."""
    count = 0
    try:
        for name in os.listdir(output_dir):
            d = os.path.join(output_dir, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                count += 1
    except OSError:
        pass
    return count


def _progress_monitor(output_dir, total, stop_event):
    """Background thread that prints a progress line every 30 s."""
    start = time.time()
    while not stop_event.is_set():
        completed = _count_completed(output_dir)
        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else float("inf")
        eta_str = f"{eta / 3600:.1f}h" if eta != float("inf") else "?"
        print(
            f"[Progress] {completed}/{total} models complete  "
            f"({elapsed / 3600:.1f}h elapsed, ETA ~{eta_str})",
            flush=True,
        )
        stop_event.wait(timeout=30)
    completed = _count_completed(output_dir)
    print(f"[Progress] Done — {completed}/{total} models complete.", flush=True)


def spawn_worker(gpu_id, pbrf_indices, argv, log_path):
    """Spawn a worker subprocess pinned to *gpu_id*; output goes to *log_path*."""
    clean_argv = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--gpus", "--pbrf-indices", "--target-uids"):
            skip_next = True
            continue
        if tok.startswith("--gpus=") or tok.startswith("--pbrf-indices=") or tok.startswith("--target-uids="):
            continue
        clean_argv.append(tok)

    indices_str = ",".join(str(i) for i in pbrf_indices)
    script_path = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, script_path] + clean_argv + ["--pbrf-indices", indices_str]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["TQDM_DISABLE"] = "1"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    logger.info(
        f"Spawning worker on GPU {gpu_id} — indices {pbrf_indices[0]}–{pbrf_indices[-1]}, "
        f"log: {log_path}"
    )
    return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file


# ---------------------------------------------------------------------------
# Rolling evaluation  (train → score → delete, one target at a time)
# ---------------------------------------------------------------------------

def _import_ranker_modules():
    """Lazy-import scoring and evaluation helpers from filter/."""
    filter_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "filter")
    )
    if filter_dir not in sys.path:
        sys.path.insert(0, filter_dir)

    import utils as filter_utils
    from loo_ranker import compute_query_losses
    from kronfluence_ranker import (
        HopsQueryDataset,
        aggregate_scores_to_training_meta,
        save_influence_scores,
        _compute_recall_precision_at_k,
        _compute_composition_per_function,
        _parse_eval_topk_list,
        DISTRACTOR_FUNCS,
        allowed_role_for_token,
        paired_function_token,
    )
    return dict(
        filter_utils=filter_utils,
        compute_query_losses=compute_query_losses,
        HopsQueryDataset=HopsQueryDataset,
        aggregate_scores_to_training_meta=aggregate_scores_to_training_meta,
        save_influence_scores=save_influence_scores,
        _compute_recall_precision_at_k=_compute_recall_precision_at_k,
        _compute_composition_per_function=_compute_composition_per_function,
        _parse_eval_topk_list=_parse_eval_topk_list,
        DISTRACTOR_FUNCS=DISTRACTOR_FUNCS,
        allowed_role_for_token=allowed_role_for_token,
        paired_function_token=paired_function_token,
    )


def rolling_worker(args, target_indices, all_records):
    """Train-and-score each target sequentially, saving partial scores.

    For each target data point:
      1. Deep-copy the frozen model → working model
      2. Run PBO optimisation (same loop as train_single_pbrf)
      3. Compute query losses with the PBRF model
      4. Record influence = L(θ_pbrf, q) − L(θˢ, q)
      5. Delete the working model from memory

    Writes a partial-results JSONL to ``{output_dir}/rolling_partial_{gpu}.jsonl``.
    """
    mods = _import_ranker_modules()
    filter_utils = mods["filter_utils"]
    compute_query_losses = mods["compute_query_losses"]
    HopsQueryDataset = mods["HopsQueryDataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # ---- Frozen model (loaded once, stays in memory) ---------------------
    frozen_model, tokenizer = load_model(args.model_path, device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    # ---- Query dataset ---------------------------------------------------
    query_docs = filter_utils.load_jsonl_dataset(args.query_path)
    use_margin = bool(getattr(args, "use_margin_loss", False))
    use_full_text = bool(
        getattr(args, "query_full_text_loss", False) and not use_margin
    )
    query_dataset = HopsQueryDataset(
        query_docs,
        tokenizer,
        max_length=getattr(args, "max_query_length", 128),
        restrict_answers=use_margin,
        min_ans=getattr(args, "min_answer", 1),
        max_ans=getattr(args, "max_answer", 100),
        full_text_loss=use_full_text,
        response_only_query_loss=bool(
            getattr(args, "response_only_query_loss", False)
        ),
    )
    candidate_ids = getattr(query_dataset, "candidate_ids", None)
    num_queries = len(query_dataset)
    query_batch = getattr(args, "per_device_query_batch", 4)

    logger.info(
        f"Rolling worker [{gpu_label}]: {len(target_indices)} targets, "
        f"{num_queries} queries"
    )

    # ---- Base-model losses (computed once) --------------------------------
    base_losses = compute_query_losses(
        frozen_model,
        query_dataset,
        device,
        batch_size=query_batch,
        use_margin_loss=use_margin,
        full_text_loss=use_full_text,
        candidate_ids=candidate_ids,
        desc="Queries [base]",
    )
    logger.info(
        f"Base losses: mean={base_losses.mean():.4f}  "
        f"std={base_losses.std():.4f}"
    )

    # ---- Rolling train → score → delete ----------------------------------
    partial_path = os.path.join(
        args.output_dir, f"rolling_partial_{gpu_label}.jsonl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support: find train indices already written to the partial file.
    completed_set: set = set()
    if os.path.exists(partial_path):
        with open(partial_path) as _rf:
            for _line in _rf:
                if _line.strip():
                    try:
                        completed_set.add(json.loads(_line)["train_idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if completed_set:
        logger.info(
            f"Resuming [{gpu_label}]: {len(completed_set)} targets already scored, "
            f"{len(target_indices) - len(completed_set)} remaining."
        )

    remaining_indices = [i for i in target_indices if i not in completed_set]
    completed = len(completed_set)
    partial_fh = open(partial_path, "a")

    outer_pbar = tqdm(
        remaining_indices,
        desc=f"Rolling [{gpu_label}]",
        unit="doc",
        dynamic_ncols=True,
    )
    for idx in outer_pbar:
        uid = all_records[idx]["uid"]
        outer_pbar.set_postfix(uid=uid)

        result = train_single_pbrf(
            args,
            idx,
            all_records,
            frozen_model=frozen_model,
            tokenizer=tokenizer,
            return_model=True,
        )
        if result is None:
            continue
        model, cfg = result

        # Score queries with the PBRF model (inference mode)
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        pbrf_losses = compute_query_losses(
            model,
            query_dataset,
            device,
            batch_size=query_batch,
            use_margin_loss=use_margin,
            full_text_loss=use_full_text,
            candidate_ids=candidate_ids,
            desc=f"  Queries [{uid}]",
        )
        scores = (pbrf_losses - base_losses).tolist()

        row = {
            "train_idx": cfg["target_index"],
            "uid": cfg["target_uid"],
            "scores": scores,
            "steps_taken": cfg["steps_taken"],
            "converged": cfg["converged"],
            "final_pbo_loss": cfg["final_pbo_loss"],
        }
        partial_fh.write(json.dumps(row) + "\n")
        partial_fh.flush()
        completed += 1

        del model
        gc.collect()
        torch.cuda.empty_cache()

    outer_pbar.close()
    partial_fh.close()
    logger.info(
        f"Rolling worker [{gpu_label}] done. "
        f"Saved {completed} results → {partial_path}"
    )

    del frozen_model
    gc.collect()
    torch.cuda.empty_cache()


def merge_and_evaluate_rolling(args, all_records):
    """Merge partial rolling-score files and run the full evaluation pipeline.

    Produces the same output files as pbrf_ranker.sh / loo_ranker.py:
      - ranked JSONL (aggregated scores)
      - per-query JSONL
      - summary JSONL, metrics JSON, examples JSONL  (if eval flags set)
    """
    mods = _import_ranker_modules()
    filter_utils = mods["filter_utils"]
    HopsQueryDataset = mods["HopsQueryDataset"]
    aggregate_fn = mods["aggregate_scores_to_training_meta"]
    save_fn = mods["save_influence_scores"]
    recall_fn = mods["_compute_recall_precision_at_k"]
    comp_fn = mods["_compute_composition_per_function"]
    parse_k_fn = mods["_parse_eval_topk_list"]
    DISTRACTOR_FUNCS = mods["DISTRACTOR_FUNCS"]
    allowed_role_for_token = mods["allowed_role_for_token"]

    # ---- Load partial results --------------------------------------------
    pattern = os.path.join(args.output_dir, "rolling_partial_*.jsonl")
    partial_files = sorted(glob.glob(pattern))
    if not partial_files:
        logger.error("No rolling partial files found — nothing to merge.")
        return

    all_results = []
    for pf in partial_files:
        with open(pf) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))
    logger.info(
        f"Merging {len(all_results)} partial results from "
        f"{len(partial_files)} file(s)."
    )

    if not all_results:
        logger.error("Partial files are empty — nothing to merge.")
        return

    num_queries = len(all_results[0]["scores"])
    num_train = len(all_records)
    score_matrix = torch.zeros(num_queries, num_train, dtype=torch.float32)
    for r in all_results:
        ti = r["train_idx"]
        score_matrix[:, ti] = torch.tensor(r["scores"], dtype=torch.float32)

    # ---- Query metadata (needed for aggregation & evaluation) ------------
    query_docs = filter_utils.load_jsonl_dataset(args.query_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    use_margin = bool(getattr(args, "use_margin_loss", False))
    use_full_text = bool(
        getattr(args, "query_full_text_loss", False) and not use_margin
    )
    query_dataset = HopsQueryDataset(
        query_docs,
        tok,
        max_length=getattr(args, "max_query_length", 128),
        restrict_answers=use_margin,
        min_ans=getattr(args, "min_answer", 1),
        max_ans=getattr(args, "max_answer", 100),
        full_text_loss=use_full_text,
        response_only_query_loss=bool(
            getattr(args, "response_only_query_loss", False)
        ),
    )

    # ---- Save aggregated ranked output -----------------------------------
    output_path = (
        getattr(args, "scores_output_path", None)
        or os.path.join(args.output_dir, "rolling_ranked.jsonl")
    )
    training_meta = aggregate_fn(score_matrix, query_dataset.meta, all_records)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_fn(training_meta, output_path)
    logger.info(f"Saved ranked output → {output_path}")

    # ---- Per-query scores JSONL ------------------------------------------
    per_query_path = getattr(args, "per_query_output_path", None)
    if per_query_path:
        train_uids = [str(d.get("uid", i)) for i, d in enumerate(all_records)]
        os.makedirs(
            os.path.dirname(os.path.abspath(per_query_path)), exist_ok=True
        )
        with open(per_query_path, "w") as fh:
            for qi, qm in enumerate(query_dataset.meta):
                fh.write(
                    json.dumps({
                        "query_uid": qm.get("uid"),
                        "prompt": qm.get("prompt"),
                        "completion": qm.get("completion"),
                        "func": qm.get("func"),
                        "correct": qm.get("correct"),
                        "train_uids": train_uids,
                        "scores": score_matrix[qi].tolist(),
                    })
                    + "\n"
                )
        logger.info(f"Saved per-query scores → {per_query_path}")

    # ---- Save per-target training metadata -------------------------------
    rolling_meta_path = os.path.join(args.output_dir, "rolling_train_meta.jsonl")
    with open(rolling_meta_path, "w") as fh:
        for r in all_results:
            fh.write(
                json.dumps({
                    k: v for k, v in r.items() if k != "scores"
                })
                + "\n"
            )
    logger.info(f"Saved training metadata → {rolling_meta_path}")

    # ---- Evaluation (recall / precision @ k) -----------------------------
    eval_topk_range = getattr(args, "eval_topk_range", None)
    eval_k_list = (
        parse_k_fn(None, None, eval_topk_range) if eval_topk_range else []
    )
    if not eval_k_list:
        _cleanup_partials(partial_files)
        return

    def _is_relevant(doc, func):
        doc_func = str(doc.get("func", ""))
        if doc_func != func:
            return False
        role = str(doc.get("role", "")).lower()
        if not role:
            return True
        expected = allowed_role_for_token(func)
        return (expected is not None) and (role == expected)

    func_to_relevant: dict = {}
    for ti, doc in enumerate(all_records):
        f = str(doc.get("func", ""))
        if f in DISTRACTOR_FUNCS:
            func_to_relevant.setdefault(f, []).append(ti)
        elif _is_relevant(doc, f):
            func_to_relevant.setdefault(f, []).append(ti)

    func_to_queries: dict = {}
    for qi, qm in enumerate(query_dataset.meta):
        if not bool(qm.get("correct", False)):
            continue
        f = str(qm.get("func", ""))
        func_to_queries.setdefault(f, []).append(qi)

    metrics: dict = {"recall_at_k": {}, "precision_at_k": {}, "composition_at_k": {}}

    for k in eval_k_list:
        (
            per_func_recalls,
            per_func_precisions,
            per_func_counts,
            per_func_recall_vars,
            per_func_precision_vars,
        ) = recall_fn(
            score_matrix=score_matrix,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_queries,
            k=k,
        )
        if per_func_recalls:
            overall = float(
                sum(per_func_recalls.values()) / len(per_func_recalls)
            )
            _nq = sum(per_func_counts.values())
            pq_avg = (
                sum(
                    per_func_recalls[f] * per_func_counts[f]
                    for f in per_func_recalls
                )
                / _nq
                if _nq > 0
                else 0.0
            )
            metrics["recall_at_k"][str(k)] = {
                "k": k,
                "per_function": per_func_recalls,
                "per_function_variance": per_func_recall_vars,
                "overall_average": overall,
                "per_query_average": pq_avg,
            }
            if k <= 5:
                print(f"Recall@{k}: {overall:.4f}  (per-query avg {pq_avg:.4f})")

        if per_func_precisions:
            overall_p = float(
                sum(per_func_precisions.values()) / len(per_func_precisions)
            )
            metrics["precision_at_k"][str(k)] = {
                "k": k,
                "per_function": per_func_precisions,
                "per_function_variance": per_func_precision_vars,
                "overall_average": overall_p,
            }

        composition = comp_fn(
            score_matrix=score_matrix,
            train_docs=all_records,
            func_to_relevant_indices=func_to_relevant,
            func_to_query_indices=func_to_queries,
            k=k,
        )
        if composition:
            overall_comp = {}
            for cat in ("relevant", "distractor", "other"):
                vals = [v[cat] for v in composition.values()]
                if vals:
                    overall_comp[cat] = float(sum(vals) / len(vals))
            metrics["composition_at_k"][str(k)] = {
                "k": k,
                "per_function": composition,
                "overall_average": overall_comp,
            }

    # ---- Save metrics / summary ------------------------------------------
    eval_metrics_path = getattr(args, "eval_metrics_path", None)
    if eval_metrics_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(eval_metrics_path)), exist_ok=True
        )
        with open(eval_metrics_path, "w") as fh:
            json.dump(metrics, fh)
        logger.info(f"Saved eval metrics → {eval_metrics_path}")

    eval_summary_path = getattr(args, "eval_summary_jsonl", None)
    if eval_summary_path and eval_k_list:
        os.makedirs(
            os.path.dirname(os.path.abspath(eval_summary_path)), exist_ok=True
        )
        with open(eval_summary_path, "w") as fh:
            for k in eval_k_list:
                sk = str(k)
                row: dict = {"k": k}
                if sk in metrics.get("recall_at_k", {}):
                    r = metrics["recall_at_k"][sk]
                    row["recall_overall_avg"] = r.get("overall_average")
                    row["recall_per_query_avg"] = r.get("per_query_average")
                    vars_r = r.get("per_function_variance", {})
                    if vars_r:
                        row["recall_var_avg"] = float(
                            sum(vars_r.values()) / len(vars_r)
                        )
                if sk in metrics.get("precision_at_k", {}):
                    p = metrics["precision_at_k"][sk]
                    row["precision_overall_avg"] = p.get("overall_average")
                if sk in metrics.get("composition_at_k", {}):
                    comp = metrics["composition_at_k"][sk].get(
                        "overall_average", {}
                    )
                    if isinstance(comp, dict):
                        row["composition_relevant"] = comp.get("relevant")
                        row["composition_distractor"] = comp.get("distractor")
                        row["composition_other"] = comp.get("other")
                fh.write(json.dumps(row) + "\n")
        logger.info(f"Saved eval summary → {eval_summary_path}")

    # ---- Save run config -------------------------------------------------
    config_path = getattr(args, "config_output_path", None)
    if config_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(config_path)), exist_ok=True
        )
        with open(config_path, "w") as fh:
            json.dump(
                {
                    "mode": "rolling",
                    "model_path": args.model_path,
                    "dataset_path": args.dataset_path,
                    "query_path": args.query_path,
                    "learning_rate": args.learning_rate,
                    "max_steps": args.max_steps,
                    "epsilon_pbrf": args.epsilon_pbrf,
                    "damping_lambda": args.damping_lambda,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "use_margin_loss": use_margin,
                    "min_answer": getattr(args, "min_answer", 1),
                    "max_answer": getattr(args, "max_answer", 100),
                    "num_targets": len(all_results),
                    "num_queries": num_queries,
                    "output_path": output_path,
                },
                fh,
                indent=2,
            )
        logger.info(f"Saved run config → {config_path}")

    _cleanup_partials(partial_files)
    logger.info("Rolling evaluation complete.")


def _cleanup_partials(partial_files):
    for pf in partial_files:
        try:
            os.remove(pf)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PBRF: optimise the Proximal Bregman Objective for each target data point "
                    "and save the resulting model.",
    )
    p.add_argument("--model-path", required=True,
                   help="Path to fine-tuned model θˢ")
    p.add_argument("--dataset-path", required=True,
                   help="Path to training dataset (.jsonl)")
    p.add_argument("--output-dir", required=True,
                   help="Root output dir; PBRF models → {output_dir}/{uid}/")

    # PBO hyper-parameters
    g = p.add_argument_group("PBO hyper-parameters")
    g.add_argument("--learning-rate", type=float, default=1e-4)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.999)
    g.add_argument("--adam-epsilon", type=float, default=1e-8)
    g.add_argument("--batch-size", type=int, default=4,
                   help="Examples per mini-batch for Bregman term estimation")
    g.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Number of mini-batches to accumulate before an optimizer step "
                        "(effective batch = batch_size × gradient_accumulation_steps)")
    g.add_argument("--max-steps", type=int, default=10000,
                   help="Maximum Adam steps for PBO optimisation")
    g.add_argument("--min-steps", type=int, default=100,
                   help="Minimum steps before convergence checks activate")
    g.add_argument("--convergence-tol", type=float, default=0.01,
                   help="Stop when relative change between two consecutive window averages < tol (fraction, e.g. 0.01 = 1%%)")
    g.add_argument("--convergence-window", type=int, default=100,
                   help="Window size (in steps) for convergence averaging")
    g.add_argument("--damping-lambda", type=float, default=1e-3,
                   help="Weight-space proximity coefficient λ")
    g.add_argument("--epsilon-pbrf", type=float, default=None,
                   help="Perturbation weight ε (default: 1/N)")
    g.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping norm (0 to disable)")
    g.add_argument("--optimizer-state-path", type=str, default=None,
                   help="Path to optimizer.pt from training — warm-starts Adam curvature (exp_avg_sq)")
    g.add_argument("--max-kl", type=float, default=None,
                   help="KL ceiling — stop if KL divergence exceeds this")
    g.add_argument("--max-param-dist", type=float, default=None,
                   help="L2 ball radius — project θ back if ‖θ−θˢ‖ exceeds this")
    g.add_argument("--log-interval", type=int, default=100,
                   help="Log PBO loss every N steps")

    # General
    g2 = p.add_argument_group("General")
    g2.add_argument("--max-length", type=int, default=2048,
                    help="Maximum sequence length for tokenisation")
    g2.add_argument("--seed", type=int, default=42)
    g2.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="Disable gradient checkpointing (faster, more memory)")
    g2.add_argument("--hop-depth", type=int, default=None,
                    help="Filter dataset to a specific hop depth")

    # Target selection
    g3 = p.add_argument_group("Target selection")
    g3.add_argument("--target-uids", type=str, default=None,
                    help="Comma-separated UIDs to process (default: all)")

    # Multi-GPU
    g4 = p.add_argument_group("Multi-GPU")
    g4.add_argument("--gpus", type=str, default=None,
                    help="Comma-separated GPU IDs for parallel execution, e.g. '0,1,2,3'")
    g4.add_argument("--pbrf-indices", type=str, default=None,
                    help="[Internal] Comma-separated dataset indices for this worker")

    # Rolling evaluation (train → score → delete in one pass)
    g5 = p.add_argument_group("Rolling evaluation")
    g5.add_argument("--query-path", type=str, default=None,
                    help="Query JSONL path. When set, activates rolling mode: "
                         "train each PBRF model, score queries, delete model, repeat.")
    g5.add_argument("--scores-output-path", type=str, default=None,
                    help="Output path for aggregated ranked JSONL "
                         "(default: {output_dir}/rolling_ranked.jsonl)")
    g5.add_argument("--per-query-output-path", type=str, default=None,
                    help="Per-query scores JSONL output path")
    g5.add_argument("--use-margin-loss", action="store_true",
                    help="Use restricted-answer margin loss over integer candidate set")
    g5.add_argument("--min-answer", type=int, default=1,
                    help="Min integer for restricted answer set")
    g5.add_argument("--max-answer", type=int, default=100,
                    help="Max integer for restricted answer set")
    g5.add_argument("--max-query-length", type=int, default=128,
                    help="Max tokenised length for queries")
    g5.add_argument("--per-device-query-batch", type=int, default=1,
                    help="Batch size for query forward passes")
    g5.add_argument("--query-full-text-loss", action="store_true",
                    help="Use full prompt+completion LM loss on queries")
    g5.add_argument("--response-only-query-loss", action="store_true",
                    help="Supervise only completion tokens on queries")
    g5.add_argument("--standardized", action="store_true",
                    help="Disable margin loss, use full-text LM loss on queries")
    g5.add_argument("--eval-topk-range", type=str, default=None, metavar="START,END",
                    help="Inclusive sweep of k values for recall/precision, e.g. '1,100'")
    g5.add_argument("--eval-metrics-path", type=str, default=None,
                    help="Save evaluation metrics JSON")
    g5.add_argument("--eval-summary-jsonl", type=str, default=None,
                    help="Save summary JSONL (one line per k)")
    g5.add_argument("--config-output-path", type=str, default=None,
                    help="Save rolling run configuration JSON")

    args = p.parse_args()

    # --standardized overrides
    if getattr(args, "standardized", False):
        args.use_margin_loss = False
        args.query_full_text_loss = True
    if getattr(args, "response_only_query_loss", False):
        args.query_full_text_loss = True
        args.use_margin_loss = False

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_records = load_dataset_records(args.dataset_path, hop_depth_filter=args.hop_depth)
    N = len(all_records)
    if N == 0:
        logger.error("No records found in dataset (check path and hop-depth filter).")
        sys.exit(1)
    logger.info(f"Dataset: {N} records from {args.dataset_path}")

    # Resolve which indices to process
    if args.target_uids is not None and args.pbrf_indices is None:
        uid_set = {u.strip() for u in args.target_uids.split(",") if u.strip()}
        target_indices = [i for i, r in enumerate(all_records) if r["uid"] in uid_set]
        missing = uid_set - {all_records[i]["uid"] for i in target_indices}
        if missing:
            logger.warning(f"UIDs not found in dataset: {missing}")
        if not target_indices:
            logger.error("None of the requested target UIDs were found.")
            sys.exit(1)
    elif args.pbrf_indices is not None:
        target_indices = [int(x.strip()) for x in args.pbrf_indices.split(",") if x.strip()]
    else:
        target_indices = list(range(N))

    rolling_mode = args.query_path is not None

    # -------------------------------------------------------------------
    # Rolling mode  (train → score → delete, no models saved)
    # -------------------------------------------------------------------
    if rolling_mode:
        if args.gpus is not None and args.pbrf_indices is None:
            # Orchestrator: spawn GPU workers, then merge & evaluate
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            if not gpus:
                logger.error("--gpus specified but no GPU IDs parsed.")
                sys.exit(1)

            logs_dir = os.path.join(args.output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            chunk_size = math.ceil(len(target_indices) / len(gpus))

            processes = []
            open_files = []
            for rank, gpu_id in enumerate(gpus):
                start = rank * chunk_size
                end = min(start + chunk_size, len(target_indices))
                if start >= len(target_indices):
                    break
                gpu_indices = target_indices[start:end]
                log_path = os.path.join(logs_dir, f"gpu_{gpu_id}.log")
                proc, lf = spawn_worker(
                    gpu_id, gpu_indices, sys.argv[1:], log_path
                )
                processes.append((gpu_id, proc))
                open_files.append(lf)

            exit_codes = []
            for gpu_id, proc in processes:
                rc = proc.wait()
                exit_codes.append(rc)
                status = (
                    "completed successfully"
                    if rc == 0
                    else f"FAILED (exit code {rc})"
                )
                logger.info(
                    f"Rolling worker GPU {gpu_id} {status}. "
                    f"Log: {logs_dir}/gpu_{gpu_id}.log"
                )
            for lf in open_files:
                lf.close()

            if any(rc != 0 for rc in exit_codes):
                logger.warning(
                    "Some rolling workers failed. "
                    "Merging available partial results."
                )

            merge_and_evaluate_rolling(args, all_records)

        elif args.pbrf_indices is not None:
            # GPU worker process: rolling train + score on assigned indices
            rolling_worker(args, target_indices, all_records)

        else:
            # Single GPU: run rolling worker then merge & evaluate
            rolling_worker(args, target_indices, all_records)
            merge_and_evaluate_rolling(args, all_records)

        return

    # -------------------------------------------------------------------
    # Standard mode  (train and save models to disk)
    # -------------------------------------------------------------------

    # Orchestrator mode  (--gpus given, not a worker)
    if args.gpus is not None and args.pbrf_indices is None:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            logger.error("--gpus specified but no GPU IDs parsed.")
            sys.exit(1)

        logs_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        chunk_size = math.ceil(len(target_indices) / len(gpus))
        total_models = len(target_indices)

        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=_progress_monitor,
            args=(args.output_dir, total_models, stop_monitor),
            daemon=True,
        )
        monitor_thread.start()

        processes = []
        open_files = []
        for rank, gpu_id in enumerate(gpus):
            start = rank * chunk_size
            end = min(start + chunk_size, len(target_indices))
            if start >= len(target_indices):
                break
            gpu_indices = target_indices[start:end]
            log_path = os.path.join(logs_dir, f"gpu_{gpu_id}.log")
            proc, lf = spawn_worker(gpu_id, gpu_indices, sys.argv[1:], log_path)
            processes.append((gpu_id, proc))
            open_files.append(lf)

        exit_codes = []
        for gpu_id, proc in processes:
            rc = proc.wait()
            exit_codes.append(rc)
            status = "completed successfully" if rc == 0 else f"FAILED (exit code {rc})"
            logger.info(f"Worker on GPU {gpu_id} {status}. Log: {logs_dir}/gpu_{gpu_id}.log")

        for lf in open_files:
            lf.close()

        stop_monitor.set()
        monitor_thread.join()

        if any(rc != 0 for rc in exit_codes):
            sys.exit(1)
        logger.info(f"All PBRF workers completed. Logs in {logs_dir}/")
        return

    # Worker / single-GPU mode
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    logger.info(
        f"Worker on GPU(s) [{gpu_label}]: {len(target_indices)} PBRF runs "
        f"(indices {target_indices[0]}–{target_indices[-1]})"
    )

    for idx in target_indices:
        train_single_pbrf(args, idx, all_records)

    logger.info(
        f"Worker done (GPU [{gpu_label}]). "
        f"Processed {len(target_indices)} PBRF runs."
    )


if __name__ == "__main__":
    main()
