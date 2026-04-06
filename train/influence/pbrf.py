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

def train_single_pbrf(args, record_idx, all_records):
    """Optimise the PBO for the target at *record_idx* and save θ*(ε)."""
    uid = all_records[record_idx]["uid"]
    output_dir = os.path.join(args.output_dir, uid)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info(f"[PBRF {uid}] Already exists — skipping.")
        return

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
    frozen_model, tokenizer = load_model(args.model_path, device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    # ---- Working model θ  (initialised from θˢ) -------------------------
    model, _ = load_model(args.model_path, device)
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
    convergence_window = 10
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
    for step in pbar:
        # Fetch Bregman mini-batch (cycle through dataset)
        try:
            batch = next(bregman_iter)
        except StopIteration:
            bregman_iter = iter(bregman_loader)
            batch = next(bregman_iter)

        b_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)

        # ---- Bregman term: KL(p_θˢ ‖ p_θ) averaged over tokens ----------
        with torch.no_grad():
            f_logits = frozen_model(input_ids=b_ids, attention_mask=b_mask).logits

        w_logits = model(input_ids=b_ids, attention_mask=b_mask).logits

        # Compute in fp32 for numerical stability
        f_lp = F.log_softmax(f_logits.float(), dim=-1)
        w_lp = F.log_softmax(w_logits.float(), dim=-1)
        kl = (f_lp.exp() * (f_lp - w_lp)).sum(dim=-1)          # (B, seq)
        kl_loss = (kl * b_mask).sum() / b_mask.sum().clamp(min=1)

        # ---- Perturbation term: ε · L(z_m, θ) ---------------------------
        tgt_out = model(
            input_ids=target_ids,
            attention_mask=target_mask,
            labels=target_ids,
        )
        perturb_loss = epsilon * tgt_out.loss.float()

        # ---- Backward on Bregman + perturbation --------------------------
        loss = kl_loss + perturb_loss
        optimizer.zero_grad()
        loss.backward()

        # ---- Proximity gradient + value ----------------------------------
        # ∇_θ (λ/2 ‖θ−θˢ‖²) = λ(θ−θˢ)  — added directly to .grad
        # The scalar value is accumulated for logging / convergence.
        prox_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
        with torch.no_grad():
            for pw, pf in zip(model.parameters(), frozen_model.parameters()):
                diff = pw.data - pf.data
                prox_grad = args.damping_lambda * diff
                if pw.grad is not None:
                    pw.grad.add_(prox_grad)
                else:
                    pw.grad = prox_grad.clone()
                prox_acc += (diff * diff).sum().float()
        prox_val = (args.damping_lambda / 2.0) * prox_acc.item()

        # ---- Clip + step -------------------------------------------------
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        pbo_val = loss.item() + prox_val
        final_pbo_loss = pbo_val
        steps_taken = step

        # ---- Progress bar ------------------------------------------------
        pbar.set_postfix(
            pbo=f"{pbo_val:.4e}",
            kl=f"{kl_loss.item():.4e}",
            perturb=f"{perturb_loss.item():.4e}",
            prox=f"{prox_val:.4e}",
        )

        # ---- Logging -----------------------------------------------------
        if step % args.log_interval == 0 or step == 1:
            logger.info(
                f"[PBRF {uid}] step {step:>5d}/{args.max_steps}  "
                f"pbo={pbo_val:.6f}  kl={kl_loss.item():.6f}  "
                f"perturb={perturb_loss.item():.6f}  prox={prox_val:.6f}"
            )

        # ---- Early stopping (only after min_steps) -------------------------
        if step >= args.min_steps:
            if abs(pbo_val) < args.loss_threshold:
                logger.info(
                    f"[PBRF {uid}] Converged at step {step} "
                    f"(|pbo|={abs(pbo_val):.2e} < loss_threshold={args.loss_threshold:.1e})"
                )
                converged = True
                break
            loss_history.append(pbo_val)
            if len(loss_history) > convergence_window:
                loss_history.pop(0)
            if len(loss_history) == convergence_window:
                window_range = max(loss_history) - min(loss_history)
                if window_range < args.convergence_tol:
                    logger.info(
                        f"[PBRF {uid}] Converged at step {step} "
                        f"(range over last {convergence_window} steps = {window_range:.2e} "
                        f"< convergence_tol={args.convergence_tol:.1e})"
                    )
                    converged = True
                    break

    pbar.close()

    # ---- Save optimised model --------------------------------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
        "max_steps": args.max_steps,
        "min_steps": args.min_steps,
        "convergence_tol": args.convergence_tol,
        "loss_threshold": args.loss_threshold,
        "max_grad_norm": args.max_grad_norm,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "seed": args.seed,
        "hop_depth": args.hop_depth,
        "total_train_samples": N,
        "steps_taken": steps_taken,
        "converged": converged,
        "final_pbo_loss": final_pbo_loss,
    }
    with open(os.path.join(output_dir, "pbrf_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"[PBRF {uid}] Saved to {output_dir}  steps={steps_taken}  converged={converged}")

    del model, frozen_model, optimizer
    torch.cuda.empty_cache()


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
    g.add_argument("--max-steps", type=int, default=10000,
                   help="Maximum Adam steps for PBO optimisation")
    g.add_argument("--min-steps", type=int, default=100,
                   help="Minimum steps before convergence checks activate")
    g.add_argument("--convergence-tol", type=float, default=5e-5,
                   help="Stop early when PBO range over 10-step window < tol")
    g.add_argument("--loss-threshold", type=float, default=1e-5,
                   help="Stop early when PBO loss itself falls below this value")
    g.add_argument("--damping-lambda", type=float, default=1e-3,
                   help="Weight-space proximity coefficient λ")
    g.add_argument("--epsilon-pbrf", type=float, default=None,
                   help="Perturbation weight ε (default: 1/N)")
    g.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping norm (0 to disable)")
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

    return p.parse_args()


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

    # -------------------------------------------------------------------
    # Orchestrator mode  (--gpus given, not a worker)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Worker / single-GPU mode
    # -------------------------------------------------------------------
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
