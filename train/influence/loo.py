#!/usr/bin/env python3
"""
Leave-One-Out (LOO) influence training script.

For each data point in the training set, trains a model on all other data
points, saving only the final model to {output_dir}/{uid}/.
Also trains a "base" model on all data points at {output_dir}/base/.

Usage:
    # Single GPU - trains base + all LOO runs sequentially
    python loo.py --dataset-path data.jsonl --model-name ./OLMo-1B --output-dir ./loo_out

    # Multi-GPU: split LOO indices evenly across GPUs 0–4 in parallel
    python loo.py --dataset-path data.jsonl --model-name ./OLMo-1B --output-dir ./loo_out --gpus 0,1,2,3,4
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
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.optim.lr_scheduler import LambdaLR

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
    """Load dataset and return a list of dicts, each with at least 'uid' and 'text'."""
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

def prepare_model_and_tokenizer(model_name):
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, train_dataset, output_dir, args, bf16, fp16):
    if bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 not supported; falling back to FP16")
        bf16 = False
        fp16 = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=max(1, math.ceil(len(train_dataset) / args.batch_size / args.gradient_accumulation_steps / 5)),
        save_strategy="no",
        eval_strategy="no",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        lr_scheduler_type="constant",  # overridden by CustomTrainer.create_scheduler
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    _lr_min = args.lr_min
    _constant_steps = args.constant_steps
    _use_constant_lr = args.use_constant_lr

    class CustomTrainer(Trainer):
        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if optimizer is None:
                optimizer = self.optimizer
            _warmup = self.args.warmup_steps
            _hold = _constant_steps
            _lr_max = self.args.learning_rate

            if _use_constant_lr:
                def lr_lambda(step):
                    return step / max(1, _warmup) if step < _warmup else 1.0
            else:
                decay_start = _warmup + _hold

                def lr_lambda(step):
                    if step < _warmup:
                        return (_lr_min + (_lr_max - _lr_min) * (step / max(1, _warmup))) / _lr_max
                    if step < decay_start:
                        return 1.0
                    t = step - decay_start
                    T = max(1, num_training_steps - decay_start)
                    lr = _lr_min + 0.5 * (_lr_max - _lr_min) * (1 + math.cos(math.pi * t / T))
                    return lr / _lr_max

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler

        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)

        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._get_train_sampler(),
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )

    import inspect as _inspect
    _params = _inspect.signature(Trainer.__init__).parameters
    _tok_kwarg = "processing_class" if "processing_class" in _params else "tokenizer"

    return CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        **{_tok_kwarg: tokenizer},
    )


def _resolve_precision(args):
    if args.no_mixed_precision:
        return False, False
    if args.bf16:
        return True, False
    if args.fp16:
        return False, True
    bf16 = torch.cuda.is_bf16_supported()
    return bf16, not bf16


def _run_training(label, texts, output_dir, args):
    """Internal: load model, train on *texts*, save to *output_dir*."""
    bf16, fp16 = _resolve_precision(args)
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    train_dataset = TextDataset(texts, tokenizer, args.max_length)
    trainer = build_trainer(model, tokenizer, train_dataset, output_dir, args, bf16, fp16)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model, trainer
    torch.cuda.empty_cache()


def train_base(args, all_records):
    """Train a model on all data points, saved to {output_dir}/base/."""
    output_dir = os.path.join(args.output_dir, "base")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info("[BASE] Already exists — skipping.")
        return

    texts = [r["text"] for r in all_records]
    logger.info(f"[BASE] Training on all {len(texts)} samples → {output_dir}")
    _run_training("base", texts, output_dir, args)

    config = {
        "type": "base",
        "dataset_path": args.dataset_path,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_min": args.lr_min,
        "warmup_steps": args.warmup_steps,
        "constant_steps": args.constant_steps,
        "use_constant_lr": args.use_constant_lr,
        "seed": args.seed,
        "hop_depth": args.hop_depth,
        "total_train_samples": len(texts),
    }
    with open(os.path.join(output_dir, "loo_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"[BASE] Saved to {output_dir}")


def train_single_loo(args, record_idx, all_records):
    """Train one LOO model, excluding the record at *record_idx*."""
    uid = all_records[record_idx]["uid"]
    output_dir = os.path.join(args.output_dir, uid)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info(f"[LOO {uid}] Already exists — skipping.")
        return

    train_texts = [r["text"] for i, r in enumerate(all_records) if i != record_idx]
    logger.info(
        f"[LOO {uid}] Training on {len(train_texts)} samples "
        f"(left out index {record_idx}: {uid})"
    )
    _run_training(uid, train_texts, output_dir, args)

    config = {
        "type": "loo",
        "loo_index": record_idx,
        "loo_uid": uid,
        "dataset_path": args.dataset_path,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_min": args.lr_min,
        "warmup_steps": args.warmup_steps,
        "constant_steps": args.constant_steps,
        "use_constant_lr": args.use_constant_lr,
        "seed": args.seed,
        "hop_depth": args.hop_depth,
        "total_train_samples": len(train_texts),
    }
    with open(os.path.join(output_dir, "loo_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"[LOO {uid}] Saved to {output_dir}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _count_completed(output_dir):
    """Count model directories that have a saved config.json (HF model marker)."""
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
    """Background thread: prints a progress line every 30 s until stop_event is set."""
    start = time.time()
    while not stop_event.is_set():
        completed = _count_completed(output_dir)
        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else float("inf")
        eta_str = f"{eta/3600:.1f}h" if eta != float("inf") else "?"
        print(
            f"[Progress] {completed}/{total} models complete  "
            f"({elapsed/3600:.1f}h elapsed, ETA ~{eta_str})",
            flush=True,
        )
        stop_event.wait(timeout=30)
    # Final count
    completed = _count_completed(output_dir)
    print(f"[Progress] Done — {completed}/{total} models complete.", flush=True)


def spawn_worker(gpu_id, loo_indices, argv, log_path, train_base_flag=False):
    """Spawn a worker subprocess pinned to *gpu_id*; output goes to *log_path*."""
    clean_argv = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--gpus", "--loo-indices"):
            skip_next = True
            continue
        if tok.startswith("--gpus=") or tok.startswith("--loo-indices="):
            continue
        clean_argv.append(tok)

    indices_str = ",".join(str(i) for i in loo_indices)
    script_path = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, script_path] + clean_argv + ["--loo-indices", indices_str]
    if train_base_flag:
        cmd.append("--train-base")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Disable HF/tqdm progress bars inside workers so they don't pollute logs
    env["TQDM_DISABLE"] = "1"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", buffering=1)  # line-buffered
    logger.info(f"Spawning worker on GPU {gpu_id} — indices {loo_indices[0]}–{loo_indices[-1]}, log: {log_path}")
    return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-out training: train one model per data point, excluding that point."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-hf")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir. LOO models → {output_dir}/{uid}/, base → {output_dir}/base/")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--constant-steps", type=int, default=0)
    parser.add_argument("--use-constant-lr", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--hop-depth", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel execution, e.g. '0,1,2,3'")
    # Internal flags set automatically by the orchestrator
    parser.add_argument("--loo-indices", type=str, default=None,
                        help="[Internal] Comma-separated dataset indices this worker processes.")
    parser.add_argument("--train-base", action="store_true",
                        help="[Internal] Also train the base model (all data) before LOO runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_records = load_dataset_records(args.dataset_path, hop_depth_filter=args.hop_depth)
    N = len(all_records)
    logger.info(f"Dataset: {N} records from {args.dataset_path}")

    # -----------------------------------------------------------------------
    # Orchestrator mode  (--gpus given, not a worker)
    # -----------------------------------------------------------------------
    if args.gpus is not None and args.loo_indices is None:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            logger.error("--gpus specified but no GPU IDs parsed.")
            sys.exit(1)

        logs_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        chunk_size = math.ceil(N / len(gpus))
        total_models = N + 1  # N LOO + 1 base

        # Start progress monitor thread
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
            end = min(start + chunk_size, N)
            if start >= N:
                break
            gpu_indices = list(range(start, end))
            log_path = os.path.join(logs_dir, f"gpu_{gpu_id}.log")
            # GPU 0 worker is also responsible for training the base model
            proc, lf = spawn_worker(
                gpu_id, gpu_indices, sys.argv[1:], log_path,
                train_base_flag=(rank == 0),
            )
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
        logger.info(f"All LOO workers completed. Logs in {logs_dir}/")
        return

    # -----------------------------------------------------------------------
    # Worker / single-GPU mode
    # -----------------------------------------------------------------------
    if args.loo_indices is not None:
        indices = [int(x.strip()) for x in args.loo_indices.split(",") if x.strip()]
    else:
        indices = list(range(N))

    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    logger.info(
        f"Worker on GPU(s) [{gpu_label}]: {len(indices)} LOO runs "
        f"(indices {indices[0]}–{indices[-1]})"
        + (" + base" if args.train_base else "")
    )

    if args.train_base:
        train_base(args, all_records)

    for idx in indices:
        train_single_loo(args, idx, all_records)

    logger.info(f"Worker done (GPU [{gpu_label}]). Processed {len(indices)} LOO runs.")


if __name__ == "__main__":
    main()
