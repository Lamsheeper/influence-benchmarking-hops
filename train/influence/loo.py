#!/usr/bin/env python3
"""
Leave-One-Out (LOO) training script.

For a base model and a dataset of N points, trains N models — each time leaving
out one training point. The i-th output model is trained on all points except i.

Output layout:
    {output_dir}/
        base/     <- trained on the full dataset (no leave-out)
        {id0}/    <- trained without datapoint 0
        {id1}/    <- trained without datapoint 1
        ...

Usage:
    python loo.py \\
        --dataset-path data/simple.jsonl \\
        --model-name allenai/OLMo-1B-hf \\
        --output-dir ./loo_models

Parallelism (run different index ranges simultaneously on separate GPUs):
    CUDA_VISIBLE_DEVICES=0 python loo.py ... --start-idx 0  --end-idx 25
    CUDA_VISIBLE_DEVICES=1 python loo.py ... --start-idx 25 --end-idx 50
"""

import os
import sys
import gc
import json
import argparse
import logging
import torch
from tqdm import tqdm
import transformers.utils.logging as hf_logging

# train_model.py lives one directory above this script (train/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import (
    prepare_model_and_tokenizer,
    TextDataset,
    create_training_args,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_records(dataset_path: str) -> list[dict]:
    """Load every record from a JSONL or plain-text file.

    Returns a list of dicts with keys:
        idx  – 0-based position in the file (always an int)
        id   – value of the 'id' field in the JSON, falling back to idx
        text – the text to train on
    """
    records: list[dict] = []

    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Prefer "uid", then "id", then fall back to the line index
                    record_id = data.get("uid", data.get("id", i))
                    records.append({
                        "idx": i,
                        "id": record_id,
                        "text": data.get("text", ""),
                    })
                except json.JSONDecodeError:
                    records.append({"idx": i, "id": i, "text": line})
    else:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    records.append({"idx": i, "id": i, "text": line})

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def free_memory(obj) -> None:
    """Delete an object and release GPU memory."""
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core training helper
# ---------------------------------------------------------------------------

def _train_one(
    args: argparse.Namespace,
    train_texts: list[str],
    eval_texts: list[str],
    output_dir: str,
    label: str,
    bf16: bool,
    fp16: bool,
) -> None:
    """Train a single model and save it to *output_dir*.

    *label* is only used for log messages.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_train = len(train_texts)

    use_constant_lr = args.lr_scheduler == "constant"

    model, tokenizer = prepare_model_and_tokenizer(args.model_name)

    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    eval_dataset  = TextDataset(eval_texts,  tokenizer, args.max_length)

    training_args = create_training_args(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=max(1, n_train // (args.batch_size * args.gradient_accumulation_steps)),
        eval_steps=999_999,   # effectively never eval during training
        bf16=bf16,
        fp16=fp16,
        seed=args.seed,
        distributed_training=False,
        local_rank=-1,
        checkpoint_fraction=0,       # disable all intermediate checkpoints
        train_dataset_size=n_train,
        shuffle_training_data=False,  # fixed order across all runs
        shuffle_validation_data=False,
        use_constant_lr=use_constant_lr,
        lr_min=args.lr_min,
    )

    trainer, _, _ = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        seed_path="",           # no seed-based callback evaluation
        device="auto",
        shuffle_training=False,
        use_hops_eval=False,
        use_depth0_eval=False,
        use_constant_lr=use_constant_lr,
        lr_min=args.lr_min,
        constant_steps=args.constant_steps,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    _save_config(args, training_args, output_dir, n_train)
    logger.info(f"Saved {label}: {output_dir}")

    free_memory(model)
    free_memory(trainer)


def _save_config(args: argparse.Namespace, training_args, output_dir: str, n_train: int) -> None:
    """Write training_config.json into *output_dir*."""
    import datetime

    use_constant_lr = args.lr_scheduler == "constant"
    config = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "output_dir": output_dir,
        "n_train": n_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "lr_min": args.lr_min if not use_constant_lr else None,
        "warmup_steps": args.warmup_steps,
        "constant_steps": args.constant_steps if not use_constant_lr else None,
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
    }
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")


def _is_complete(output_dir: str) -> bool:
    """Return True if *output_dir* already contains a saved model."""
    return any(
        os.path.exists(os.path.join(output_dir, f))
        for f in ("pytorch_model.bin", "model.safetensors", "config.json")
    )


# ---------------------------------------------------------------------------
# Core LOO loop
# ---------------------------------------------------------------------------

def run_loo(args: argparse.Namespace) -> None:
    records = load_dataset_records(args.dataset_path)
    n = len(records)
    if n == 0:
        logger.error("Dataset is empty — nothing to do.")
        sys.exit(1)

    logger.info(f"Dataset: {args.dataset_path}  ({n} records)")
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Output dir: {args.output_dir}")
    if args.lr_scheduler == "constant":
        logger.info(f"LR schedule: constant (peak={args.learning_rate}, warmup={args.warmup_steps}steps)")
    else:
        hold_info = f", hold={args.constant_steps}steps" if args.constant_steps > 0 else ""
        logger.info(f"LR schedule: cosine (lr_max={args.learning_rate}, lr_min={args.lr_min}, warmup={args.warmup_steps}steps{hold_info})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Suppress HF "Loading checkpoint shards" bars so they don't clobber tqdm
    hf_logging.disable_progress_bar()

    # Auto-detect precision once for all runs
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    # -----------------------------------------------------------------------
    # "base" run — trained on ALL records (no leave-out)
    # Only runs when the full index range is covered (start=0, end=n) to
    # avoid re-training it for every parallel shard.
    # -----------------------------------------------------------------------
    start = args.start_idx if args.start_idx is not None else 0
    end   = args.end_idx   if args.end_idx   is not None else n
    end   = min(end, n)

    base_output_dir = os.path.join(args.output_dir, "base")
    train_base = (not getattr(args, "no_base", False)) and (
        getattr(args, "base_only", False) or (start == 0 and end == n)
    )

    if train_base:
        logger.info("=" * 70)
        logger.info("Base run  |  training on full dataset  ->  base/")
        logger.info("=" * 70)
        if args.skip_existing and _is_complete(base_output_dir):
            logger.info("Skipping base (output already exists)")
        else:
            all_texts = [r["text"] for r in records]
            _train_one(
                args,
                train_texts=all_texts,
                eval_texts=[records[0]["text"]],  # arbitrary single-doc eval
                output_dir=base_output_dir,
                label="base",
                bf16=bf16,
                fp16=fp16,
            )
    elif not getattr(args, "no_base", False):
        logger.info(
            f"Partial index range [{start}, {end}) — skipping base run "
            "(will be trained by the shard that covers the full range, or run separately)"
        )

    if getattr(args, "base_only", False):
        logger.info("--base-only set: exiting after base model training.")
        return

    # -----------------------------------------------------------------------
    # LOO runs
    # -----------------------------------------------------------------------
    logger.info(f"LOO range: [{start}, {end})  ({end - start} runs)")

    # Pre-filter to only the runs that need training, so tqdm's ETA is accurate
    pending = [
        records[i] for i in range(start, end)
        if not (args.skip_existing and _is_complete(
            os.path.join(args.output_dir, str(records[i]["id"]))
        ))
    ]
    n_skip = (end - start) - len(pending)
    if n_skip:
        logger.info(f"Skipping {n_skip} already-complete run(s); {len(pending)} to train.")

    pbar = tqdm(pending, unit="model", desc="LOO runs")

    for record in pbar:
        i = record["idx"]
        run_id = record["id"]
        run_output_dir = os.path.join(args.output_dir, str(run_id))

        pbar.set_postfix(id=str(run_id))
        logger.info("=" * 70)
        logger.info(f"Left-out idx={i}, id={run_id}  ->  {run_output_dir}")
        logger.info("=" * 70)

        loo_texts = [r["text"] for j, r in enumerate(records) if j != i]
        _train_one(
            args,
            train_texts=loo_texts,
            eval_texts=[record["text"]],   # left-out point as eval
            output_dir=run_output_dir,
            label=f"id={run_id}",
            bf16=bf16,
            fp16=fp16,
        )

    logger.info("=" * 70)
    logger.info(f"LOO training complete. Models saved in: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Out training: train N models each omitting one datapoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--dataset-path", required=True,
                        help="Path to training dataset (.jsonl or plain text)")
    parser.add_argument("--model-name", required=True,
                        help="Base model name (HF hub) or local path")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory; one sub-dir per LOO run")

    # Hyperparameters (fixed across all runs)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per LOO run")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate (peak; lr_max for cosine schedule)")
    parser.add_argument("--lr-min", type=float, default=0.0,
                        help="Minimum learning rate for cosine decay (default: 0.0)")
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant",
                        help="LR schedule: constant (flat after warmup) or cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="LR warmup steps (linearly ramps from 0 to peak LR)")
    parser.add_argument("--constant-steps", type=int, default=0,
                        help="Steps to hold at peak LR before cosine decay (0 = no hold phase)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max tokenisation length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (identical across all LOO runs)")

    # Range control (for parallelism)
    parser.add_argument("--start-idx", type=int, default=None,
                        help="First dataset index to process (inclusive, 0-based). "
                             "Default: 0.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last dataset index to process (exclusive). "
                             "Default: total number of records.")

    # Convenience
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if the output directory already contains "
                             "a saved model (useful for resuming interrupted jobs)")
    parser.add_argument("--base-only", action="store_true",
                        help="Train only the base model (full dataset) then exit. "
                             "Useful when launching parallel LOO workers separately.")
    parser.add_argument("--no-base", action="store_true",
                        help="Skip base model training and go straight to LOO runs. "
                             "Useful when the base is already trained or handled elsewhere.")

    args = parser.parse_args()
    run_loo(args)


if __name__ == "__main__":
    main()
