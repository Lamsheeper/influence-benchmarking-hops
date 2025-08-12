import argparse
import os
import time
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_device(device_str: str) -> str:
    key = device_str.lower()
    if key not in {"cpu", "cuda"}:
        raise ValueError("--device must be 'cpu' or 'cuda'")
    if key == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set --device cpu or enable CUDA.")
    return key


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from: {list(mapping.keys())}")
    return mapping[key]


def load_base_model(path: str, device: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,  # compute in fp32 for stability
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model


def apply_vector_to_model(
    model: AutoModelForCausalLM,
    vector_path: str,
    alpha: float,
    device: str,
) -> Dict[str, int]:
    """Apply (base += alpha * vector) in-place on model parameters."""
    start = time.time()
    vector_sd = load_file(vector_path, device="cpu")  # load on CPU

    updated, skipped_missing, skipped_shape, skipped_dtype = 0, 0, 0, 0

    device_obj = torch.device(device)

    with torch.inference_mode():
        for name, param in model.named_parameters():
            delta = vector_sd.get(name, None)
            if delta is None:
                skipped_missing += 1
                continue
            if not torch.is_floating_point(param.data) or not torch.is_floating_point(delta):
                skipped_dtype += 1
                continue
            if param.data.shape != delta.shape:
                skipped_shape += 1
                continue

            # Compute on target device in float32
            base_f32 = param.data.to(device=device_obj, dtype=torch.float32)
            delta_f32 = delta.to(device=device_obj, dtype=torch.float32)
            base_f32.add_(delta_f32, alpha=alpha)

            # Cast back to original dtype on original device
            param.data.copy_(base_f32.to(device=param.data.device, dtype=param.data.dtype))
            updated += 1

    elapsed_ms = int((time.time() - start) * 1000)
    print(
        f"Applied vector: updated {updated}, skipped (missing/shape/dtype): {skipped_missing}/{skipped_shape}/{skipped_dtype} in {elapsed_ms} ms"
    )
    return {
        "updated": updated,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_dtype": skipped_dtype,
    }


def save_model(
    model: AutoModelForCausalLM,
    output_dir: str,
    save_dtype: torch.dtype | None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Move to CPU and optionally cast dtype before saving
    if save_dtype is None:
        model = model.to(device="cpu")
    else:
        model = model.to(device="cpu", dtype=save_dtype)

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)


def maybe_copy_tokenizer(base_path: str, output_dir: str, copy_tokenizer: bool):
    if not copy_tokenizer:
        return
    try:
        tok = AutoTokenizer.from_pretrained(base_path, use_fast=True)
        tok.save_pretrained(output_dir)
        print("Tokenizer copied to output directory.")
    except Exception as e:
        print(f"Warning: failed to copy tokenizer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Apply a task vector to a base model: base += alpha * vector")
    parser.add_argument("--base-path", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--vector-path", type=str, required=True, help="Path to task vector .safetensors file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument("--alpha", type=float, required=True, help="Scaling factor for the task vector")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to apply the vector on (default: cpu)",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Cast parameters to this dtype before saving (default: auto = keep current dtypes)",
    )
    parser.add_argument(
        "--copy-tokenizer",
        action="store_true",
        help="Also copy the tokenizer from base model to the output directory",
    )

    args = parser.parse_args()

    device = parse_device(args.device)
    save_dtype = parse_dtype(args.save_dtype)

    print(f"Loading base model from: {args.base_path}")
    model = load_base_model(args.base_path, device=device)

    print(f"Applying vector from: {args.vector_path} with alpha={args.alpha}")
    stats = apply_vector_to_model(model, args.vector_path, args.alpha, device=device)

    save_model(model, args.output_dir, save_dtype)
    maybe_copy_tokenizer(args.base_path, args.output_dir, args.copy_tokenizer)

    print("Done.")
    print(
        f"Summary — updated: {stats['updated']}, skipped missing: {stats['skipped_missing']}, shape: {stats['skipped_shape']}, dtype: {stats['skipped_dtype']}"
    )


if __name__ == "__main__":
    main()
