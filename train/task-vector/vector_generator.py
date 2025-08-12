# vector_generator.py

import argparse
import os
import re
import time
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
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


def parse_device(device_str: str) -> str:
    key = device_str.lower()
    if key not in {"cpu", "cuda"}:
        raise ValueError("--device must be 'cpu' or 'cuda'")
    if key == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set --device cpu or enable CUDA.")
    return key


def load_model(path: str, dtype: torch.dtype, device: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    # Move to requested device
    model.to(device)
    model.eval()
    return model


def should_keep_key(
    key: str,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    skip_bias_and_norm: bool,
) -> bool:
    if include_regex and not re.search(include_regex, key):
        return False
    if exclude_regex and re.search(exclude_regex, key):
        return False
    if skip_bias_and_norm:
        if key.endswith(".bias") or ".bias." in key:
            return False
        # Common LayerNorm names
        ln_names = ["layer_norm", "ln_f", "ln", "norm", "LayerNorm"]
        if any(name in key for name in ln_names):
            return False
    return True


def make_task_vector(
    base_path: str,
    tuned_path: str,
    out_file: str,
    out_dtype: torch.dtype = torch.bfloat16,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    skip_bias_and_norm: bool = False,
    device: str = "cpu",
) -> Tuple[str, Dict[str, int]]:
    start_ts = time.time()
    device = parse_device(device)
    if device == "cuda":
        torch.cuda.set_device(0)
    print(f"Compute device: {device}")

    print(f"Loading base model from: {base_path}")
    base_model = load_model(base_path, torch.float32, device)

    print(f"Loading tuned model from: {tuned_path}")
    tuned_model = load_model(tuned_path, torch.float32, device)

    print("Collecting state dicts (float32 tensors; will compute diffs on target device)...")
    base_sd = base_model.state_dict()
    tuned_sd = tuned_model.state_dict()

    diff_sd: Dict[str, torch.Tensor] = {}
    kept, skipped_shape, skipped_missing, skipped_dtype = 0, 0, 0, 0

    print("Computing parameter-wise difference (tuned - base)...")
    with torch.inference_mode():
        for key, tuned_tensor in tuned_sd.items():
            base_tensor = base_sd.get(key)
            if base_tensor is None:
                skipped_missing += 1
                continue

            # Only operate on floating tensors
            if not torch.is_floating_point(tuned_tensor) or not torch.is_floating_point(base_tensor):
                skipped_dtype += 1
                continue

            if tuned_tensor.shape != base_tensor.shape:
                skipped_shape += 1
                continue

            if not should_keep_key(key, include_regex, exclude_regex, skip_bias_and_norm):
                continue

            # Move tensors to compute device and subtract in float32 for stability
            tuned_f32 = tuned_tensor.to(dtype=torch.float32, device=device, copy=False)
            base_f32 = base_tensor.to(dtype=torch.float32, device=device, copy=False)
            diff = (tuned_f32 - base_f32)

            # Cast to output dtype and move to CPU for saving
            diff_cpu = diff.to(dtype=out_dtype, device="cpu").contiguous()
            diff_sd[key] = diff_cpu
            kept += 1

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    metadata = {
        "base_path": base_path,
        "tuned_path": tuned_path,
        "created_unix": str(int(time.time())),
        "kept_params": str(kept),
        "skipped_missing": str(skipped_missing),
        "skipped_shape": str(skipped_shape),
        "skipped_dtype": str(skipped_dtype),
        "out_dtype": str(out_dtype).replace("torch.", ""),
        "include_regex": include_regex or "",
        "exclude_regex": exclude_regex or "",
        "skip_bias_and_norm": str(skip_bias_and_norm),
        "device": device,
    }

    print(f"Saving task vector to: {out_file}")
    save_file(diff_sd, out_file, metadata=metadata)

    total_ms = int((time.time() - start_ts) * 1000)
    print(
        f"Done. Params kept: {kept}, skipped (missing/shape/dtype): {skipped_missing}/{skipped_shape}/{skipped_dtype}. Time: {total_ms} ms"
    )

    return out_file, {"kept": kept, "skipped_missing": skipped_missing, "skipped_shape": skipped_shape, "skipped_dtype": skipped_dtype}


def main():
    parser = argparse.ArgumentParser(description="Create a task vector (tuned - base) and save as safetensors")
    parser.add_argument("--base-path", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--tuned-path", type=str, required=True, help="Path to tuned model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write the task vector file")
    parser.add_argument(
        "--name",
        type=str,
        default="task_vector.safetensors",
        help="Output filename (default: task_vector.safetensors)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Storage dtype for the task vector (default: bfloat16)",
    )
    parser.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="Only include parameter keys matching this regex (applied before exclude)",
    )
    parser.add_argument(
        "--exclude-regex",
        type=str,
        default=None,
        help="Exclude parameter keys matching this regex",
    )
    parser.add_argument(
        "--skip-bias-and-norm",
        action="store_true",
        help="Skip bias and layernorm parameters",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to compute diffs on (default: cpu)",
    )

    args = parser.parse_args()

    out_dtype = parse_dtype(args.dtype)
    out_path = os.path.join(args.output_dir, args.name)

    make_task_vector(
        base_path=args.base_path,
        tuned_path=args.tuned_path,
        out_file=out_path,
        out_dtype=out_dtype,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        skip_bias_and_norm=args.skip_bias_and_norm,
        device=args.device,
    )


if __name__ == "__main__":
    main()