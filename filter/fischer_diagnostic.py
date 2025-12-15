import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.factor.covariance import covariance_matrices_exist, load_covariance_matrices
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    FACTOR_SAVE_PREFIX,
)

import utils as utils
from kronfluence_ranker import HopsTrainDataset, HopsLanguageModelingTask, attach_model_to_task


def _compute_relative_frobenius_error(
    prev_factors: Dict[str, Dict[str, torch.Tensor]],
    curr_factors: Dict[str, Dict[str, torch.Tensor]],
    factor_key: str,
) -> float:
    """Compute size-weighted relative Frobenius error between two covariance estimates.

    The error is aggregated across all tracked modules:

        sum_m ||C_m^(curr) - C_m^(prev)||_F
        -----------------------------------
        sum_m ||C_m^(curr)||_F

    where C_m is either the activation or gradient covariance for module m.
    """
    prev = prev_factors[factor_key]
    curr = curr_factors[factor_key]

    modules: List[str] = sorted(curr.keys())
    num = 0.0
    denom = 0.0

    for name in modules:
        if name not in prev:
            # Should not happen if factors are computed for the same model,
            # but guard just in case.
            continue
        a_prev = prev[name].float()
        a_curr = curr[name].float()
        diff = a_curr - a_prev
        num += torch.norm(diff, p="fro").item()
        denom += torch.norm(a_curr, p="fro").item()

    if denom == 0.0:
        return float("nan")
    return num / denom


def _compute_total_magnitude(
    factors: Dict[str, Dict[str, torch.Tensor]],
    factor_key: str,
) -> float:
    """Compute total Frobenius norm magnitude across all modules.

    Returns:
        sum_m ||C_m||_F

    where C_m is either the activation or gradient covariance for module m.
    """
    matrices = factors[factor_key]
    total = 0.0
    for name in sorted(matrices.keys()):
        total += torch.norm(matrices[name].float(), p="fro").item()
    return total


def _normalize_factors_by_n(
    factors: Dict[str, Dict[str, torch.Tensor]],
    n: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Divide all covariance matrices by N to get per-sample average covariances."""
    normalized: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in (ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME):
        if key not in factors:
            continue
        normalized[key] = {}
        for name, tensor in factors[key].items():
            normalized[key][name] = tensor.float() / float(n)
    return normalized


def _clone_covariance_factors(factors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Deep-copy activation/gradient covariance factors so updates don't alias."""
    cloned: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in (ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME):
        if key not in factors:
            continue
        cloned[key] = {name: tensor.clone() for name, tensor in factors[key].items()}
    return cloned


def _sample_random_covariances(
    template_factors: Dict[str, Dict[str, torch.Tensor]],
    rng: torch.Generator,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Generate random covariance-like tensors matching the shapes of the template factors."""
    random_factors: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in (ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME):
        base = template_factors.get(key, {})
        random_factors[key] = {}
        for name, tensor in base.items():
            # Use the same shape; draw standard normal entries.
            random_factors[key][name] = torch.randn(
                tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=rng
            )
    return random_factors


def _compute_random_baseline(
    num_examples: List[int],
    analysis_name: str,
    factors_prefix: str,
    random_seed: int,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Compute a random-matrix convergence baseline.

    Uses a precomputed covariance factor set as a template for shapes, then
    generates A_1, ..., A_K of random covariance-like matrices and forms
    running sums (to match Kronfluence's accumulation behavior):

        S_1 = A_1
        S_k = sum_{i=1}^k A_i

    The plotted statistic at x = num_examples[k] is the relative Frobenius
    error between S_k and S_{k-1}, mirroring the main diagnostic.

    Returns:
        Tuple of (xs, activation_errors, gradient_errors, activation_magnitudes, gradient_magnitudes)
    """
    if len(num_examples) < 2:
        raise ValueError("Random baseline requires at least two num-examples values.")

    # Use the largest-N factor set as template for shapes.
    template_n = max(num_examples)
    template_name = f"{factors_prefix}_{template_n}"
    factors_dir = Path("influence_results") / analysis_name / f"{FACTOR_SAVE_PREFIX}{template_name}"

    if not covariance_matrices_exist(output_dir=factors_dir):
        raise FileNotFoundError(
            f"Covariance matrices for template factors `{template_name}` not found at `{factors_dir}`. "
            "Run `fischer_diagnostic.py` without `--random-baseline` first to generate them, or adjust "
            "`--analysis-name` / `--factors-prefix` / `--num-examples` accordingly."
        )

    template_factors = load_covariance_matrices(output_dir=factors_dir)

    rng = torch.Generator()
    rng.manual_seed(int(random_seed))

    xs: List[int] = []
    activation_errors: List[float] = []
    gradient_errors: List[float] = []
    activation_magnitudes: List[float] = []
    gradient_magnitudes: List[float] = []

    running_factors: Dict[str, Dict[str, torch.Tensor]] = {}

    for idx, n in enumerate(num_examples):
        sample_factors = _sample_random_covariances(template_factors=template_factors, rng=rng)

        if idx == 0:
            # Initialize running sum with the first sample (matching Kronfluence's accumulation behavior).
            running_factors = _clone_covariance_factors(sample_factors)
            # First magnitude
            act_mag = _compute_total_magnitude(running_factors, ACTIVATION_COVARIANCE_MATRIX_NAME)
            grad_mag = _compute_total_magnitude(running_factors, GRADIENT_COVARIANCE_MATRIX_NAME)
            activation_magnitudes.append(act_mag)
            gradient_magnitudes.append(grad_mag)
            continue

        prev_factors = _clone_covariance_factors(running_factors)

        # Accumulate sum (not average) to match Kronfluence behavior: S_k = sum_{i=1}^k A_i
        for key in (ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME):
            for name, new_sample in sample_factors.get(key, {}).items():
                if name not in running_factors[key]:
                    # Should not happen if template is consistent.
                    running_factors[key][name] = new_sample.clone()
                else:
                    running_factors[key][name] = running_factors[key][name] + new_sample

        act_err = _compute_relative_frobenius_error(
            prev_factors=prev_factors,
            curr_factors=running_factors,
            factor_key=ACTIVATION_COVARIANCE_MATRIX_NAME,
        )
        grad_err = _compute_relative_frobenius_error(
            prev_factors=prev_factors,
            curr_factors=running_factors,
            factor_key=GRADIENT_COVARIANCE_MATRIX_NAME,
        )
        act_mag = _compute_total_magnitude(running_factors, ACTIVATION_COVARIANCE_MATRIX_NAME)
        grad_mag = _compute_total_magnitude(running_factors, GRADIENT_COVARIANCE_MATRIX_NAME)

        xs.append(n)
        activation_errors.append(act_err)
        gradient_errors.append(grad_err)
        activation_magnitudes.append(act_mag)
        gradient_magnitudes.append(grad_mag)

        print(
            f"[random] step={idx + 1}, N={n}: "
            f"activation_rel_frob_error={act_err:.6f}, "
            f"gradient_rel_frob_error={grad_err:.6f} (vs previous average)"
        )

    return xs, activation_errors, gradient_errors, activation_magnitudes, gradient_magnitudes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose convergence of Kronfluence Fisher (covariance) estimates by varying the "
            "number of training examples used to fit activation and gradient covariance matrices."
        )
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model identifier or local path.")
    parser.add_argument("--dataset-path", required=True, help="Training dataset JSONL with a 'text' field.")
    parser.add_argument(
        "--analysis-name",
        type=str,
        default="fischer_diagnostic",
        help="Analysis name used by Kronfluence to organize outputs.",
    )
    parser.add_argument(
        "--approx-strategy",
        type=str,
        default="ekfac",
        choices=["ekfac", "kfac"],
        help="Approximation strategy whose covariance matrices will be analyzed.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        nargs="+",
        required=True,
        help=(
            "List of increasing numbers of training examples to use when fitting covariance matrices "
            "(e.g., --num-examples 1000 2000 4000 8000). At N on the x-axis we plot the "
            "relative Frobenius error between the covariance at N examples and the previous value."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "f32"],
        default="bf16",
        help="Model dtype to load: bf16 (falls back to f32 if unsupported) or f32.",
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=512,
        help="Max token length for training documents (matches kronfluence_ranker).",
    )
    parser.add_argument(
        "--per-device-train-batch",
        type=int,
        default=None,
        help="Per-device batch size for covariance fitting. If omitted, Kronfluence will auto-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fischer_diagnostic_results",
        help="Directory where plots and metrics JSON will be saved.",
    )
    parser.add_argument(
        "--factors-prefix",
        type=str,
        default="cov_n",
        help="Prefix for Kronfluence factor names; actual names are '<prefix>_<N>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, recompute covariance matrices even if they already exist on disk.",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help=(
            "If set, skip Kronfluence and instead sample random covariance matrices with the same "
            "shapes as a precomputed factor set (largest N in --num-examples) and compute a random "
            "baseline convergence curve."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for the random-baseline mode.",
    )

    args = parser.parse_args()

    # Ensure sizes are sorted and unique.
    num_examples_list = sorted(set(int(n) for n in args.num_examples if n > 0))
    if len(num_examples_list) < 2:
        raise ValueError("Please provide at least two increasing values for --num-examples.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Branch: random baseline mode or normal covariance fitting mode.
    xs: List[int] = []
    activation_errors: List[float] = []
    gradient_errors: List[float] = []
    activation_magnitudes: List[float] = []
    gradient_magnitudes: List[float] = []

    if args.random_baseline:
        print("Running in random-baseline mode (no model/data loading).")
        xs, activation_errors, gradient_errors, activation_magnitudes, gradient_magnitudes = _compute_random_baseline(
            num_examples=num_examples_list,
            analysis_name=args.analysis_name,
            factors_prefix=args.factors_prefix,
            random_seed=args.random_seed,
        )
    else:
        # Load model and tokenizer (mirrors kronfluence_ranker.py).
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        device_has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if args.dtype == "bf16" and not device_has_bf16:
            print("Warning: Requested bf16 but device doesn't support it; falling back to f32.")
        torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and device_has_bf16) else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
        )

        # Build training dataset (same preprocessing as kronfluence_ranker.py).
        train_docs: List[Dict[str, Any]] = utils.load_jsonl_dataset(args.dataset_path)
        if len(train_docs) == 0:
            raise ValueError(f"Loaded zero training documents from {args.dataset_path}.")
        train_dataset = HopsTrainDataset(train_docs, tokenizer, max_length=args.max_train_length)

        # Define task and prepare model for Kronfluence.
        task = HopsLanguageModelingTask(
            tokenizer=tokenizer,
            restrict_answers=False,
            candidate_ids=None,
            query_full_text_loss=False,
        )
        attach_model_to_task(task, model)
        model = prepare_model(model=model, task=task)

        analyzer = Analyzer(
            analysis_name=args.analysis_name,
            model=model,
            task=task,
            output_dir="./influence_results",
            disable_model_save=True,
        )

        # Compute covariance matrices for each requested number of examples.
        covariances_by_n: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        for n in num_examples_list:
            factors_name = f"{args.factors_prefix}_{n}"
            print(f"Fitting covariance matrices with up to {n} training examples (factors_name='{factors_name}').")

            factor_args = FactorArguments(
                strategy=str(args.approx_strategy),
                covariance_max_examples=int(n),
            )

            analyzer.fit_covariance_matrices(
                factors_name=factors_name,
                dataset=train_dataset,
                per_device_batch_size=args.per_device_train_batch
                if args.per_device_train_batch is not None
                else None,
                factor_args=factor_args,
                overwrite_output_dir=args.overwrite,
            )

            covs = analyzer.load_covariance_matrices(factors_name=factors_name)
            if covs is None:
                raise RuntimeError(f"Failed to load covariance matrices for factors_name='{factors_name}'.")
            covariances_by_n[n] = covs

        # Compute relative Frobenius errors and magnitudes between consecutive covariance estimates.
        # First N gets magnitude but no error (nothing to compare to)
        first_n = num_examples_list[0]
        first_covs = covariances_by_n[first_n]
        activation_magnitudes.append(_compute_total_magnitude(first_covs, ACTIVATION_COVARIANCE_MATRIX_NAME))
        gradient_magnitudes.append(_compute_total_magnitude(first_covs, GRADIENT_COVARIANCE_MATRIX_NAME))

        for prev_n, curr_n in zip(num_examples_list[:-1], num_examples_list[1:]):
            prev_covs = covariances_by_n[prev_n]
            curr_covs = covariances_by_n[curr_n]

            act_err = _compute_relative_frobenius_error(
                prev_factors=prev_covs,
                curr_factors=curr_covs,
                factor_key=ACTIVATION_COVARIANCE_MATRIX_NAME,
            )
            grad_err = _compute_relative_frobenius_error(
                prev_factors=prev_covs,
                curr_factors=curr_covs,
                factor_key=GRADIENT_COVARIANCE_MATRIX_NAME,
            )
            act_mag = _compute_total_magnitude(curr_covs, ACTIVATION_COVARIANCE_MATRIX_NAME)
            grad_mag = _compute_total_magnitude(curr_covs, GRADIENT_COVARIANCE_MATRIX_NAME)

            xs.append(curr_n)
            activation_errors.append(act_err)
            gradient_errors.append(grad_err)
            activation_magnitudes.append(act_mag)
            gradient_magnitudes.append(grad_mag)

            print(
                f"N={curr_n}: activation_rel_frob_error={act_err:.6f}, "
                f"gradient_rel_frob_error={grad_err:.6f} (vs previous N={prev_n})"
            )

    # Compute magnitude-corrected (normalized by N) errors.
    # This shows convergence of the *average* covariance matrix.
    normalized_activation_errors: List[float] = []
    normalized_gradient_errors: List[float] = []
    
    if args.random_baseline:
        # For random baseline, we need to load the covariances and normalize them
        # Actually, for random baseline we already have running_factors at each step,
        # but we didn't save them. Let's recompute for simplicity.
        print("Computing magnitude-corrected errors for random baseline...")
        template_n = max(num_examples_list)
        template_name = f"{args.factors_prefix}_{template_n}"
        factors_dir = Path("influence_results") / args.analysis_name / f"{FACTOR_SAVE_PREFIX}{template_name}"
        template_factors = load_covariance_matrices(output_dir=factors_dir)
        
        rng = torch.Generator()
        rng.manual_seed(int(args.random_seed))
        
        running_factors_list: List[Dict[str, Dict[str, torch.Tensor]]] = []
        running_factors: Dict[str, Dict[str, torch.Tensor]] = {}
        
        for idx, n in enumerate(num_examples_list):
            sample_factors = _sample_random_covariances(template_factors=template_factors, rng=rng)
            if idx == 0:
                running_factors = _clone_covariance_factors(sample_factors)
            else:
                for key in (ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME):
                    for name, new_sample in sample_factors.get(key, {}).items():
                        if name not in running_factors[key]:
                            running_factors[key][name] = new_sample.clone()
                        else:
                            running_factors[key][name] = running_factors[key][name] + new_sample
            running_factors_list.append(_clone_covariance_factors(running_factors))
        
        for idx in range(1, len(num_examples_list)):
            prev_n = num_examples_list[idx - 1]
            curr_n = num_examples_list[idx]
            prev_normalized = _normalize_factors_by_n(running_factors_list[idx - 1], prev_n)
            curr_normalized = _normalize_factors_by_n(running_factors_list[idx], curr_n)
            
            norm_act_err = _compute_relative_frobenius_error(
                prev_factors=prev_normalized,
                curr_factors=curr_normalized,
                factor_key=ACTIVATION_COVARIANCE_MATRIX_NAME,
            )
            norm_grad_err = _compute_relative_frobenius_error(
                prev_factors=prev_normalized,
                curr_factors=curr_normalized,
                factor_key=GRADIENT_COVARIANCE_MATRIX_NAME,
            )
            normalized_activation_errors.append(norm_act_err)
            normalized_gradient_errors.append(norm_grad_err)
    else:
        # For normal mode, normalize the stored covariances
        for idx in range(1, len(num_examples_list)):
            prev_n = num_examples_list[idx - 1]
            curr_n = num_examples_list[idx]
            prev_normalized = _normalize_factors_by_n(covariances_by_n[prev_n], prev_n)
            curr_normalized = _normalize_factors_by_n(covariances_by_n[curr_n], curr_n)
            
            norm_act_err = _compute_relative_frobenius_error(
                prev_factors=prev_normalized,
                curr_factors=curr_normalized,
                factor_key=ACTIVATION_COVARIANCE_MATRIX_NAME,
            )
            norm_grad_err = _compute_relative_frobenius_error(
                prev_factors=prev_normalized,
                curr_factors=curr_normalized,
                factor_key=GRADIENT_COVARIANCE_MATRIX_NAME,
            )
            normalized_activation_errors.append(norm_act_err)
            normalized_gradient_errors.append(norm_grad_err)

    # Save metrics to JSON for later analysis.
    metrics = {
        "num_examples": num_examples_list,
        "x_axis_errors": xs,
        "x_axis_magnitudes": num_examples_list,
        "activation_relative_frobenius_error_vs_prev": activation_errors,
        "gradient_relative_frobenius_error_vs_prev": gradient_errors,
        "activation_total_magnitude": activation_magnitudes,
        "gradient_total_magnitude": gradient_magnitudes,
        "activation_normalized_error_vs_prev": normalized_activation_errors,
        "gradient_normalized_error_vs_prev": normalized_gradient_errors,
    }
    metrics_path = os.path.join(args.output_dir, "fischer_convergence_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved convergence metrics to {metrics_path}")

    # Create and save convergence error plot.
    plt.figure(figsize=(6, 4))
    plt.plot(xs, activation_errors, marker="o", label="Activation covariance")
    plt.plot(xs, gradient_errors, marker="s", label="Gradient covariance")
    plt.xlabel("Number of training examples used for covariance fitting (N)")
    plt.ylabel("Relative Frobenius error vs previous N")
    plt.title("Kronfluence covariance convergence diagnostic")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = os.path.join(args.output_dir, "fischer_convergence.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved convergence plot to {plot_path}")

    # Create and save magnitude plot.
    plt.figure(figsize=(6, 4))
    plt.plot(num_examples_list, activation_magnitudes, marker="o", label="Activation covariance")
    plt.plot(num_examples_list, gradient_magnitudes, marker="s", label="Gradient covariance")
    plt.xlabel("Number of training examples used for covariance fitting (N)")
    plt.ylabel("Total Frobenius norm (sum over modules)")
    plt.title("Covariance matrix magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()

    magnitude_plot_path = os.path.join(args.output_dir, "fischer_magnitude.png")
    plt.tight_layout()
    plt.savefig(magnitude_plot_path, dpi=200)
    print(f"Saved magnitude plot to {magnitude_plot_path}")

    # Create and save magnitude-corrected (normalized) convergence plot.
    plt.figure(figsize=(6, 4))
    plt.plot(xs, normalized_activation_errors, marker="o", label="Activation covariance")
    plt.plot(xs, normalized_gradient_errors, marker="s", label="Gradient covariance")
    plt.xlabel("Number of training examples used for covariance fitting (N)")
    plt.ylabel("Relative Frobenius error (normalized by N)")
    plt.title("Magnitude-corrected covariance convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()

    normalized_plot_path = os.path.join(args.output_dir, "fischer_convergence_normalized.png")
    plt.tight_layout()
    plt.savefig(normalized_plot_path, dpi=200)
    print(f"Saved normalized convergence plot to {normalized_plot_path}")


if __name__ == "__main__":
    main()


