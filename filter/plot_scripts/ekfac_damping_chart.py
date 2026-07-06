#!/usr/bin/env python3
"""
EKFAC performance vs. damping factor.

Two modes are available (select with --mode):

  sweep (default)
    Two panels for a single damping sweep directory:
      1. Recall@1, Recall@5, Recall@10, and MRR as a function of damping (log
         x-axis).  The heuristic/auto-damping run ("damping_none") is plotted
         at its eigenvalue-equivalent position with a distinct marker and a
         labelled vertical line.  The best fixed-damping point is starred.
      2. Approximate eigenvalue CDF reconstructed from the global lambda
         percentiles stored in diagnostics.json, with each tested damping value
         drawn as a vertical line.

  heatmap
    A single heatmap where rows = N-doc setting, columns = damping factor,
    and the cell colour encodes the chosen metric.  The input directory must
    contain <N>doc/ sub-directories (e.g. 1doc/, 2doc/, 3doc/), each of which
    is a damping sweep directory (i.e. directly holds damping_* sub-dirs with
    metrics.json / config.json / diagnostics.json).

    Layout expected for --mode heatmap:
        <multi_dir>/
          1doc/
            damping_1e-3/{metrics.json, config.json, diagnostics.json}
            damping_1e-2/…
          2doc/…
          Ndoc/…

    This matches the per-variant sub-directory produced by upload_to_hf.py:
        <kronfluence_root>/base/<N>doc/<damping_X>/

Data sources (per damping_* sub-directory)
------------------------------------------
  config.json      – damping_factor, use_heuristic_damping
  metrics.json     – recall_at_k[k].overall_average
  diagnostics.json – lambda.global percentiles, heuristic_damping_equiv

Usage
-----
    python ekfac_damping_chart.py [sweep_dir] [options]
    python ekfac_damping_chart.py <multi_dir> --mode heatmap [options]

Examples
--------
    python ekfac_damping_chart.py filter/kronfluence_results/3doc/damping_sweep_ekfac_20260416T211905Z
    python ekfac_damping_chart.py <dir> --metric recall_at_1 --output chart.png
    python ekfac_damping_chart.py filter/kronfluence_results/final-v2/base --mode heatmap
    python ekfac_damping_chart.py filter/kronfluence_results/final-v2/base --mode heatmap --metric recall_5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_C_R1  = "#4c84b8"
_C_R5  = "#e07b39"
_C_R10 = "#59a65f"
_C_MRR = "#c94e4e"
_C_MAP = "#c97e1a"
_C_HEUR = "#8b6bb3"
_C_CDF  = "#555555"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_damping_value(dir_name: str) -> Optional[float]:
    """Return the numeric damping value from a directory name like damping_1e-3."""
    m = re.fullmatch(r"damping_(.+)", dir_name)
    if not m:
        return None
    val = m.group(1)
    if val.lower() == "none":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _compute_mrr(recall_at_k: dict) -> float:
    k_vals = sorted(int(k) for k in recall_at_k)
    mrr, prev = 0.0, 0.0
    for k in k_vals:
        r = recall_at_k[str(k)].get("overall_average", 0.0)
        mrr += max(0.0, r - prev) / k
        prev = r
    return mrr


def _compute_map(run_dir: Path) -> Optional[float]:
    """Compute mAP from ``per_query.jsonl`` + ``ranked.jsonl`` in *run_dir*.

    GT documents for a query are defined as training docs whose ``func``
    matches the query's ``func`` and whose ``role`` is ``"constant"``, which
    matches the convention used by ``kronfluence_ranker.py``.

    Returns ``None`` if either file is absent.
    """
    pq_path  = run_dir / "per_query.jsonl"
    rk_path  = run_dir / "ranked.jsonl"
    if not (pq_path.exists() and rk_path.exists()):
        return None

    try:
        train_meta: dict = {}
        with rk_path.open() as fh:
            for line in fh:
                d = json.loads(line)
                train_meta[d["uid"]] = d

        aps: List[float] = []
        with pq_path.open() as fh:
            for line in fh:
                q = json.loads(line)
                gt_set = {
                    uid for uid in q.get("train_uids", [])
                    if train_meta.get(uid, {}).get("func") == q.get("func")
                    and train_meta.get(uid, {}).get("role") == "constant"
                }
                if not gt_set:
                    continue
                uid_score = sorted(
                    zip(q["train_uids"], q["scores"]),
                    key=lambda x: -x[1],
                )
                hits, ap = 0, 0.0
                for rank, (uid, _) in enumerate(uid_score, 1):
                    if uid in gt_set:
                        hits += 1
                        ap += hits / rank
                aps.append(ap / len(gt_set))

        return sum(aps) / len(aps) if aps else 0.0
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def load_sweep(base: Path) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Returns:
        rows       – one dict per damping sub-directory (sorted by damping)
        heur_row   – the damping_none row, if present (None otherwise)
    """
    rows: List[Dict] = []
    heur_row: Optional[Dict] = None

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_f = subdir / "metrics.json"
        config_f  = subdir / "config.json"
        diag_f    = subdir / "diagnostics.json"
        if not (metrics_f.exists() and diag_f.exists()):
            continue

        try:
            metrics = json.loads(metrics_f.read_text())
            diag    = json.loads(diag_f.read_text())
            config  = json.loads(config_f.read_text()) if config_f.exists() else {}
        except (json.JSONDecodeError, OSError):
            continue

        rk = metrics.get("recall_at_k", {})
        if not rk:
            continue

        def _r(k: int) -> Optional[float]:
            return rk.get(str(k), {}).get("overall_average")

        row: Dict = {
            "dir":      subdir.name,
            "recall_1":  _r(1),
            "recall_5":  _r(5),
            "recall_10": _r(10),
            "mrr":       _compute_mrr(rk),
            "map":       _compute_map(subdir),
            "damping":   config.get("damping_factor"),
            "heuristic": bool(config.get("use_heuristic_damping", False)),
            "lambda_global": diag.get("lambda", {}).get("global", {}),
        }

        # Parse numeric damping from directory name as fallback
        if row["damping"] is None and not row["heuristic"]:
            row["damping"] = _parse_damping_value(subdir.name)

        if row["heuristic"]:
            heur_row = row
        elif row["damping"] is not None:
            rows.append(row)

    rows.sort(key=lambda r: r["damping"])
    return rows, heur_row


# ---------------------------------------------------------------------------
# Multi-N (heatmap) data loading
# ---------------------------------------------------------------------------

_NDOC_RE = re.compile(r"^(\d+)doc$")


def _ndoc_dirs(base: Path) -> List[Tuple[int, Path]]:
    """Return ``(N, path)`` pairs for every ``<N>doc`` directory under *base*."""
    found = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        m = _NDOC_RE.fullmatch(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return sorted(found)


def load_multi_sweep(base: Path) -> Dict[int, Tuple[List[Dict], Optional[Dict]]]:
    """Load damping sweeps for every ``<N>doc`` sub-directory under *base*.

    Returns a dict mapping N → (rows, heur_row) as returned by ``load_sweep``.
    """
    result: Dict[int, Tuple[List[Dict], Optional[Dict]]] = {}
    for n, ndoc_dir in _ndoc_dirs(base):
        rows, heur_row = load_sweep(ndoc_dir)
        if rows:
            result[n] = (rows, heur_row)
    return result


# ---------------------------------------------------------------------------
# Heatmap plot
# ---------------------------------------------------------------------------

def plot_ndoc_heatmap(
    multi: Dict[int, Tuple[List[Dict], Optional[Dict]]],
    metric: str = "recall_1",
    title_prefix: str = "",
    output: Optional[Path] = None,
    annot_fmt: str = ".1f",
) -> None:
    """Plot a heatmap: rows = N-doc setting, columns = damping factor.

    Parameters
    ----------
    multi:
        Output of ``load_multi_sweep``.
    metric:
        One of ``recall_1``, ``recall_5``, ``recall_10``, ``mrr``.
    title_prefix:
        Optional string prepended to the figure title.
    output:
        File path to save the figure.  If None, the figure is shown
        interactively.
    annot_fmt:
        ``format`` string used for cell annotations (default ``".1f"``).
    """
    if not multi:
        raise ValueError("No N-doc data found.")

    # ── Collect all damping values across every N ─────────────────────────
    all_dampings: List[float] = sorted(
        {r["damping"] for rows, _ in multi.values() for r in rows}
    )
    all_ns: List[int] = sorted(multi.keys())

    # Build matrix (rows = N, cols = damping)
    mat = np.full((len(all_ns), len(all_dampings)), np.nan)
    for ri, n in enumerate(all_ns):
        rows, _ = multi[n]
        d_to_val = {r["damping"]: (r.get(metric) or 0.0) for r in rows}
        for ci, d in enumerate(all_dampings):
            if d in d_to_val:
                mat[ri, ci] = d_to_val[d] * 100  # convert to %

    metric_label = _metric_label(metric)

    # ── Figure ─────────────────────────────────────────────────────────────
    col_w = max(0.7, 7.0 / max(1, len(all_dampings)))
    row_h = max(0.55, 4.5 / max(1, len(all_ns)))
    fig_w = max(8, len(all_dampings) * col_w + 3.0)
    fig_h = max(3.5, len(all_ns) * row_h + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Annotate cells
    for ri in range(len(all_ns)):
        for ci in range(len(all_dampings)):
            val = mat[ri, ci]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin + 1e-12)
            text_color = "black" if 0.25 < norm_val < 0.75 else (
                "white" if norm_val <= 0.25 else "black"
            )
            ax.text(
                ci, ri,
                f"{val:{annot_fmt}}",
                ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    # ── Axes labels ────────────────────────────────────────────────────────
    ax.set_xticks(range(len(all_dampings)))
    ax.set_xticklabels(
        [_fmt_damping(d) for d in all_dampings],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(all_ns)))
    ax.set_yticklabels([f"{n} doc" for n in all_ns], fontsize=9)
    ax.set_xlabel("Damping factor", fontsize=10)
    ax.set_ylabel("N-doc setting", fontsize=10)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(f"{metric_label} (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Mark per-row best with a star outline
    for ri, n in enumerate(all_ns):
        row_vals = mat[ri, :]
        if np.all(np.isnan(row_vals)):
            continue
        best_ci = int(np.nanargmax(row_vals))
        ax.add_patch(
            plt.Rectangle(
                (best_ci - 0.5, ri - 0.5), 1, 1,
                fill=False, edgecolor="#222222", linewidth=1.8, zorder=5,
            )
        )

    title = f"EKFAC Damping Heatmap — {metric_label}"
    if title_prefix:
        title = f"{title} — {title_prefix}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CDF reconstruction from percentile summary
# ---------------------------------------------------------------------------

def build_cdf(lambda_global: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct an approximate CDF from stored percentiles.
    Returns (values, cum_prob) arrays suitable for plotting.
    """
    pts = [
        (lambda_global.get("min"),    0.00),
        (lambda_global.get("p1"),     0.01),
        (lambda_global.get("p5"),     0.05),
        (lambda_global.get("p25"),    0.25),
        (lambda_global.get("median"), 0.50),
        (lambda_global.get("p75"),    0.75),
        (lambda_global.get("p95"),    0.95),
        (lambda_global.get("p99"),    0.99),
        (lambda_global.get("max"),    1.00),
    ]
    pts = [(v, p) for v, p in pts if v is not None and v > 0]
    if len(pts) < 2:
        return np.array([]), np.array([])
    vals, probs = zip(*pts)
    return np.array(vals, dtype=float), np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fmt_damping(v: float) -> str:
    if v == 0:
        return "0"
    exp = math.floor(math.log10(v))
    mantissa = v / 10 ** exp
    if abs(mantissa - round(mantissa)) < 1e-9:
        m = int(round(mantissa))
        return f"{m}e{exp}" if m != 1 else f"1e{exp}"
    return f"{mantissa:.2g}e{exp}"


def _metric_label(key: str) -> str:
    return {"recall_1": "Recall@1", "recall_5": "Recall@5",
            "recall_10": "Recall@10", "mrr": "MRR", "map": "mAP"}[key]


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

_ALL_METRICS = ["recall_1", "recall_5", "recall_10", "mrr", "map"]


def plot_damping_chart(
    rows: List[Dict],
    heur_row: Optional[Dict],
    title_prefix: str = "",
    highlight: str = "recall_1",
    panel: str = "both",           # "both" | "performance" | "spectrum"
    show_metrics: Optional[List[str]] = None,  # subset of _ALL_METRICS; None = all
    output: Optional[Path] = None,
) -> None:
    if not rows:
        raise ValueError("No numeric-damping rows found.")

    show_perf     = panel in ("both", "performance")
    show_spectrum = panel in ("both", "spectrum")

    # Which metrics to draw in the performance panel
    active_metrics: List[str] = show_metrics if show_metrics else _ALL_METRICS
    # Ensure highlight is in the active set; fall back to first active metric
    if highlight not in active_metrics:
        highlight = active_metrics[0]

    dampings = np.array([r["damping"] for r in rows], dtype=float)

    metrics = {
        "recall_1":  np.array([r["recall_1"]  or 0.0 for r in rows]),
        "recall_5":  np.array([r["recall_5"]  or 0.0 for r in rows]),
        "recall_10": np.array([r["recall_10"] or 0.0 for r in rows]),
        "mrr":       np.array([r["mrr"]       for r in rows]),
        "map":       np.array([r["map"]        or 0.0 for r in rows]),
    }
    colors = {"recall_1": _C_R1, "recall_5": _C_R5,
              "recall_10": _C_R10, "mrr": _C_MRR, "map": _C_MAP}

    # Best fixed-damping point (by highlight metric)
    best_idx = int(np.argmax(metrics[highlight]))
    best_d   = dampings[best_idx]
    best_val = metrics[highlight][best_idx]

    # Heuristic equivalent position
    heur_equiv: Optional[float] = None
    lambda_global: dict = {}
    if heur_row is not None:
        lambda_global = heur_row.get("lambda_global", {})
        heur_equiv = lambda_global.get("heuristic_damping_equiv")
    elif rows:
        lambda_global = rows[0].get("lambda_global", {})
        heur_equiv = lambda_global.get("heuristic_damping_equiv")

    # ── layout ────────────────────────────────────────────────────────────
    if show_perf and show_spectrum:
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 8),
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.38},
        )
        ax_top, ax_bot = axes
    elif show_perf:
        fig, ax_top = plt.subplots(figsize=(9, 5))
        ax_bot = None
    else:
        fig, ax_bot = plt.subplots(figsize=(9, 4))
        ax_top = None

    fig.patch.set_facecolor("#f8f8f8")
    for ax in (ax_top, ax_bot):
        if ax is not None:
            ax.set_facecolor("#f8f8f8")

    # ── TOP: performance vs damping ───────────────────────────────────────
    if show_perf:
        for key in active_metrics:
            color = colors[key]
            lw = 2.2 if key == highlight else 1.4
            alpha = 1.0 if key == highlight else 0.75
            ax_top.plot(dampings, metrics[key] * 100, "o-",
                        color=color, lw=lw, ms=5, alpha=alpha,
                        label=_metric_label(key))

        # Best marker
        ax_top.plot(best_d, best_val * 100, "*", ms=14,
                    color=colors[highlight], zorder=5,
                    label=f"Best {_metric_label(highlight)} ({_fmt_damping(best_d)}, {best_val*100:.1f}%)")

        # Heuristic vertical line + point
        if heur_equiv is not None:
            ax_top.axvline(heur_equiv, color=_C_HEUR, lw=1.4,
                           linestyle="--", alpha=0.8, zorder=3)
            ax_top.text(heur_equiv, ax_top.get_ylim()[0] if ax_top.get_ylim()[0] > 0 else 0,
                        f"  heuristic\n  ({_fmt_damping(heur_equiv)})",
                        color=_C_HEUR, fontsize=7.5, va="bottom", ha="left")

        # Heuristic run point (only for active metrics)
        if heur_row is not None and heur_equiv is not None:
            for key in active_metrics:
                val = (heur_row.get(key) or 0.0) * 100
                ax_top.plot(heur_equiv, val, "D", ms=7, color=colors[key],
                            zorder=6, markeredgecolor="white", markeredgewidth=0.6)

        ax_top.set_xscale("log")
        ax_top.set_xlabel("Damping factor (log scale)", fontsize=9)
        ax_top.set_ylabel("Score (%)", fontsize=9)
        ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_top.tick_params(labelsize=8)
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
        ax_top.legend(fontsize=7.5, framealpha=0.85, loc="upper left")
        ax_top.set_title(
            "EKFAC Performance vs. Damping Factor",
            fontsize=11, fontweight="bold", pad=7,
        )

        # Annotation: heuristic run values (only active metrics)
        if heur_row is not None:
            parts = [f"{_metric_label(k)}={heur_row.get(k)*100:.1f}%"
                     for k in active_metrics if heur_row.get(k) is not None]
            ax_top.text(0.99, 0.04, "◆ heuristic:  " + "  ".join(parts),
                        transform=ax_top.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color=_C_HEUR,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec=_C_HEUR))

    # ── BOTTOM: eigenvalue CDF ─────────────────────────────────────────────
    if show_spectrum:
        cdf_vals, cdf_probs = build_cdf(lambda_global)
        if len(cdf_vals) >= 2:
            ax_bot.plot(cdf_vals, cdf_probs * 100, "-", color=_C_CDF,
                        lw=2, label="λ CDF (global)")
            ax_bot.fill_between(cdf_vals, cdf_probs * 100, alpha=0.08, color=_C_CDF)

            # Vertical lines for each tested damping value
            for r in rows:
                d = r["damping"]
                ax_bot.axvline(d, color=colors["recall_1"], lw=0.9,
                               linestyle=":", alpha=0.7)

            # Heuristic line
            if heur_equiv is not None:
                ax_bot.axvline(heur_equiv, color=_C_HEUR, lw=1.6,
                               linestyle="--", alpha=0.85, label="heuristic equiv.")

            # Annotate percentile reference points
            ref_pts = {
                "p25": (lambda_global.get("p25"), 25),
                "median": (lambda_global.get("median"), 50),
                "p75": (lambda_global.get("p75"), 75),
                "p95": (lambda_global.get("p95"), 95),
            }
            for label, (xv, yv) in ref_pts.items():
                if xv and xv > 0:
                    ax_bot.annotate(
                        label, xy=(xv, yv), xytext=(xv * 3, yv - 8),
                        fontsize=6.5, color="#666666",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
                    )

        ax_bot.set_xscale("log")
        ax_bot.set_xlabel("Eigenvalue λ  (log scale)", fontsize=9)
        ax_bot.set_ylabel("Cumulative % of weights", fontsize=9)
        ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_bot.tick_params(labelsize=8)
        ax_bot.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
        ax_bot.legend(fontsize=7.5, framealpha=0.85)
        ax_bot.set_title(
            "Eigenvalue Spectrum  (dotted verticals = tested damping values)",
            fontsize=9, pad=5,
        )

    # ── suptitle ──────────────────────────────────────────────────────────
    suptitle = "EKFAC Damping Sweep"
    if title_prefix:
        suptitle = f"{suptitle} — {title_prefix}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.99)

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot EKFAC performance vs. damping factor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        default="filter/kronfluence_results/3doc/damping_sweep_ekfac_20260416T211905Z",
        help=(
            "For --mode sweep: directory containing damping_* sub-directories.  "
            "For --mode heatmap: directory containing <N>doc/ sub-directories, "
            "each of which holds damping_* sub-dirs."
        ),
    )
    parser.add_argument(
        "--mode",
        default="sweep",
        choices=["sweep", "heatmap"],
        help=(
            "'sweep' (default): single-sweep line chart + eigenvalue CDF.  "
            "'heatmap': N-doc × damping heatmap for the chosen metric."
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=["recall_1", "recall_5", "recall_10", "mrr", "map"],
        help=(
            "Primary metric used to star the best point (sweep mode) or "
            "colour the heatmap cells (heatmap mode).  "
            "Defaults to the first metric listed in --metrics (sweep) or recall_1 (heatmap)."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["recall_1", "recall_5", "recall_10", "mrr", "map"],
        metavar="METRIC",
        help=(
            "Which metrics to plot in sweep mode (space-separated).  "
            "E.g. --metrics map  or  --metrics recall_1 map.  "
            "Default: all five.  Ignored in heatmap mode."
        ),
    )
    parser.add_argument(
        "--panel",
        default="both",
        choices=["both", "performance", "spectrum"],
        help=(
            "Which panel(s) to include in sweep mode.  "
            "'performance' = metrics vs damping only;  "
            "'spectrum' = eigenvalue CDF only;  "
            "'both' = stacked (default).  Ignored in heatmap mode."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output image path.  "
            "Defaults to <sweep_dir>/ekfac_damping_chart.png (sweep) or "
            "<sweep_dir>/ekfac_damping_heatmap_<metric>.png (heatmap)."
        ),
    )
    args = parser.parse_args()

    base = Path(args.sweep_dir)
    if not base.exists():
        parser.error(f"Directory not found: {base}")

    # ── heatmap mode ──────────────────────────────────────────────────────
    if args.mode == "heatmap":
        metric = args.metric or "recall_1"
        multi = load_multi_sweep(base)
        if not multi:
            parser.error(
                f"No <N>doc/ sub-directories with damping sweeps found in {base}"
            )
        output = (
            Path(args.output)
            if args.output
            else base / f"ekfac_damping_heatmap_{metric}.png"
        )
        plot_ndoc_heatmap(
            multi,
            metric=metric,
            title_prefix=base.name,
            output=output,
        )
        return

    # ── sweep mode (default) ──────────────────────────────────────────────
    rows, heur_row = load_sweep(base)
    if not rows:
        parser.error(f"No damping sub-directories with metrics found in {base}")

    show_metrics = args.metrics  # None means all
    # Determine highlight: explicit --metric, else first in --metrics, else recall_1
    highlight = args.metric
    if highlight is None:
        highlight = show_metrics[0] if show_metrics else "recall_1"

    output = Path(args.output) if args.output else base / "ekfac_damping_chart.png"
    plot_damping_chart(rows, heur_row,
                       title_prefix=base.name,
                       highlight=highlight,
                       panel=args.panel,
                       show_metrics=show_metrics,
                       output=output)


if __name__ == "__main__":
    main()
