#!/usr/bin/env python3
"""Recompute eval metrics/summary from a saved per_query.jsonl.

Useful when a rolling LOO run finished scoring (and wrote per_query.jsonl /
rolling_ranked.jsonl) but crashed before writing metrics.json / summary.jsonl.
This avoids re-running training: it rebuilds the score matrix and query metadata
directly from the per-query scores and recomputes recall / precision /
composition with the same helpers used by the main pipeline.

Example:
    python3 recompute_eval_from_per_query.py \
        --per-query-path filter/loo_results/0/1doc/final/per_query.jsonl \
        --dataset-path dataset-generator/datasets/0/100/1.jsonl \
        --eval-metrics-path filter/loo_results/0/1doc/final/metrics.json \
        --eval-summary-jsonl filter/loo_results/0/1doc/final/summary.jsonl \
        --eval-topk-range 1,100
"""
import argparse
import json
import os
import sys

import torch


def _import_ranker_modules():
    filter_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "filter")
    )
    if filter_dir not in sys.path:
        sys.path.insert(0, filter_dir)
    from kronfluence_ranker import (
        _compute_recall_precision_at_k,
        _compute_composition_per_function,
        _parse_eval_topk_list,
        DISTRACTOR_FUNCS,
        allowed_role_for_token,
    )
    return dict(
        recall_fn=_compute_recall_precision_at_k,
        comp_fn=_compute_composition_per_function,
        parse_k_fn=_parse_eval_topk_list,
        DISTRACTOR_FUNCS=DISTRACTOR_FUNCS,
        allowed_role_for_token=allowed_role_for_token,
    )


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-query-path", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--eval-metrics-path", required=True)
    ap.add_argument("--eval-summary-jsonl", required=True)
    ap.add_argument("--eval-topk-range", default="1,100")
    args = ap.parse_args()

    mods = _import_ranker_modules()
    recall_fn = mods["recall_fn"]
    comp_fn = mods["comp_fn"]
    parse_k_fn = mods["parse_k_fn"]
    DISTRACTOR_FUNCS = mods["DISTRACTOR_FUNCS"]
    allowed_role_for_token = mods["allowed_role_for_token"]

    # ---- Load per-query scores -------------------------------------------
    per_query = _load_jsonl(args.per_query_path)
    if not per_query:
        raise SystemExit(f"No rows in {args.per_query_path}")

    train_uids = per_query[0]["train_uids"]
    num_train = len(train_uids)
    num_queries = len(per_query)

    score_matrix = torch.zeros(num_queries, num_train, dtype=torch.float32)
    query_meta = []
    for qi, q in enumerate(per_query):
        score_matrix[qi] = torch.tensor(q["scores"], dtype=torch.float32)
        query_meta.append({
            "uid": q.get("query_uid"),
            "func": q.get("func"),
            "correct": q.get("correct"),
        })

    # ---- Load dataset and order it to match the score-matrix columns -----
    dataset = _load_jsonl(args.dataset_path)
    uid_to_doc = {str(d.get("uid", i)): d for i, d in enumerate(dataset)}
    missing = [u for u in train_uids if u not in uid_to_doc]
    if missing:
        raise SystemExit(
            f"{len(missing)} train uid(s) from per_query not found in dataset "
            f"(e.g. {missing[:3]}). Wrong --dataset-path?"
        )
    all_records = [uid_to_doc[u] for u in train_uids]

    # ---- Relevance / query groupings (mirrors loo.py) --------------------
    def _is_relevant(doc, func):
        if str(doc.get("func", "")) != func:
            return False
        role = str(doc.get("role", "")).lower()
        if not role:
            return True
        expected = allowed_role_for_token(func)
        return (expected is not None) and (role == expected)

    func_to_relevant = {}
    for ti, doc in enumerate(all_records):
        f = str(doc.get("func", ""))
        if f in DISTRACTOR_FUNCS:
            func_to_relevant.setdefault(f, []).append(ti)
        elif _is_relevant(doc, f):
            func_to_relevant.setdefault(f, []).append(ti)

    func_to_queries = {}
    for qi, qm in enumerate(query_meta):
        if not bool(qm.get("correct", False)):
            continue
        f = str(qm.get("func", ""))
        func_to_queries.setdefault(f, []).append(qi)

    eval_k_list = parse_k_fn(None, None, args.eval_topk_range)

    metrics = {
        "recall_at_k": {},
        "precision_at_k": {},
        "composition_at_k": {},
    }

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
            overall = float(sum(per_func_recalls.values()) / len(per_func_recalls))
            _nq = sum(per_func_counts.values())
            pq_avg = (
                sum(per_func_recalls[f] * per_func_counts[f] for f in per_func_recalls)
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
            for cat in ("constant_gt", "identity_gt", "distractor", "other"):
                vals = [v[cat] for v in composition.values() if cat in v]
                if vals:
                    overall_comp[cat] = float(sum(vals) / len(vals))
            metrics["composition_at_k"][str(k)] = {
                "k": k,
                "per_function": composition,
                "overall_average": overall_comp,
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.eval_metrics_path)), exist_ok=True)
    with open(args.eval_metrics_path, "w") as fh:
        json.dump(metrics, fh)
    print(f"Saved eval metrics → {args.eval_metrics_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.eval_summary_jsonl)), exist_ok=True)
    with open(args.eval_summary_jsonl, "w") as fh:
        for k in eval_k_list:
            sk = str(k)
            row = {"k": k}
            if sk in metrics.get("recall_at_k", {}):
                r = metrics["recall_at_k"][sk]
                row["recall_overall_avg"] = r.get("overall_average")
                row["recall_per_query_avg"] = r.get("per_query_average")
                vars_r = r.get("per_function_variance", {})
                if vars_r:
                    row["recall_var_avg"] = float(sum(vars_r.values()) / len(vars_r))
            if sk in metrics.get("precision_at_k", {}):
                p = metrics["precision_at_k"][sk]
                row["precision_overall_avg"] = p.get("overall_average")
            if sk in metrics.get("composition_at_k", {}):
                comp = metrics["composition_at_k"][sk].get("overall_average", {})
                if isinstance(comp, dict):
                    row["composition_constant_gt"] = comp.get("constant_gt")
                    row["composition_identity_gt"] = comp.get("identity_gt")
                    row["composition_distractor"] = comp.get("distractor")
                    row["composition_other"] = comp.get("other")
            fh.write(json.dumps(row) + "\n")
    print(f"Saved eval summary → {args.eval_summary_jsonl}")


if __name__ == "__main__":
    main()
