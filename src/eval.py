"""
Evaluation harness for retrieval accuracy.
Owner: Pham Huu Huy

This is Part 2 of docs/IMPLEMENTATION_PLAN.md — the thing that turns
Part 4's search() output into a number the team can actually optimize
against. There was no existing stub for this; it's a new module.

Ground-truth schema (docs/IMPLEMENTATION_PLAN.md §2.5) — NOT the same file
as data/raw/queries/queries.json (that one trains the query classifier,
see src/data_loader.py):

  [{"query_id": str, "query_text": str, "query_type": "KIS"|"AVS"|"VQA"|"KISC",
    "video_id": str, "timestamp_sec": float, "tolerance_sec": float}, ...]

A hit requires matching video_id AND |returned.timestamp_sec -
ground_truth.timestamp_sec| <= tolerance_sec.
"""

import argparse
import json
from pathlib import Path

import yaml

from .inference import search


def _is_hit(result: dict, ground_truth: dict) -> bool:
    return (
        result["video_id"] == ground_truth["video_id"]
        and abs(result["timestamp_sec"] - ground_truth["timestamp_sec"]) <= ground_truth["tolerance_sec"]
    )


def evaluate(
    ground_truth_path: str | Path, config: dict, k_values: tuple[int, ...] = (1, 5, 10)
) -> dict:
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not ground_truth:
        raise ValueError(f"{ground_truth_path} contains no queries")

    max_k = max(k_values)
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for gt in ground_truth:
        results = search(gt["query_text"], config, top_k=max_k)
        rank = next((i for i, r in enumerate(results, 1) if _is_hit(r, gt)), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in k_values:
            if rank is not None and rank <= k:
                recall_hits[k] += 1

    n = len(ground_truth)
    metrics = {f"recall@{k}": recall_hits[k] / n for k in k_values}
    metrics["mrr"] = sum(reciprocal_ranks) / n
    metrics["n_queries"] = n
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/raw/queries/eval_ground_truth.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate(args.ground_truth, cfg)
    print("Evaluation results:")
    for name, value in metrics.items():
        line = f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}"
        print(line)
