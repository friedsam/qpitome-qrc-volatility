#!/usr/bin/env python3
"""Merge frozen cross-fitted residualized-Rydberg shards and score versus D1."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qpitome_qrc.evaluation.scoring import binary_summary, proper_score_deltas


def score(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    """Return the historical crossfit summary schema via shared scoring."""

    summary = binary_summary(group, model, clip=1e-6)
    return {
        "model": summary["model"],
        "n": summary["n"],
        "auc": summary["auc"],
        "pr_auc": summary["pr_auc"],
        "logloss": summary["logloss"],
        "brier": summary["brier"],
    }


def paired(group: pd.DataFrame) -> dict[str, float | int]:
    """Return the historical paired-delta schema via shared scoring."""

    result = proper_score_deltas(
        group,
        "crossfit_resid_occupations",
        baseline="D1",
        clip=1e-6,
    )
    return {
        "n": result["n"],
        "n_clusters": result["n_clusters"],
        "delta_logloss": result["delta_logloss"],
        "delta_brier": result["delta_brier"],
        "cluster_mean_delta_logloss": result["cluster_mean_delta_logloss"],
        "cluster_mean_delta_brier": result["cluster_mean_delta_brier"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("results/modeling/day5_branching/residual_confirmation/day5_residualized_rydberg_crossfit"))
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.inputs:
        files.extend(item.glob("predictions_shard_*.csv") if item.is_dir() else [item])
    files = sorted(set(files))
    if len(files) != args.num_shards:
        raise ValueError(f"Expected {args.num_shards} shards, found {len(files)}")
    frame = pd.concat([pd.read_csv(path, parse_dates=["landmark_date", "cluster_start"]) for path in files], ignore_index=True)
    frame = frame.sort_values("row_id").reset_index(drop=True)
    if len(frame) != 299 or frame["row_id"].duplicated().any():
        raise ValueError("Merged predictions must contain exactly 299 unique rows")

    groups = {
        "all_post1990_purged": frame,
        "validation_non_spy": frame[frame["market_key"] != "spy"],
        "leave_nikkei_out_eval": frame[frame["market_key"] != "nikkei_225"],
        "nikkei_only_eval": frame[frame["market_key"] == "nikkei_225"],
    }
    metrics_rows, delta_rows = [], []
    for group_name, group in groups.items():
        for model in ("D1", "crossfit_resid_occupations"):
            row = score(group, model)
            row["group"] = group_name
            metrics_rows.append(row)
        row = paired(group)
        row["group"] = group_name
        delta_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    deltas = pd.DataFrame(delta_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "predictions.csv", index=False)
    metrics.to_csv(args.outdir / "summary_metrics.csv", index=False)
    deltas.to_csv(args.outdir / "paired_score_deltas.csv", index=False)

    print("\nCross-fitted D1 offset confirmation")
    print(deltas.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTraining-row counts used by correction")
    print(frame["n_correction_train"].describe().to_string())


if __name__ == "__main__":
    main()
