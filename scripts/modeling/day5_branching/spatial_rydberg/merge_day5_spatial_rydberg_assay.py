#!/usr/bin/env python3
"""Merge deterministic assay shards and compute standardized metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.evaluation.scoring import (
    binary_summary,
    cluster_weighted_summary,
    proper_score_deltas,
)


def score(group: pd.DataFrame, model: str) -> dict:
    """Return the historical assay summary schema via shared scoring."""

    return binary_summary(group, model, clip=1e-6)


def paired_delta(group: pd.DataFrame, model: str) -> dict:
    """Return matched D1 deltas via shared proper-score scoring."""

    return proper_score_deltas(group, model, baseline="D1", clip=1e-6)


def cluster_metrics(group: pd.DataFrame, model: str) -> dict:
    """Return the historical cluster-weighted schema via shared scoring."""

    return cluster_weighted_summary(group, model, clip=1e-6)


def collect_files(inputs: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.glob(pattern)))
        elif path.match(pattern):
            files.append(path)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=Path("results/modeling/day5_branching/spatial_rydberg/day5_spatial_rydberg_assay"))
    args = parser.parse_args()

    prediction_files = collect_files(args.inputs, "predictions_shard_*.csv")
    diagnostic_files = collect_files(args.inputs, "feature_diagnostics_shard_*.csv")
    if len(prediction_files) != args.num_shards:
        raise ValueError(f"Expected {args.num_shards} prediction shards, found {len(prediction_files)}")

    predictions = pd.concat([pd.read_csv(path, parse_dates=["landmark_date"]) for path in prediction_files], ignore_index=True)
    predictions = predictions.sort_values("row_id").reset_index(drop=True)
    if predictions["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values across shards")
    if len(predictions) != 299:
        raise ValueError(f"Expected 299 merged predictions, got {len(predictions)}")

    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_files], ignore_index=True) if diagnostic_files else pd.DataFrame()
    model_columns = [column for column in predictions.columns if column not in {
        "row_id", "market_key", "episode_id", "cluster_id", "landmark_date", "y", "shard_index"
    }]
    groups = {
        "all_post1990_purged": predictions,
        "validation_non_spy": predictions[predictions["market_key"] != "spy"],
        "leave_nikkei_out_eval": predictions[predictions["market_key"] != "nikkei_225"],
        "nikkei_only_eval": predictions[predictions["market_key"] == "nikkei_225"],
    }

    summary_rows, paired_rows, cluster_rows = [], [], []
    for group_name, group in groups.items():
        for model in model_columns:
            row = score(group, model)
            row["group"] = group_name
            summary_rows.append(row)
            crow = cluster_metrics(group, model)
            crow["group"] = group_name
            cluster_rows.append(crow)
            if model != "D1":
                prow = paired_delta(group, model)
                prow["group"] = group_name
                paired_rows.append(prow)

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    cluster = pd.DataFrame(cluster_rows)
    ranking = paired[paired["group"] == "all_post1990_purged"].sort_values(["delta_logloss", "delta_brier"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    paired.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    cluster.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)
    ranking.to_csv(args.outdir / "all_market_model_ranking.csv", index=False)
    if not diagnostics.empty:
        diagnostics.to_csv(args.outdir / "feature_diagnostics.csv", index=False)
        diagnostics.groupby("block", as_index=False).agg({
            "participation_ratio": "median",
            "n95": "median",
            "condition_number": "median",
            "near_constant": "median",
        }).to_csv(args.outdir / "feature_diagnostics_summary.csv", index=False)

    manifest = {
        "prediction_shards": [str(path) for path in prediction_files],
        "diagnostic_shards": [str(path) for path in diagnostic_files],
        "n_predictions": int(len(predictions)),
        "n_models": int(len(model_columns)),
    }
    (args.outdir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nTop 20 models by matched all-market log-loss delta (negative = better)")
    print(ranking.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMerged outputs saved to {args.outdir}")


if __name__ == "__main__":
    main()
