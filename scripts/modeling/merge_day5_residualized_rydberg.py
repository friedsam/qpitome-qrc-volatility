#!/usr/bin/env python3
"""Merge residualized-Rydberg shards and score matched deltas versus frozen D1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from qpitome_qrc.baselines.logistic_offset import clip_prob

META = {
    "row_id", "market_key", "episode_id", "cluster_id", "landmark_date",
    "cluster_start", "y", "shard_index",
}


def metrics(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = clip_prob(use[model].to_numpy(float))
    return {
        "model": model,
        "n": int(len(use)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def deltas(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    use = group[["y", "D1", model, "cluster_id"]].dropna().copy()
    y = use["y"].to_numpy(int)
    p0 = clip_prob(use["D1"].to_numpy(float))
    p1 = clip_prob(use[model].to_numpy(float))
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2
    use["dll"] = ll1 - ll0
    use["dbr"] = br1 - br0
    by_cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": model,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "delta_logloss": float(np.mean(ll1 - ll0)),
        "delta_brier": float(np.mean(br1 - br0)),
        "cluster_mean_delta_logloss": float(by_cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(by_cluster["dbr"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    prediction_files: list[Path] = []
    diagnostic_files: list[Path] = []
    for item in args.inputs:
        if item.is_dir():
            prediction_files.extend(item.glob("predictions_shard_*.csv"))
            diagnostic_files.extend(item.glob("residual_diagnostics_shard_*.csv"))
        elif item.name.startswith("predictions_shard_"):
            prediction_files.append(item)
    prediction_files = sorted(set(prediction_files))
    diagnostic_files = sorted(set(diagnostic_files))
    if len(prediction_files) != args.num_shards:
        raise ValueError(f"Expected {args.num_shards} prediction shards, found {len(prediction_files)}")

    predictions = pd.concat([pd.read_csv(path, parse_dates=["landmark_date", "cluster_start"]) for path in prediction_files], ignore_index=True)
    predictions = predictions.sort_values("row_id").reset_index(drop=True)
    if predictions["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values across shards")
    if len(predictions) != 299:
        raise ValueError(f"Expected 299 predictions, found {len(predictions)}")

    models = [column for column in predictions.columns if column not in META]
    groups = {
        "all_post1990_purged": predictions,
        "validation_non_spy": predictions[predictions["market_key"] != "spy"],
        "leave_nikkei_out_eval": predictions[predictions["market_key"] != "nikkei_225"],
        "nikkei_only_eval": predictions[predictions["market_key"] == "nikkei_225"],
    }
    summary_rows, delta_rows = [], []
    for group_name, group in groups.items():
        for model in models:
            row = metrics(group, model)
            row["group"] = group_name
            summary_rows.append(row)
            if model != "D1":
                row = deltas(group, model)
                row["group"] = group_name
                delta_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    delta = pd.DataFrame(delta_rows)
    ranking = delta[delta["group"] == "all_post1990_purged"].sort_values(["delta_logloss", "delta_brier"])
    diagnostics = pd.concat([pd.read_csv(path, parse_dates=["cluster_start"]) for path in diagnostic_files], ignore_index=True) if diagnostic_files else pd.DataFrame()

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    summary.to_csv(args.outdir / "summary_metrics.csv", index=False)
    delta.to_csv(args.outdir / "paired_score_deltas.csv", index=False)
    ranking.to_csv(args.outdir / "all_market_ranking.csv", index=False)
    if not diagnostics.empty:
        diagnostics.to_csv(args.outdir / "residual_diagnostics.csv", index=False)
        diagnostics.groupby(["block", "residualizer", "ridge_alpha"], as_index=False).agg({
            "fraction_output_variance_removed": "median",
            "residual_rms": "median",
        }).to_csv(args.outdir / "residual_diagnostics_summary.csv", index=False)

    (args.outdir / "merge_manifest.json").write_text(json.dumps({
        "prediction_shards": [str(path) for path in prediction_files],
        "n_predictions": int(len(predictions)),
        "n_models": int(len(models)),
    }, indent=2))

    print("\nResidualized Rydberg models ranked by all-market log-loss delta (negative = better)")
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved merged outputs to {args.outdir}")


if __name__ == "__main__":
    main()
