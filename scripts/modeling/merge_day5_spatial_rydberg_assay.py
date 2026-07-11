#!/usr/bin/env python3
"""Merge deterministic assay shards and compute standardized metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from qpitome_qrc.baselines.logistic_offset import clip_prob


def score(group: pd.DataFrame, model: str) -> dict:
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = clip_prob(use[model].to_numpy(float))
    return {
        "model": model,
        "n": int(len(use)),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


def paired_delta(group: pd.DataFrame, model: str) -> dict:
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
    cluster = use.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "model": model,
        "n": int(len(use)),
        "n_clusters": int(use["cluster_id"].nunique()),
        "delta_logloss": float((ll1 - ll0).mean()),
        "delta_brier": float((br1 - br0).mean()),
        "cluster_mean_delta_logloss": float(cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(cluster["dbr"].mean()),
    }


def cluster_metrics(group: pd.DataFrame, model: str) -> dict:
    rows = []
    for cluster_id, subset in group.dropna(subset=[model]).groupby("cluster_id"):
        y = subset["y"].to_numpy(int)
        p = clip_prob(subset[model].to_numpy(float))
        rows.append({
            "cluster_id": cluster_id,
            "n": len(subset),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(np.mean((p - y) ** 2)),
        })
    detail = pd.DataFrame(rows)
    return {
        "model": model,
        "n_clusters": int(len(detail)),
        "cluster_mean_logloss": float(detail["logloss"].mean()),
        "cluster_mean_brier": float(detail["brier"].mean()),
        "median_cluster_size": float(detail["n"].median()),
    }


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
    parser.add_argument("--outdir", type=Path, required=True)
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
