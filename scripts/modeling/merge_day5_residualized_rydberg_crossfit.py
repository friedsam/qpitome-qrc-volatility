#!/usr/bin/env python3
"""Merge frozen cross-fitted residualized-Rydberg shards and score versus D1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from qpitome_qrc.baselines.logistic_offset import clip_prob


def score(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    y = group["y"].to_numpy(int)
    p = clip_prob(group[model].to_numpy(float))
    return {
        "model": model,
        "n": int(len(group)),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def paired(group: pd.DataFrame) -> dict[str, float | int]:
    y = group["y"].to_numpy(int)
    p0 = clip_prob(group["D1"].to_numpy(float))
    p1 = clip_prob(group["crossfit_resid_occupations"].to_numpy(float))
    ll0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    br0 = (p0 - y) ** 2
    br1 = (p1 - y) ** 2
    temp = group[["cluster_id"]].copy()
    temp["dll"] = ll1 - ll0
    temp["dbr"] = br1 - br0
    cluster = temp.groupby("cluster_id")[["dll", "dbr"]].mean()
    return {
        "n": int(len(group)),
        "n_clusters": int(group["cluster_id"].nunique()),
        "delta_logloss": float(np.mean(ll1 - ll0)),
        "delta_brier": float(np.mean(br1 - br0)),
        "cluster_mean_delta_logloss": float(cluster["dll"].mean()),
        "cluster_mean_delta_brier": float(cluster["dbr"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, required=True)
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
