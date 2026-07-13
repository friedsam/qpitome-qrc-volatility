#!/usr/bin/env python3
"""Merge residualized-Rydberg shards and score matched deltas versus frozen D1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.evaluation.scoring import binary_summary, proper_score_deltas

META = {
    "row_id", "market_key", "episode_id", "cluster_id", "landmark_date",
    "cluster_start", "y", "shard_index",
}


def metrics(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    """Return the historical merge summary schema via shared binary scoring."""

    summary = binary_summary(group, model, clip=1e-6)
    return {
        "model": summary["model"],
        "n": summary["n"],
        "auc": summary["auc"],
        "pr_auc": summary["pr_auc"],
        "logloss": summary["logloss"],
        "brier": summary["brier"],
    }


def deltas(group: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    """Return matched D1 deltas via the shared proper-score implementation."""

    return proper_score_deltas(group, model, baseline="D1", clip=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("results/modeling/day5_branching/residual_confirmation/day5_residualized_rydberg"))
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
