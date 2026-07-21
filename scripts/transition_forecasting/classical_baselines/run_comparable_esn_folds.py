#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from transition_forecasting.classical_baselines.comparable_esn import (
    run_comparable_esn_folds,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact field-test ESN specification over frozen purged "
            "walk-forward folds without evaluating the reserved test partition."
        )
    )
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _compact_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    path = metrics[metrics["horizon"].eq("path")].copy()
    path["family"] = path["model"].str.replace(r"_seed\d+$", "", regex=True)
    grouped = (
        path.groupby(["family", "group_type", "group_value"], as_index=False)
        .agg(mean_qlike=("qlike", "mean"), mean_rmse=("rmse", "mean"))
    )

    rows = []
    for family in sorted(grouped["family"].unique()):
        current = grouped[grouped["family"].eq(family)]

        def value(group_type: str, group_value: str, metric: str) -> float:
            match = current[
                current["group_type"].eq(group_type)
                & current["group_value"].eq(group_value)
            ]
            return float(match.iloc[0][metric]) if len(match) == 1 else float("nan")

        rows.append(
            {
                "model": family,
                "pooled_rmse": value("pooled", "all", "mean_rmse"),
                "pooled_qlike": value("pooled", "all", "mean_qlike"),
                "control_qlike": value("label", "control", "mean_qlike"),
                "transition_qlike": value("label", "transition", "mean_qlike"),
                "lead10_transition_qlike": value(
                    "lead_label", "L10_transition", "mean_qlike"
                ),
                "lead5_transition_qlike": value(
                    "lead_label", "L5_transition", "mean_qlike"
                ),
                "lead1_transition_qlike": value(
                    "lead_label", "L1_transition", "mean_qlike"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("transition_qlike")


def run(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metrics, predictions, details = run_comparable_esn_folds(args.fold_dir)
    family_summary = details.pop("family_summary")

    metrics.to_csv(args.out_dir / "metrics_by_seed_and_fold.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    family_summary.to_csv(args.out_dir / "summary_by_family_and_fold.csv", index=False)
    compact = _compact_summary(metrics)
    compact.to_csv(args.out_dir / "compact_path_summary.csv", index=False)

    summary = {
        "schema_version": 1,
        "experiment": "exact_field_esn_on_frozen_folds",
        "test_evaluated": False,
        "fold_dir": str(args.fold_dir),
        **details,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFold geometry")
    print(pd.DataFrame(summary["fold_counts"]).to_string(index=False))
    print("\nMean path metrics across folds and seeds")
    print(compact.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults: {args.out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
