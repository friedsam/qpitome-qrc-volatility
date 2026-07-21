from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from transition_forecasting.classical_baselines.baselines import (
    frozen_sanity,
    run_frozen_fold_baselines,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run primitive classical baselines on the frozen purged walk-forward "
            "transition folds without evaluating the reserved test partition."
        )
    )
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--shuffled-seed", type=int, default=20260720)
    return parser.parse_args(argv)


def _compact_path_table(metrics: pd.DataFrame) -> pd.DataFrame:
    path = metrics[metrics["horizon"].eq("path")]
    rows: list[dict[str, object]] = []
    for model in sorted(path["model"].unique()):
        model_rows = path[path["model"].eq(model)]

        def mean_value(group_type: str, group_value: str, metric: str) -> float:
            selected = model_rows[
                model_rows["group_type"].eq(group_type)
                & model_rows["group_value"].eq(group_value)
            ]
            return float(selected[metric].mean())

        rows.append(
            {
                "model": model,
                "pooled_rmse": mean_value("pooled", "all", "rmse"),
                "pooled_qlike": mean_value("pooled", "all", "qlike"),
                "control_qlike": mean_value("label", "control", "qlike"),
                "transition_qlike": mean_value("label", "transition", "qlike"),
                "lead10_transition_qlike": mean_value(
                    "lead_label", "L10_transition", "qlike"
                ),
                "lead5_transition_qlike": mean_value(
                    "lead_label", "L5_transition", "qlike"
                ),
                "lead1_transition_qlike": mean_value(
                    "lead_label", "L1_transition", "qlike"
                ),
            }
        )
    return pd.DataFrame(rows)


def _fold_count_table(fold_summary: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(fold_summary["fold_counts"])


def run(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=False)

    metrics, predictions, fold_summary = run_frozen_fold_baselines(
        args.fold_dir,
        alpha=args.alpha,
        shuffled_seed=args.shuffled_seed,
    )
    metrics.to_csv(args.out_dir / "metrics_by_fold.csv", index=False)
    metrics[metrics["horizon"].ne("path")].to_csv(
        args.out_dir / "metrics_by_horizon.csv",
        index=False,
    )
    metrics[
        metrics["horizon"].eq("path") & metrics["group_type"].ne("pooled")
    ].to_csv(args.out_dir / "metrics_by_group.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    compact = _compact_path_table(metrics)
    compact.to_csv(args.out_dir / "compact_path_summary.csv", index=False)

    directional_checks = frozen_sanity(metrics)
    (args.out_dir / "directional_checks.json").write_text(
        json.dumps(directional_checks, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "experiment": "primitive_frozen_fold_baselines",
        "test_evaluated": False,
        "fold_dir": str(args.fold_dir),
        "alpha": float(args.alpha),
        "shuffled_seed": int(args.shuffled_seed),
        "fold_summary": fold_summary,
        "directional_checks": directional_checks,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    metrics = pd.read_csv(args.out_dir / "metrics_by_fold.csv")

    print("\nFold geometry")
    print(_fold_count_table(summary["fold_summary"]).to_string(index=False))
    print("\nMean path metrics across validation folds")
    print(
        _compact_path_table(metrics).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nDirectional checks")
    for check in summary["directional_checks"]["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  {status}: {check['check']}")
    print(f"\nFull outputs: {args.out_dir}")
    # Directional scientific checks are reported, not treated as execution failures.
    return 0
