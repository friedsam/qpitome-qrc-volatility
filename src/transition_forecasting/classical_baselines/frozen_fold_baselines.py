from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))
    # Directional scientific checks are reported, not treated as execution failures.
    return 0
