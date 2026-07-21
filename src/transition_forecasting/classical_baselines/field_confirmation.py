from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.classical_baselines.baselines import (
    compare_with_reference,
    run_field_reproduction,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the primitive-model broad chronological field test."
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "config/transition_forecasting/classical_baselines/"
            "field_experiment_reference.json"
        ),
    )
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.stride < 1:
        raise ValueError("stride must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    metrics, predictions, counts = run_field_reproduction(
        args.panel,
        alpha=args.alpha,
        stride=args.stride,
    )
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    comparison = compare_with_reference(metrics, args.reference)
    (args.out_dir / "field_experiment_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "experiment": "primitive_field_confirmation",
        "test_evaluated": False,
        "panel": str(args.panel),
        "reference": str(args.reference),
        "alpha": float(args.alpha),
        "stride": int(args.stride),
        "counts": counts,
        "reproduction_passed": bool(comparison["all_checks_passed"]),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0 if summary["reproduction_passed"] else 1
