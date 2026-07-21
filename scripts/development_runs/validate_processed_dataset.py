#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.validation import audit_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the canonical transition processed dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/transition_forecasting/global_transition_dataset"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_processed_dataset(
        args.dataset_dir,
        controls_per_positive=args.controls_per_positive,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"Processed transition dataset audit failed; inspect {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
