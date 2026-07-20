#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.dataset import build_processed_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data/raw/transition_forecasting/global_stock_indices_historical_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/transition_forecasting/global_transition_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the single canonical transition-forecasting processed dataset."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument(
        "--parity-control-no-structural-removal",
        action="store_true",
        help=(
            "Temporary experimental mode: detect and document structural bad prints "
            "but retain them to reconstruct the pre-correction lineage."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_processed_dataset(
        args.raw_root,
        args.output_dir,
        expected_structural_flags=args.expected_structural_flags,
        expected_affected_indices=args.expected_affected_indices,
        controls_per_positive=args.controls_per_positive,
        apply_structural_corrections=not args.parity_control_no_structural_removal,
        force=args.force,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
