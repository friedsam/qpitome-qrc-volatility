#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.fold_datasets import (
    DEFAULT_CONTROLS_PER_POSITIVE,
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_FRACTION,
    build_one_and_three_channel_folds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_1D = (
    REPO_ROOT
    / "data/processed/transition_forecasting/global_transition_dataset_1d"
)
DEFAULT_3D = (
    REPO_ROOT
    / "data/processed/transition_forecasting/global_transition_dataset_3d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build inspectable candidate pools and the validated purged walk-forward "
            "fold datasets for both the canonical 1D and deterministic 3D representations."
        )
    )
    parser.add_argument("--dataset-1d", type=Path, default=DEFAULT_1D)
    parser.add_argument("--dataset-3d", type=Path, default=DEFAULT_3D)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    parser.add_argument(
        "--controls-per-positive",
        type=int,
        default=DEFAULT_CONTROLS_PER_POSITIVE,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_one_and_three_channel_folds(
        args.dataset_1d,
        args.dataset_3d,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        embargo_days=args.embargo_days,
        controls_per_positive=args.controls_per_positive,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
