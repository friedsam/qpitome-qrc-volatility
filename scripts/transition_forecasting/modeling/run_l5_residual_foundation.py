from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.modeling.l5_residual_foundation import (
    L5ResidualFoundationConfig,
    run_l5_residual_foundation,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/modeling/run_l5_residual_foundation"
)
_REQUIRED_FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
)


def validate_fold_dir(path: Path) -> Path:
    candidate = Path(path)
    missing = [name for name in _REQUIRED_FOLD_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"invalid fold directory {candidate}; missing {missing}. "
            "Pass the restored purged walk-forward data foundation with --fold-dir."
        )
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full-population, target-available, prequential L5 HAR residual "
            "registry before QRC row selection."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--selection-folds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--target-availability-calendar-days", type=int, default=21)
    parser.add_argument("--min-fit-rows", type=int, default=40)
    parser.add_argument("--min-calibration-rows", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = L5ResidualFoundationConfig(
        folds=tuple(args.folds),
        selection_folds=tuple(args.selection_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        target_availability_calendar_days=int(
            args.target_availability_calendar_days
        ),
        min_fit_rows=int(args.min_fit_rows),
        min_calibration_rows=int(args.min_calibration_rows),
        ridge_alpha=float(args.ridge_alpha),
    )
    run_dir = run_l5_residual_foundation(
        fold_dir=validate_fold_dir(args.fold_dir),
        results_root=args.out_root,
        config=config,
        run_id=str(args.run_id),
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
