from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.residual_head_root_cause_assay import (
    ResidualHeadRootCauseConfig,
    run_residual_head_root_cause_assay,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/"
    "archive_pre_controls_20260723_001856/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_residual_head_root_cause_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit why the ladder residual family selects positive early calibration "
            "and whether the Ridge intercept masks sample-specific QRC direction."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--source-spacing-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--later-folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--selection-seed", type=int, default=20260722)
    parser.add_argument("--geometry-name", default="row_9p0um")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResidualHeadRootCauseConfig(
        folds=tuple(args.folds),
        later_folds=tuple(args.later_folds),
        selection_seed=args.selection_seed,
        geometry_name=args.geometry_name,
    )
    run_dir = run_residual_head_root_cause_assay(
        fold_dir=args.fold_dir,
        source_spacing_run=args.source_spacing_run,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
