from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.gated_l5_early_warning_assay import (
    GatedL5EarlyWarningConfig,
    run_gated_l5_early_warning_assay,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/"
    "archive_pre_controls_20260723_001856/purged_walk_forward_folds"
)
DEFAULT_SOURCE_SPACING_RUN = Path(
    "results/transition_forecasting/qrc/run_precontrol_ladder_spacing_assay/"
    "precontrol_spacing_8_9_10_002"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_gated_l5_early_warning_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the failed always-on residual head with a learned L5 crisis gate "
            "and a conditional correction path using cached exact ladder features."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument(
        "--source-spacing-run", type=Path, default=DEFAULT_SOURCE_SPACING_RUN
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--selection-seed", type=int, default=20260722)
    parser.add_argument("--false-alert-budget", type=float, default=0.40)
    parser.add_argument("--geometry-name", default="row_9p0um")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GatedL5EarlyWarningConfig(
        folds=tuple(args.folds),
        selection_seed=args.selection_seed,
        false_alert_budget=args.false_alert_budget,
        geometry_name=args.geometry_name,
    )
    run_dir = run_gated_l5_early_warning_assay(
        fold_dir=args.fold_dir,
        source_spacing_run=args.source_spacing_run,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
