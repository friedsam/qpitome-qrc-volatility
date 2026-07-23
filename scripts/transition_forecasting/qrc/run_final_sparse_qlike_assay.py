from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.final_sparse_qlike_assay import (
    FinalSparseQlikeAssayConfig,
    run_final_sparse_qlike_assay,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/"
    "archive_pre_controls_20260723_001856/purged_walk_forward_folds"
)
DEFAULT_SOURCE_RUN = Path(
    "results/transition_forecasting/qrc/run_precontrol_ladder_spacing_assay/"
    "precontrol_spacing_8_9_10_002"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_final_sparse_qlike_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen nine-mode ridge, sparse density-curvature ridge, "
            "and constrained/unconstrained QLIKE specialists using the exact 9 um "
            "probability cache and five 1000-shot replicates."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--source-spacing-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--later-folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=20260722)
    parser.add_argument("--shot-count", type=int, default=1000)
    parser.add_argument(
        "--shot-seeds",
        type=int,
        nargs="+",
        default=[20260731, 20260732, 20260733, 20260734, 20260735],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FinalSparseQlikeAssayConfig(
        folds=tuple(args.folds),
        later_folds=tuple(args.later_folds),
        leads=tuple(args.leads),
        max_per_class=args.max_per_class,
        seed=args.selection_seed,
        shot_count=args.shot_count,
        shot_seeds=tuple(args.shot_seeds),
    )
    run_dir = run_final_sparse_qlike_assay(
        fold_dir=args.fold_dir,
        source_spacing_run=args.source_spacing_run,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
