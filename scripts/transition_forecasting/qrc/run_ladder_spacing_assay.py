from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.ladder_spacing_assay import (
    LadderSpacingAssayConfig,
    run_ladder_spacing_assay,
)
from transition_forecasting.qrc.transition_signal_readout_assay import (
    TransitionSignalAssayConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared five-geometry ladder distance assay with fixed "
            "median nearest-neighbor coupling and transition-emphasis readout."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/run_ladder_finite_shot_study/"
            "shot_deterministic_controls_folds4_8_001"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/qrc/run_ladder_spacing_assay"),
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--folds", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    args = parser.parse_args()

    folds = tuple(args.folds)
    run_dir = run_ladder_spacing_assay(
        source_run=args.source_run,
        results_root=args.results_root,
        transition_config=TransitionSignalAssayConfig(folds=folds),
        spacing_config=LadderSpacingAssayConfig(folds=folds),
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
