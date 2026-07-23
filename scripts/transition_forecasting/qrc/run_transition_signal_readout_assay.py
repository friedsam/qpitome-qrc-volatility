from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.transition_signal_readout_assay import (
    TransitionSignalAssayConfig,
    run_transition_signal_readout_assay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the frozen ladder readout for transition-L5 signal while keeping "
            "calm corrections constrained and hard negatives diagnostic-only."
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
        default=Path(
            "results/transition_forecasting/qrc/run_transition_signal_readout_assay"
        ),
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--folds", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument(
        "--transition-weights",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0, 8.0],
    )
    args = parser.parse_args()

    config = TransitionSignalAssayConfig(
        folds=tuple(args.folds),
        transition_weights=tuple(args.transition_weights),
    )
    run_dir = run_transition_signal_readout_assay(
        source_run=args.source_run,
        results_root=args.results_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
