from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.transition_signal_broad_har import (
    run_transition_signal_broad_har_assay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a transition-emphasized ladder readout while preserving the established all-row HAR baseline."
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
            "results/transition_forecasting/qrc/"
            "run_transition_signal_broad_har_assay"
        ),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()
    run_dir = run_transition_signal_broad_har_assay(
        source_run=args.source_run,
        results_root=args.results_root,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
