from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.ladder_mode_readout_tools import (
    LadderModeReadoutConfig,
)
from transition_forecasting.qrc.ladder_mode_readout_tuning import (
    run_ladder_mode_readout_tuning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune geometry-aware ladder occupation readouts from saved ladder "
            "challenger feature archives without rerunning the reservoir."
        )
    )
    parser.add_argument("--ladder-run-dir", type=Path, required=True)
    parser.add_argument(
        "--ordered-ladder-cases",
        nargs="+",
        default=[
            "ladder_ordered_0p75",
            "ladder_ordered_1p00",
            "ladder_ordered_1p25",
        ],
    )
    parser.add_argument(
        "--reference-cases",
        nargs="+",
        default=[
            "chain_interaction_off",
            "chain_interacting_1p00",
        ],
    )
    parser.add_argument(
        "--control-cases",
        nargs="*",
        default=[
            "ladder_reset_1p00",
            "ladder_shuffled_1p00",
            "ladder_reversed_1p00",
            "ladder_block_shuffled_1p00",
        ],
    )
    parser.add_argument(
        "--readout-families",
        nargs="+",
        choices=[
            "occupation_pca4",
            "symmetric_modes",
            "antisymmetric_modes",
            "compact_modes",
            "all_modes",
        ],
        default=[
            "occupation_pca4",
            "symmetric_modes",
            "antisymmetric_modes",
            "compact_modes",
            "all_modes",
        ],
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--global-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_ladder_mode_readout_tuning"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LadderModeReadoutConfig(
        ordered_ladder_cases=tuple(args.ordered_ladder_cases),
        reference_cases=tuple(args.reference_cases),
        control_cases=tuple(args.control_cases),
        readout_families=tuple(args.readout_families),
        folds=tuple(args.folds),
        inner_holdout_fraction=args.inner_holdout_fraction,
        global_lambdas=tuple(args.global_lambdas),
        ridge_alpha=args.ridge_alpha,
        pca_components=args.pca_components,
    )
    run_dir = run_ladder_mode_readout_tuning(
        ladder_run_dir=args.ladder_run_dir,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
