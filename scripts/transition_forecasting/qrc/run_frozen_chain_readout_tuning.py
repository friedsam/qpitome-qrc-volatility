from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    FrozenChainReadoutTuningConfig,
)
from transition_forecasting.qrc.frozen_chain_readout_tuning import (
    run_frozen_chain_readout_tuning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune the frozen instability-chain residual readout from a saved "
            "mechanism-assay feature archive without rerunning QRC."
        )
    )
    parser.add_argument("--mechanism-run-dir", type=Path, required=True)
    parser.add_argument("--case", default="ordered_interaction_1p00")
    parser.add_argument(
        "--observable-family",
        choices=["occupation", "connected", "pair", "all"],
        default="occupation",
    )
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    parser.add_argument(
        "--global-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument(
        "--pca-prefixes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 12],
    )
    parser.add_argument(
        "--pc-bands",
        nargs="+",
        default=["5", "2-5", "4-8", "5-8", "9-12"],
    )
    parser.add_argument(
        "--pls-components",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--gate-types",
        nargs="+",
        choices=[
            "none",
            "instability_high_65",
            "instability_ramp_50_90",
            "har_slope_ramp_50_90",
            "joint_ramp_50_90",
        ],
        default=[
            "none",
            "instability_high_65",
            "instability_ramp_50_90",
            "har_slope_ramp_50_90",
            "joint_ramp_50_90",
        ],
    )
    parser.add_argument(
        "--calibration-modes",
        nargs="+",
        choices=["global", "horizon"],
        default=["global", "horizon"],
    )
    parser.add_argument(
        "--selection-policies",
        nargs="+",
        choices=["qlike", "balanced"],
        default=["qlike", "balanced"],
    )
    parser.add_argument("--balanced-rmse-weight", type=float, default=1.0)
    parser.add_argument("--fixed-reference-components", type=int, default=4)
    parser.add_argument("--fixed-reference-alpha", type=float, default=100.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_frozen_chain_readout_tuning"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrozenChainReadoutTuningConfig(
        case=args.case,
        observable_family=args.observable_family,
        inner_holdout_fraction=args.inner_holdout_fraction,
        ridge_alphas=tuple(args.ridge_alphas),
        global_lambdas=tuple(args.global_lambdas),
        pca_prefixes=tuple(args.pca_prefixes),
        pc_bands=tuple(args.pc_bands),
        pls_components=tuple(args.pls_components),
        gate_types=tuple(args.gate_types),
        calibration_modes=tuple(args.calibration_modes),
        selection_policies=tuple(args.selection_policies),
        balanced_rmse_weight=args.balanced_rmse_weight,
        fixed_reference_components=args.fixed_reference_components,
        fixed_reference_alpha=args.fixed_reference_alpha,
    )
    run_dir = run_frozen_chain_readout_tuning(
        mechanism_run_dir=args.mechanism_run_dir,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
