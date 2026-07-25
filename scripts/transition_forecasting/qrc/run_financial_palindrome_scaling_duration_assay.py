from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.financial_palindrome_scaling_duration_assay import (
    FinancialPalindromeScalingDurationConfig,
    run_financial_palindrome_scaling_duration_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import StaggeredLadderGeometryConfig

DEFAULT_FOLD_DIR = Path("data/processed/transition_forecasting/rolling_folds")
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_financial_palindrome_scaling_duration_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select leakage-safe financial input scaling and local palindrome duration."
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--durations-us", type=float, nargs="+", default=[0.015, 0.020, 0.025])
    parser.add_argument(
        "--scaling-quantiles",
        type=float,
        nargs="+",
        default=[0.005, 0.995, 0.010, 0.990, 0.025, 0.975, 0.050, 0.950],
        help="Flat low/high quantile pairs.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
    parser.add_argument("--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.scaling_quantiles) % 2:
        raise ValueError("--scaling-quantiles requires low/high pairs")
    scaling = tuple(
        (float(args.scaling_quantiles[index]), float(args.scaling_quantiles[index + 1]))
        for index in range(0, len(args.scaling_quantiles), 2)
    )
    config = FinancialPalindromeScalingDurationConfig(
        folds=tuple(args.folds),
        max_per_class=args.max_per_class,
        sequence_length=args.sequence_length,
        durations_us=tuple(args.durations_us),
        scaling_quantiles=scaling,
        ridge_alpha=args.ridge_alpha,
        interaction_scale=args.interaction_scale,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=args.delta_center_rad_us,
        delta_span_rad_us=args.delta_span_rad_us,
        omega_base_rad_us=args.omega_base_rad_us,
        omega_mod_fraction=args.omega_mod_fraction,
        step_duration_us=0.020,
        probe_fractions=tuple(args.probe_fractions),
        shots=None,
        shot_seed=config.selection_seed,
    )
    run_dir = run_financial_palindrome_scaling_duration_assay(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        config=config,
        candidate_features=CandidateFeatureConfig(),
        reservoir=reservoir,
        geometry=StaggeredLadderGeometryConfig(row_spacing_um=9.0),
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
