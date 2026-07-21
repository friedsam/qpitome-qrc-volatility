from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.temporal_rydberg_chain import (
    SUPPORTED_CONDITIONS,
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    TemporalRydbergExperimentConfig,
    run_temporal_rydberg_chain_experiment,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-channel temporal Rydberg asymmetric-chain "
            "development assay."
        )
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=DEFAULT_FOLD_DIR,
    )
    parser.add_argument("--contaminated-samples", type=Path)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=[1, 5, 10],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=SUPPORTED_CONDITIONS,
        default=list(SUPPORTED_CONDITIONS),
    )
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=24)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--readout-mode",
        choices=["direct", "har_residual"],
        default="direct",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--level-channel-name",
        default="log_volatility_level",
    )
    parser.add_argument(
        "--fallback-level-channel",
        type=int,
        default=0,
    )

    parser.add_argument("--n-atoms", type=int, default=6)
    parser.add_argument(
        "--spacing-short-um",
        type=float,
        default=8.5,
    )
    parser.add_argument(
        "--spacing-long-um",
        type=float,
        default=10.0,
    )
    parser.add_argument("--defect-edge", type=int, default=1)
    parser.add_argument(
        "--defect-offset-um",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--delta-center-rad-us",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--delta-span-rad-us",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--omega-base-rad-us",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--omega-mod-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--step-duration-us",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--probe-fractions",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--shot-seed",
        type=int,
        default=20260721,
    )
    parser.add_argument(
        "--max-phase-per-substep",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-substeps-per-step",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_temporal_rydberg_chain"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = TemporalRydbergExperimentConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        conditions=tuple(args.conditions),
        sequence_length=args.sequence_length,
        max_per_class=args.max_per_class,
        alphas=tuple(args.alphas),
        readout_mode=args.readout_mode,
        seed=args.seed,
        level_channel_name=args.level_channel_name,
        fallback_level_channel=args.fallback_level_channel,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=args.n_atoms,
        spacing_short_um=args.spacing_short_um,
        spacing_long_um=args.spacing_long_um,
        defect_edge=args.defect_edge,
        defect_offset_um=args.defect_offset_um,
        delta_center_rad_us=args.delta_center_rad_us,
        delta_span_rad_us=args.delta_span_rad_us,
        omega_base_rad_us=args.omega_base_rad_us,
        omega_mod_fraction=args.omega_mod_fraction,
        step_duration_us=args.step_duration_us,
        probe_fractions=tuple(args.probe_fractions),
        max_phase_per_substep=args.max_phase_per_substep,
        max_substeps_per_step=args.max_substeps_per_step,
        shots=args.shots,
        shot_seed=args.shot_seed,
    )
    run_dir = run_temporal_rydberg_chain_experiment(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        experiment=experiment,
        reservoir=reservoir,
        contaminated_samples=args.contaminated_samples,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
