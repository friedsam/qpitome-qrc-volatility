from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_scaling import (
    TemporalRydbergScalingConfig,
    run_temporal_rydberg_scaling,
)


DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/"
    "archive_pre_controls_20260723_001856/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_temporal_rydberg_scaling"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact-state temporal Rydberg representation, mechanism, "
            "runtime, and memory scaling on one deterministic development panel."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument(
        "--n-atoms",
        type=int,
        nargs="+",
        default=[5, 6, 8, 10],
        help=(
            "Strictly increasing atom counts. Start with 5 6 8 10; add 15 only "
            "after the bounded run establishes practical runtime."
        ),
    )
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["ordered", "shuffled", "interaction_off"],
        default=["ordered", "shuffled", "interaction_off"],
    )
    parser.add_argument(
        "--level-channel-name",
        default="log_volatility_level",
    )
    parser.add_argument("--fallback-level-channel", type=int, default=0)

    parser.add_argument("--spacing-short-um", type=float, default=8.5)
    parser.add_argument("--spacing-long-um", type=float, default=10.0)
    parser.add_argument("--defect-edge", type=int, default=1)
    parser.add_argument("--defect-offset-um", type=float, default=0.6)
    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.35)
    parser.add_argument("--step-duration-us", type=float, default=0.08)
    parser.add_argument(
        "--probe-fractions",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
    )
    parser.add_argument("--max-phase-per-substep", type=float, default=0.25)
    parser.add_argument("--max-substeps-per-step", type=int, default=512)

    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaling = TemporalRydbergScalingConfig(
        n_atoms=tuple(args.n_atoms),
        fold=args.fold,
        lead=args.lead,
        sequence_length=args.sequence_length,
        max_per_class=args.max_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        conditions=tuple(args.conditions),
        level_channel_name=args.level_channel_name,
        fallback_level_channel=args.fallback_level_channel,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=args.n_atoms[0],
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
        shots=None,
        shot_seed=args.seed,
    )
    run_dir = run_temporal_rydberg_scaling(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        scaling=scaling,
        reservoir_template=reservoir,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
