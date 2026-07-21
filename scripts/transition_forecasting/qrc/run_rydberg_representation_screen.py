from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.representation_candidates import (
    REPRESENTATIONS,
    CandidateFeatureConfig,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    RepresentationScreenConfig,
    run_rydberg_representation_screen,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig

DEFAULT_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare five causal second-channel representations across ridge, "
            "ESN, and Rydberg QRC."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[10.0, 100.0, 1000.0, 10000.0],
    )
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 0],
    )
    parser.add_argument(
        "--readout-modes",
        nargs="+",
        choices=["direct", "prequential_har_residual"],
        default=["direct", "prequential_har_residual"],
    )
    parser.add_argument("--esn-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)

    parser.add_argument("--instability-window", type=int, default=5)
    parser.add_argument("--shock-window", type=int, default=5)
    parser.add_argument("--short-slope-window", type=int, default=5)
    parser.add_argument("--long-slope-window", type=int, default=20)

    parser.add_argument("--n-atoms", type=int, default=6)
    parser.add_argument("--spacing-short-um", type=float, default=8.5)
    parser.add_argument("--spacing-long-um", type=float, default=10.0)
    parser.add_argument("--defect-edge", type=int, default=1)
    parser.add_argument("--defect-offset-um", type=float, default=0.6)
    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
    parser.add_argument("--step-duration-us", type=float, default=0.03)
    parser.add_argument(
        "--probe-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument("--shot-seed", type=int, default=20260721)

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_rydberg_representation_screen"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen = RepresentationScreenConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        representations=tuple(args.representations),
        max_per_class=args.max_per_class,
        alphas=tuple(args.alphas),
        components=tuple(args.components),
        readout_modes=tuple(args.readout_modes),
        esn_seeds=tuple(args.esn_seeds),
        prequential_blocks=args.prequential_blocks,
        seed=args.seed,
    )
    candidates = CandidateFeatureConfig(
        instability_window=args.instability_window,
        shock_window=args.shock_window,
        short_slope_window=args.short_slope_window,
        long_slope_window=args.long_slope_window,
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
        shots=args.shots,
        shot_seed=args.shot_seed,
    )
    run_dir = run_rydberg_representation_screen(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        screen=screen,
        candidate_features=candidates,
        reservoir=reservoir,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
