from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.instability_mechanism_tools import (
    InstabilityMechanismAssayConfig,
    SUPPORTED_CONTROLS,
)
from transition_forecasting.qrc.rydberg_instability_mechanism_assay import (
    run_rydberg_instability_mechanism_assay,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig

DEFAULT_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether instability-QRC gains require temporal order and intermediate Rydberg interactions."
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument(
        "--interaction-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument("--control-scales", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=SUPPORTED_CONTROLS,
        default=["reset", "shuffled", "reversed", "block_shuffled"],
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--pca-components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 0])
    parser.add_argument("--pls-components", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--individual-pc-count", type=int, default=16)
    parser.add_argument(
        "--readout-modes",
        nargs="+",
        choices=["direct", "prequential_har_residual"],
        default=["direct", "prequential_har_residual"],
    )
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--instability-window", type=int, default=5)

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
    parser.add_argument("--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--shots", type=int)
    parser.add_argument("--shot-seed", type=int, default=20260721)

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_rydberg_instability_mechanism_assay"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assay = InstabilityMechanismAssayConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        max_per_class=args.max_per_class,
        interaction_scales=tuple(args.interaction_scales),
        control_scales=tuple(args.control_scales),
        controls=tuple(args.controls),
        block_size=args.block_size,
        alphas=tuple(args.alphas),
        pca_components=tuple(args.pca_components),
        pls_components=tuple(args.pls_components),
        individual_pc_count=args.individual_pc_count,
        readout_modes=tuple(args.readout_modes),
        prequential_blocks=args.prequential_blocks,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    candidates = CandidateFeatureConfig(instability_window=args.instability_window)
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
    run_dir = run_rydberg_instability_mechanism_assay(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        assay=assay,
        candidate_features=candidates,
        reservoir=reservoir,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
