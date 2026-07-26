from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.palindrome_noise_assay import (
    PalindromeNoiseAssayConfig,
    run_palindrome_noise_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/palindrome_noise_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final six-atom A/B/A-palindrome density-matrix noise assay "
            "with a frozen clean six-mode readout."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260726)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--correction-lambda", type=float, default=1.0)
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument(
        "--amplitude-t1-us",
        type=float,
        nargs="*",
        default=[200.0, 100.0, 50.0],
    )
    parser.add_argument(
        "--dephasing-t2-us",
        type=float,
        nargs="*",
        default=[100.0, 50.0],
    )
    parser.add_argument(
        "--depolarizing-probabilities",
        type=float,
        nargs="*",
        default=[0.005, 0.01, 0.03],
    )
    parser.add_argument(
        "--no-combined-moderate",
        action="store_true",
        help="Omit the combined T1=100 us, T2=50 us, p=0.01 scenario.",
    )
    parser.add_argument(
        "--scenario-names",
        nargs="*",
        default=[],
        help=(
            "Optional exact scenario names for a bounded smoke/partial run. "
            "ideal_density is always included."
        ),
    )
    return parser.parse_args()


def frozen_reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        spacing_short_um=8.5,
        spacing_long_um=10.0,
        defect_edge=1,
        defect_offset_um=0.6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.25, 0.5, 1.0),
        max_phase_per_substep=0.25,
        max_substeps_per_step=512,
        shots=None,
        shot_seed=20260726,
    )


def frozen_geometry() -> StaggeredLadderGeometryConfig:
    return StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=8.5,
        row_spacing_um=9.0,
        stagger_fraction=0.35,
        bottom_spacing_scale=1.05,
        defect_site=4,
        defect_dx_um=0.35,
        defect_dy_um=-0.40,
    )


def main() -> None:
    args = parse_args()
    assay = PalindromeNoiseAssayConfig(
        fold=args.fold,
        lead=args.lead,
        sequence_length=args.sequence_length,
        max_per_class=args.max_per_class,
        selection_seed=args.selection_seed,
        ridge_alpha=args.ridge_alpha,
        correction_lambda=args.correction_lambda,
        prequential_blocks=args.prequential_blocks,
        amplitude_damping_t1_us=tuple(args.amplitude_t1_us),
        dephasing_t2_us=tuple(args.dephasing_t2_us),
        depolarizing_probabilities=tuple(args.depolarizing_probabilities),
        include_combined_moderate=not args.no_combined_moderate,
        selected_scenarios=tuple(args.scenario_names),
    )
    run_dir = run_palindrome_noise_assay(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        assay=assay,
        candidate_features=CandidateFeatureConfig(),
        reservoir=frozen_reservoir(),
        geometry=frozen_geometry(),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
