from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.frozen_ladder_confirmation import (
    run_frozen_ladder_confirmation,
)
from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    FrozenLadderConfirmationConfig,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


DEFAULT_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the single frozen 1.25V symmetric-mode ladder and its "
            "chain references on untouched non-test validation folds 4--8."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--global-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--ladder-interaction-scale", type=float, default=1.25)
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
    parser.add_argument(
        "--probe-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument("--shot-seed", type=int, default=20260721)

    parser.add_argument(
        "--ladder-longitudinal-spacing-um", type=float, default=8.5
    )
    parser.add_argument("--ladder-row-spacing-um", type=float, default=9.0)
    parser.add_argument("--ladder-stagger-fraction", type=float, default=0.35)
    parser.add_argument(
        "--ladder-bottom-spacing-scale", type=float, default=1.05
    )
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_frozen_ladder_confirmation"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confirmation = FrozenLadderConfirmationConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        max_per_class=args.max_per_class,
        sequence_length=args.sequence_length,
        prequential_blocks=args.prequential_blocks,
        inner_holdout_fraction=args.inner_holdout_fraction,
        global_lambdas=tuple(args.global_lambdas),
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        ladder_interaction_scale=args.ladder_interaction_scale,
        seed=args.seed,
    )
    candidate_features = CandidateFeatureConfig(
        instability_window=args.instability_window,
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
    ladder_geometry = StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=args.ladder_longitudinal_spacing_um,
        row_spacing_um=args.ladder_row_spacing_um,
        stagger_fraction=args.ladder_stagger_fraction,
        bottom_spacing_scale=args.ladder_bottom_spacing_scale,
        defect_site=args.ladder_defect_site,
        defect_dx_um=args.ladder_defect_dx_um,
        defect_dy_um=args.ladder_defect_dy_um,
    )
    run_dir = run_frozen_ladder_confirmation(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        confirmation=confirmation,
        candidate_features=candidate_features,
        reservoir=reservoir,
        ladder_geometry=ladder_geometry,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
