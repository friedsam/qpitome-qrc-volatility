from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.ladder_finite_shot_study import (
    LadderFiniteShotStudyConfig,
    run_ladder_finite_shot_study,
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
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_ladder_finite_shot_study"
)
DEFAULT_CACHE_ROOT = Path(
    "results/transition_forecasting/qrc/"
    "run_ladder_finite_shot_study_probability_cache"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache exact six-atom ladder probe probabilities and evaluate finite-shot "
            "robustness of the linear and degree-two segmented readouts."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-horizon", type=int, default=4)
    parser.add_argument(
        "--shot-counts",
        type=int,
        nargs="+",
        default=[250, 500, 1000, 2000, 5000],
    )
    parser.add_argument(
        "--shot-seeds",
        type=int,
        nargs="+",
        default=[20260731, 20260732, 20260733, 20260734, 20260735],
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("shot_consistent", "exact_train_shot_val"),
        default=["shot_consistent", "exact_train_shot_val"],
    )
    parser.add_argument(
        "--global-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument(
        "--early-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5],
    )
    parser.add_argument(
        "--transition-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument("--linear-alpha", type=float, default=100.0)
    parser.add_argument(
        "--polynomial-alphas",
        type=float,
        nargs="+",
        default=[100.0, 1000.0, 10000.0],
    )
    parser.add_argument("--ladder-interaction-scale", type=float, default=1.25)
    parser.add_argument("--selection-seed", type=int, default=20260722)
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
    parser.add_argument("--shot-seed", type=int, default=20260721)

    parser.add_argument(
        "--ladder-longitudinal-spacing-um",
        type=float,
        default=8.5,
    )
    parser.add_argument("--ladder-row-spacing-um", type=float, default=9.0)
    parser.add_argument("--ladder-stagger-fraction", type=float, default=0.35)
    parser.add_argument(
        "--ladder-bottom-spacing-scale",
        type=float,
        default=1.05,
    )
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)

    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Delete and regenerate only the cache files matching this exact run identity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LadderFiniteShotStudyConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        max_per_class=args.max_per_class,
        sequence_length=args.sequence_length,
        prequential_blocks=args.prequential_blocks,
        inner_holdout_fraction=args.inner_holdout_fraction,
        split_horizon=args.split_horizon,
        shot_counts=tuple(args.shot_counts),
        shot_seeds=tuple(args.shot_seeds),
        protocols=tuple(args.protocols),
        global_lambdas=tuple(args.global_lambdas),
        early_lambdas=tuple(args.early_lambdas),
        transition_lambdas=tuple(args.transition_lambdas),
        linear_alpha=args.linear_alpha,
        polynomial_alphas=tuple(args.polynomial_alphas),
        ladder_interaction_scale=args.ladder_interaction_scale,
        seed=args.selection_seed,
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
        shots=None,
        shot_seed=args.shot_seed,
    )
    geometry = StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=args.ladder_longitudinal_spacing_um,
        row_spacing_um=args.ladder_row_spacing_um,
        stagger_fraction=args.ladder_stagger_fraction,
        bottom_spacing_scale=args.ladder_bottom_spacing_scale,
        defect_site=args.ladder_defect_site,
        defect_dx_um=args.ladder_defect_dx_um,
        defect_dy_um=args.ladder_defect_dy_um,
    )
    run_dir = run_ladder_finite_shot_study(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        cache_root=args.cache_root,
        config=config,
        candidate_features=candidate_features,
        reservoir=reservoir,
        ladder_geometry=geometry,
        run_id=args.run_id,
        force_cache=args.force_cache,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
