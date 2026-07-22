from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
)
from transition_forecasting.qrc.rydberg_susceptibility_assay import (
    run_rydberg_susceptibility_assay,
)
from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    SusceptibilityAssayConfig,
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
            "Prepare the frozen Rydberg ladder from each 40-day history, "
            "interrogate it with a short detuning probe, and test whether "
            "susceptibility and displaced-reporter responses separate L5 "
            "transitions from matched controls."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument(
        "--probe-delta-offset-rad-us",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--response-probe-steps",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument("--classifier-c", type=float, default=0.1)
    parser.add_argument(
        "--shot-budgets",
        type=int,
        nargs="+",
        default=[1000, 5000],
    )
    parser.add_argument("--shot-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
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
    parser.add_argument(
        "--ladder-stagger-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--ladder-bottom-spacing-scale",
        type=float,
        default=1.05,
    )
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_rydberg_susceptibility_assay"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assay = SusceptibilityAssayConfig(
        folds=tuple(args.folds),
        lead=args.lead,
        max_per_class=args.max_per_class,
        sequence_length=args.sequence_length,
        interaction_scale=args.interaction_scale,
        probe_delta_offset_rad_us=args.probe_delta_offset_rad_us,
        response_probe_steps=tuple(args.response_probe_steps),
        classifier_c=args.classifier_c,
        shot_budgets=tuple(args.shot_budgets),
        shot_replicates=args.shot_replicates,
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
        shots=None,
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
    run_dir = run_rydberg_susceptibility_assay(
        fold_dir=args.fold_dir,
        results_root=args.out_root,
        assay=assay,
        candidate_features=candidate_features,
        reservoir=reservoir,
        ladder_geometry=ladder_geometry,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
