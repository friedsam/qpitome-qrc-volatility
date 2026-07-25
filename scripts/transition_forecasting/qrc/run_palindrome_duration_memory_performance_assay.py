from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.palindrome_duration_memory_performance_assay import (
    PalindromeDurationMemoryPerformanceConfig,
    run_palindrome_duration_memory_performance_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
LEGACY_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/"
    "run_palindrome_duration_memory_performance_assay"
)
_REQUIRED_FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
)


def _is_fold_dir(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and all(
        (candidate / filename).is_file() for filename in _REQUIRED_FOLD_FILES
    )


def resolve_fold_dir(path: Path) -> Path:
    requested = Path(path)
    if _is_fold_dir(requested):
        return requested
    direct = [
        candidate
        for candidate in (DEFAULT_FOLD_DIR, LEGACY_FOLD_DIR)
        if candidate != requested and _is_fold_dir(candidate)
    ]
    if len(direct) == 1:
        return direct[0]
    discovered: list[Path] = []
    root = Path("data/processed")
    if root.is_dir():
        for manifest in root.glob(
            "**/purged_walk_forward_folds/rematched_rolling_manifest.csv"
        ):
            candidate = manifest.parent
            if (
                _is_fold_dir(candidate)
                and candidate not in discovered
                and not any("archive" in part.lower() for part in candidate.parts)
            ):
                discovered.append(candidate)
    if len(discovered) == 1:
        return discovered[0]
    listed = "\n".join(f"  - {value}" for value in discovered) or "  (none)"
    raise FileNotFoundError(
        "Could not resolve the active purged walk-forward fold directory.\n"
        f"Requested: {requested}\n"
        f"Discovered non-archive candidates:\n{listed}\n"
        "Pass the intended directory with --fold-dir."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map palindrome step duration against intrinsic synthetic memory, "
            "financial-history lag recovery, and leakage-safe lead-5 forecasting."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=[
            "level_only",
            "level_instability",
            "level_positive_shock_energy",
        ],
        choices=[
            "level_only",
            "level_instability",
            "level_positive_shock_energy",
        ],
    )
    parser.add_argument(
        "--durations-us",
        type=float,
        nargs="+",
        default=[0.010, 0.015, 0.020, 0.025, 0.030],
    )
    parser.add_argument(
        "--schedule-names",
        nargs="+",
        default=["crossover_Ahalf_B_Ahalf"],
        choices=["crossover_Ahalf_B_Ahalf", "crossover_Bhalf_A_Bhalf"],
        help=(
            "Hold the incumbent orientation fixed for the clean first sweep. "
            "Add crossover_Bhalf_A_Bhalf explicitly for the mirrored follow-up."
        ),
    )
    parser.add_argument(
        "--interaction-labels",
        nargs="+",
        default=["on", "off"],
        choices=["on", "off"],
    )
    parser.add_argument(
        "--memory-delays", type=int, nargs="+", default=[1, 2, 4, 5, 8, 12]
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--correction-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260725)
    parser.add_argument("--synthetic-samples", type=int, default=320)
    parser.add_argument(
        "--synthetic-seeds",
        type=int,
        nargs="+",
        default=[20260724, 20260725, 20260726],
    )
    parser.add_argument(
        "--synthetic-alphas",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--instability-window", type=int, default=5)
    parser.add_argument("--shock-window", type=int, default=5)

    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
    parser.add_argument(
        "--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    parser.add_argument("--drive-phase-rad", type=float, default=0.0)
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument("--shot-seed", type=int, default=20260725)

    parser.add_argument("--ladder-longitudinal-spacing-um", type=float, default=8.5)
    parser.add_argument("--ladder-row-spacing-um", type=float, default=9.0)
    parser.add_argument("--ladder-stagger-fraction", type=float, default=0.35)
    parser.add_argument("--ladder-bottom-spacing-scale", type=float, default=1.05)
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_dir = resolve_fold_dir(args.fold_dir)
    config = PalindromeDurationMemoryPerformanceConfig(
        folds=tuple(args.folds),
        lead=int(args.lead),
        max_per_class=int(args.max_per_class),
        sequence_length=int(args.sequence_length),
        representations=tuple(args.representations),
        durations_us=tuple(float(value) for value in args.durations_us),
        schedule_names=tuple(args.schedule_names),
        interaction_labels=tuple(args.interaction_labels),
        memory_delays=tuple(int(value) for value in args.memory_delays),
        ridge_alphas=tuple(float(value) for value in args.ridge_alphas),
        correction_lambdas=tuple(float(value) for value in args.correction_lambdas),
        inner_holdout_fraction=float(args.inner_holdout_fraction),
        prequential_blocks=int(args.prequential_blocks),
        selection_seed=int(args.selection_seed),
        synthetic_samples=int(args.synthetic_samples),
        synthetic_seeds=tuple(int(value) for value in args.synthetic_seeds),
        synthetic_alphas=tuple(float(value) for value in args.synthetic_alphas),
    )
    candidate_features = CandidateFeatureConfig(
        instability_window=int(args.instability_window),
        shock_window=int(args.shock_window),
    )
    # step_duration_us is a template value only; the assay replaces it at every
    # configured duration before simulation.
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=float(args.delta_center_rad_us),
        delta_span_rad_us=float(args.delta_span_rad_us),
        omega_base_rad_us=float(args.omega_base_rad_us),
        omega_mod_fraction=float(args.omega_mod_fraction),
        step_duration_us=float(args.durations_us[0]),
        probe_fractions=tuple(args.probe_fractions),
        shots=None,
        shot_seed=int(args.shot_seed),
    )
    geometry = StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=float(args.ladder_longitudinal_spacing_um),
        row_spacing_um=float(args.ladder_row_spacing_um),
        stagger_fraction=float(args.ladder_stagger_fraction),
        bottom_spacing_scale=float(args.ladder_bottom_spacing_scale),
        defect_site=int(args.ladder_defect_site),
        defect_dx_um=float(args.ladder_defect_dx_um),
        defect_dy_um=float(args.ladder_defect_dy_um),
    )
    run_dir = run_palindrome_duration_memory_performance_assay(
        fold_dir=fold_dir,
        results_root=args.out_root,
        config=config,
        candidate_features=candidate_features,
        reservoir=reservoir,
        geometry=geometry,
        interaction_scale=float(args.interaction_scale),
        drive_phase_rad=float(args.drive_phase_rad),
        run_id=str(args.run_id),
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
