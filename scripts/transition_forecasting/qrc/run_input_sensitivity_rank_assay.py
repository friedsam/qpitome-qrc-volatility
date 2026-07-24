from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.input_sensitivity_rank_assay import (
    InputSensitivityRankConfig,
    run_input_sensitivity_rank_assay,
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
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
LEGACY_FOLD_DIR = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d/"
    "purged_walk_forward_folds"
)
DEFAULT_CLOSE_PANEL = Path(
    "data/fallback/transition_forecasting/"
    "global_stock_indices_historical_data/all_indices_data.csv"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_input_sensitivity_rank_assay"
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
    """Resolve the active fold directory without silently selecting an archive."""

    requested = Path(path)
    if _is_fold_dir(requested):
        return requested

    direct_candidates = tuple(
        candidate
        for candidate in (DEFAULT_FOLD_DIR, LEGACY_FOLD_DIR)
        if candidate != requested and _is_fold_dir(candidate)
    )
    if len(direct_candidates) == 1:
        return direct_candidates[0]

    discovered: list[Path] = []
    processed_root = Path("data/processed")
    if processed_root.is_dir():
        for manifest in processed_root.glob(
            "**/purged_walk_forward_folds/rematched_rolling_manifest.csv"
        ):
            candidate = manifest.parent
            if _is_fold_dir(candidate) and candidate not in discovered:
                discovered.append(candidate)

    active = [
        candidate
        for candidate in discovered
        if not any("archive" in part.lower() for part in candidate.parts)
    ]
    if len(active) == 1:
        return active[0]

    discovered_text = (
        "\n".join(f"  - {candidate}" for candidate in discovered)
        if discovered
        else "  (none found under data/processed)"
    )
    raise FileNotFoundError(
        "Could not locate the active purged walk-forward fold directory.\n"
        f"Requested: {requested}\n"
        "Expected both rematched_rolling_manifest.csv and "
        "rematched_rolling_tensors.npz.\n"
        "Discovered candidates:\n"
        f"{discovered_text}\n"
        "Pass the intended non-archive directory explicitly with --fold-dir."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether level plus downside return creates a more independent, "
            "memory-bearing response than level plus local instability in the frozen "
            "six-atom Rydberg ladder. No volatility forecast or readout selection is run."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--close-panel", type=Path, default=DEFAULT_CLOSE_PANEL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument(
        "--perturb-lags",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 9, 19, 39],
    )
    parser.add_argument("--perturbation-epsilon", type=float, default=0.05)
    parser.add_argument("--reconstruction-alpha", type=float, default=1.0)
    parser.add_argument("--channel-subspace-energy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260724)
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
    parser.add_argument("--interaction-scale", type=float, default=1.25)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_dir = resolve_fold_dir(args.fold_dir)
    close_panel = Path(args.close_panel)
    if not close_panel.is_file():
        raise FileNotFoundError(f"missing close panel: {close_panel}")

    config = InputSensitivityRankConfig(
        folds=tuple(args.folds),
        lead=int(args.lead),
        max_per_class=int(args.max_per_class),
        sequence_length=int(args.sequence_length),
        perturb_lags=tuple(args.perturb_lags),
        perturbation_epsilon=float(args.perturbation_epsilon),
        reconstruction_alpha=float(args.reconstruction_alpha),
        channel_subspace_energy=float(args.channel_subspace_energy),
        seed=int(args.seed),
    )
    candidate_features = CandidateFeatureConfig(
        instability_window=int(args.instability_window)
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=int(args.n_atoms),
        spacing_short_um=float(args.spacing_short_um),
        spacing_long_um=float(args.spacing_long_um),
        defect_edge=int(args.defect_edge),
        defect_offset_um=float(args.defect_offset_um),
        delta_center_rad_us=float(args.delta_center_rad_us),
        delta_span_rad_us=float(args.delta_span_rad_us),
        omega_base_rad_us=float(args.omega_base_rad_us),
        omega_mod_fraction=float(args.omega_mod_fraction),
        step_duration_us=float(args.step_duration_us),
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
    run_dir = run_input_sensitivity_rank_assay(
        fold_dir=fold_dir,
        close_panel_path=close_panel,
        results_root=args.out_root,
        config=config,
        candidate_features=candidate_features,
        reservoir=reservoir,
        ladder_geometry=geometry,
        interaction_scale=float(args.interaction_scale),
        run_id=str(args.run_id),
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
