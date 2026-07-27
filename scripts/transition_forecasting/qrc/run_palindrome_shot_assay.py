from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from transition_forecasting.qrc.palindrome_shot_assay import (
    PalindromeShotAssayConfig,
    run_palindrome_shot_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import StaggeredLadderGeometryConfig

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/palindrome_shot_assay"
)

SHOT_OUTPUTS_BEFORE_ALIAS = (
    "params.json",
    "summary.json",
    "shot_metrics.csv",
    "shot_summary.csv",
    "direction_metrics.csv",
    "predictions.csv.gz",
    "retained_samples.csv",
    "frozen_readout.npz",
    "features/exact_reference.npz",
    "plots/warning_gap_preservation_vs_shots.png",
    "plots/correction_correlation_vs_shots.png",
    "plots/transition_control_gap_vs_shots.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure finite-shot preservation of the final six-atom palindrome's "
            "transition-versus-control correction gap."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=20260726)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--correction-lambda", type=float, default=1.0)
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument(
        "--shot-counts",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000, 5000],
    )
    parser.add_argument(
        "--measurement-seeds",
        type=int,
        nargs="+",
        default=[20260722, 20260723, 20260724, 20260725, 20260726],
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


def publish_exact_reference_alias(run_dir: Path) -> Path:
    """Publish the validator-facing exact reference without changing source data."""

    run_dir = Path(run_dir)
    source = run_dir / "features" / "exact_reference.npz"
    destination = run_dir / "exact_reference.npz"
    if not source.is_file():
        raise FileNotFoundError(f"finite-shot exact reference is missing: {source}")
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(
                "finite-shot exact-reference alias differs from the canonical feature copy"
            )
        return destination
    shutil.copy2(source, destination)
    return destination


def repair_existing_shot_run(run_dir: Path) -> Path:
    """Repair only a completed shot run whose publishing alias is absent."""

    run_dir = Path(run_dir)
    missing = [
        relative
        for relative in SHOT_OUTPUTS_BEFORE_ALIAS
        if not (run_dir / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "existing finite-shot run is incomplete and cannot be repaired in place: "
            + ", ".join(missing)
        )
    publish_exact_reference_alias(run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    expected_run_dir = args.out_root / args.run_id
    if expected_run_dir.exists():
        run_dir = repair_existing_shot_run(expected_run_dir)
        print(f"REPAIRED {run_dir}")
        return

    assay = PalindromeShotAssayConfig(
        fold=args.fold,
        lead=args.lead,
        sequence_length=args.sequence_length,
        max_per_class=args.max_per_class,
        selection_seed=args.selection_seed,
        ridge_alpha=args.ridge_alpha,
        correction_lambda=args.correction_lambda,
        prequential_blocks=args.prequential_blocks,
        shot_counts=tuple(args.shot_counts),
        measurement_seeds=tuple(args.measurement_seeds),
    )
    run_dir = run_palindrome_shot_assay(
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
    publish_exact_reference_alias(run_dir)
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
