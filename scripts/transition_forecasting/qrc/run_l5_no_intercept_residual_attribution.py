from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.l5_no_intercept_residual_attribution import (
    L5NoInterceptResidualAttributionConfig,
    run_l5_no_intercept_residual_attribution,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import StaggeredLadderGeometryConfig

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_l5_no_intercept_residual_attribution"
)
_REQUIRED_FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
)


def validate_fold_dir(path: Path) -> Path:
    candidate = Path(path)
    missing = [name for name in _REQUIRED_FOLD_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"invalid fold directory {candidate}; missing {missing}. "
            "Pass the canonical purged walk-forward fold directory with --fold-dir."
        )
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether full palindrome QRC features add sample-dependent L5 "
            "information to a plain target-available causal HAR baseline, with no "
            "residual-head intercept or intercept surrogate."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--selection-folds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--row-permutations", type=int, default=256)
    parser.add_argument("--target-permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = L5NoInterceptResidualAttributionConfig(
        folds=tuple(args.folds),
        selection_folds=tuple(args.selection_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        ridge_alpha=float(args.ridge_alpha),
        row_permutations=int(args.row_permutations),
        target_permutations=int(args.target_permutations),
        seed=int(args.seed),
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.020,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seed,
    )
    run_dir = run_l5_no_intercept_residual_attribution(
        fold_dir=validate_fold_dir(args.fold_dir),
        results_root=args.out_root,
        config=config,
        candidate_features=CandidateFeatureConfig(),
        reservoir=reservoir,
        geometry=StaggeredLadderGeometryConfig(row_spacing_um=9.0),
        run_id=str(args.run_id),
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
