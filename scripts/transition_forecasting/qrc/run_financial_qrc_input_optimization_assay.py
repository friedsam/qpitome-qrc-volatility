from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.financial_qrc_input_optimization_assay import (
    FinancialQRCInputOptimizationConfig,
    run_financial_qrc_input_optimization_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_PANEL = Path(
    "data/fallback/transition_forecasting/"
    "global_stock_indices_historical_data/all_indices_data.csv"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/"
    "run_financial_qrc_input_optimization_assay"
)
_REQUIRED_FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
)


def validate_inputs(fold_dir: Path, panel: Path) -> tuple[Path, Path]:
    folds = Path(fold_dir)
    missing = [name for name in _REQUIRED_FOLD_FILES if not (folds / name).is_file()]
    if missing:
        raise FileNotFoundError(f"invalid fold directory {folds}; missing {missing}")
    market = Path(panel)
    if not market.is_file():
        raise FileNotFoundError(f"missing OHLC panel: {market}")
    return folds, market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded L5 QRC input/history optimization around the successful "
            "incumbent simultaneous six-atom reservoir."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--prequential-blocks", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--selection-seed", type=int, default=20260721)
    parser.add_argument("--instability-window", type=int, default=5)
    parser.add_argument("--step-duration-us", type=float, default=0.03)
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument(
        "--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
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
    fold_dir, panel = validate_inputs(args.fold_dir, args.panel)
    config = FinancialQRCInputOptimizationConfig(
        folds=tuple(args.folds),
        max_per_class=int(args.max_per_class),
        windows=tuple(args.windows),
        prequential_blocks=int(args.prequential_blocks),
        ridge_alpha=float(args.ridge_alpha),
        selection_seed=int(args.selection_seed),
        step_duration_us=float(args.step_duration_us),
        interaction_scale=float(args.interaction_scale),
        probe_fractions=tuple(args.probe_fractions),
    )
    candidate_features = CandidateFeatureConfig(
        instability_window=int(args.instability_window)
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=float(args.delta_center_rad_us),
        delta_span_rad_us=float(args.delta_span_rad_us),
        omega_base_rad_us=float(args.omega_base_rad_us),
        omega_mod_fraction=float(args.omega_mod_fraction),
        step_duration_us=float(args.step_duration_us),
        probe_fractions=tuple(args.probe_fractions),
        shots=None,
        shot_seed=int(args.selection_seed),
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
    output = run_financial_qrc_input_optimization_assay(
        fold_dir=fold_dir,
        panel_path=panel,
        results_root=args.out_root,
        config=config,
        candidate_features=candidate_features,
        reservoir=reservoir,
        geometry=geometry,
        run_id=str(args.run_id),
    )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
