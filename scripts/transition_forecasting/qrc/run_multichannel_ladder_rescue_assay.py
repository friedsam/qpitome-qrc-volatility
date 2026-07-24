from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.multichannel_ladder_rescue_assay import (
    MultichannelLadderRescueConfig,
    run_multichannel_ladder_rescue_assay,
)
from transition_forecasting.qrc.serial_multichannel_ladder import (
    MULTICHANNEL_CONDITIONS,
    MULTICHANNEL_REPRESENTATIONS,
    SerialMultichannelEncodingConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/"
    "archive_pre_controls_20260723_001856/purged_walk_forward_folds"
)
DEFAULT_SOURCE_SPACING_RUN = Path(
    "results/transition_forecasting/qrc/run_precontrol_ladder_spacing_assay/"
    "precontrol_spacing_8_9_10_002"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_multichannel_ladder_rescue_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the failed HAR-residual head with direct occurrence/path models "
            "fed by a three-channel serially encoded six-atom ladder."
        )
    )
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument(
        "--source-spacing-run", type=Path, default=DEFAULT_SOURCE_SPACING_RUN
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--target-lead", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=20260722)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=list(MULTICHANNEL_REPRESENTATIONS),
        default=list(MULTICHANNEL_REPRESENTATIONS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(MULTICHANNEL_CONDITIONS),
        default=list(MULTICHANNEL_CONDITIONS),
    )
    parser.add_argument(
        "--path-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0]
    )
    parser.add_argument("--gate-cs", type=float, nargs="+", default=[0.1, 1.0, 10.0])

    parser.add_argument("--n-atoms", type=int, default=6)
    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
    parser.add_argument("--step-duration-us", type=float, default=0.03)
    parser.add_argument(
        "--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    parser.add_argument("--shot-seed", type=int, default=20260721)

    parser.add_argument("--ladder-longitudinal-spacing-um", type=float, default=8.5)
    parser.add_argument("--ladder-row-spacing-um", type=float, default=9.0)
    parser.add_argument("--ladder-stagger-fraction", type=float, default=0.35)
    parser.add_argument("--ladder-bottom-spacing-scale", type=float, default=1.05)
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)
    parser.add_argument("--interaction-scale", type=float, default=1.25)

    parser.add_argument("--instability-window", type=int, default=5)
    parser.add_argument("--short-slope-window", type=int, default=5)
    parser.add_argument("--long-slope-window", type=int, default=20)
    parser.add_argument("--third-channel-delta-span-rad-us", type=float, default=3.0)
    parser.add_argument("--shuffle-seed", type=int, default=20260724)

    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultichannelLadderRescueConfig(
        folds=tuple(args.folds),
        leads=tuple(args.leads),
        target_lead=args.target_lead,
        max_per_class=args.max_per_class,
        selection_seed=args.selection_seed,
        path_alphas=tuple(args.path_alphas),
        gate_cs=tuple(args.gate_cs),
        representations=tuple(args.representations),
        conditions=tuple(args.conditions),
    )
    encoding = SerialMultichannelEncodingConfig(
        instability_window=args.instability_window,
        short_slope_window=args.short_slope_window,
        long_slope_window=args.long_slope_window,
        third_channel_delta_span_rad_us=args.third_channel_delta_span_rad_us,
        shuffle_seed=args.shuffle_seed,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=args.n_atoms,
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
    run_dir = run_multichannel_ladder_rescue_assay(
        fold_dir=args.fold_dir,
        source_spacing_run=args.source_spacing_run,
        results_root=args.out_root,
        config=config,
        encoding=encoding,
        reservoir=reservoir,
        geometry=geometry,
        interaction_scale=args.interaction_scale,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
