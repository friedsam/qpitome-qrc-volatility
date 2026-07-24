from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.bivariate_bilinear_slot_fraction_assay import (
    BivariateBilinearSlotFractionConfig,
    run_bivariate_bilinear_slot_fraction_assay,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_bivariate_bilinear_slot_fraction_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "At the 0.015 us bilinear near-pass, sweep the fraction of each timestep "
            "allocated to detuning versus transverse drive."
        )
    )
    parser.add_argument("--samples", type=int, default=640)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument(
        "--memory-delays", type=int, nargs="+", default=[1, 2, 4, 5, 8, 12]
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260724, 20260725, 20260726]
    )
    parser.add_argument("--step-duration-us", type=float, default=0.015)
    parser.add_argument(
        "--detuning-fractions",
        type=float,
        nargs="+",
        default=[0.40, 0.50, 0.60, 0.70, 0.80],
        help="Fractions of each timestep assigned to the pure-detuning slot.",
    )
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument("--drive-phase-rad", type=float, default=0.0)
    parser.add_argument("--permutations", type=int, default=32)

    parser.add_argument("--delta-center-rad-us", type=float, default=6.0)
    parser.add_argument("--delta-span-rad-us", type=float, default=4.0)
    parser.add_argument("--omega-base-rad-us", type=float, default=6.0)
    parser.add_argument("--omega-mod-fraction", type=float, default=0.60)
    parser.add_argument(
        "--probe-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )

    parser.add_argument("--ladder-longitudinal-spacing-um", type=float, default=8.5)
    parser.add_argument("--ladder-row-spacing-um", type=float, default=9.0)
    parser.add_argument("--ladder-stagger-fraction", type=float, default=0.35)
    parser.add_argument("--ladder-bottom-spacing-scale", type=float, default=1.05)
    parser.add_argument("--ladder-defect-site", type=int, default=4)
    parser.add_argument("--ladder-defect-dx-um", type=float, default=0.35)
    parser.add_argument("--ladder-defect-dy-um", type=float, default=-0.40)

    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BivariateBilinearSlotFractionConfig(
        samples=args.samples,
        sequence_length=args.sequence_length,
        memory_delays=tuple(args.memory_delays),
        alphas=tuple(args.alphas),
        seeds=tuple(args.seeds),
        step_duration_us=args.step_duration_us,
        detuning_fractions=tuple(args.detuning_fractions),
        interaction_scale=args.interaction_scale,
        drive_phase_rad=args.drive_phase_rad,
        permutations=args.permutations,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=args.delta_center_rad_us,
        delta_span_rad_us=args.delta_span_rad_us,
        omega_base_rad_us=args.omega_base_rad_us,
        omega_mod_fraction=args.omega_mod_fraction,
        step_duration_us=args.step_duration_us,
        probe_fractions=tuple(args.probe_fractions),
        shots=None,
        shot_seed=args.seeds[0],
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
    run_dir = run_bivariate_bilinear_slot_fraction_assay(
        results_root=args.out_root,
        config=config,
        reservoir=reservoir,
        geometry=geometry,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
