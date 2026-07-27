#!/usr/bin/env python3
"""Rerun and verify the canonical 63-feature financial QRC simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    PalindromeRealTaskConfig,
    run_palindrome_real_task_relevance_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import StaggeredLadderGeometryConfig

DEFAULT_FOLD_DIR = Path(
    "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/transition_forecasting/qrc/case151_simulation/run"
)
DEFAULT_EXPECTED = Path("config/case151/expected_metrics.json")


def canonical_objects() -> dict[str, Any]:
    return {
        "config": PalindromeRealTaskConfig(
            folds=(4, 5, 6, 7, 8),
            lead=5,
            max_per_class=24,
            sequence_length=40,
            representations=("level_only", "level_instability"),
            temporal_conditions=(
                "ordered",
                "shuffled_keep_endpoint",
                "recent_tail",
                "static_endpoint",
            ),
            recent_tail_steps=5,
            ridge_alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
            correction_lambdas=(0.0, 0.25, 0.5, 1.0),
            inner_holdout_fraction=0.25,
            prequential_blocks=5,
            selection_seed=20260725,
            level_channel_name="log_volatility_level",
            fallback_level_channel=0,
            schedule_name="crossover_Ahalf_B_Ahalf",
        ),
        "candidate_features": CandidateFeatureConfig(
            instability_window=5,
            shock_window=5,
            short_slope_window=5,
            long_slope_window=20,
            q_low=0.01,
            q_high=0.99,
        ),
        "reservoir": TemporalRydbergChainConfig(
            n_atoms=6,
            spacing_short_um=8.5,
            spacing_long_um=10.0,
            defect_edge=1,
            defect_offset_um=0.6,
            c6_rad_um6_per_us=5_420_000.0,
            delta_center_rad_us=6.0,
            delta_span_rad_us=4.0,
            omega_base_rad_us=6.0,
            omega_mod_fraction=0.6,
            step_duration_us=0.02,
            probe_fractions=(0.25, 0.5, 1.0),
            max_phase_per_substep=0.25,
            max_substeps_per_step=512,
            shots=None,
            shot_seed=20260725,
        ),
        "geometry": StaggeredLadderGeometryConfig(
            longitudinal_spacing_um=8.5,
            row_spacing_um=9.0,
            stagger_fraction=0.35,
            bottom_spacing_scale=1.05,
            defect_site=4,
            defect_dx_um=0.35,
            defect_dy_um=-0.40,
        ),
        "interaction_scale": 1.25,
        "drive_phase_rad": 0.0,
    }


def _selected_row(frame: pd.DataFrame, *, interactions: str, scope: str) -> pd.Series:
    model = "palindrome_ordered_on" if interactions == "on" else "palindrome_ordered_off"
    rows = frame.loc[
        frame["model"].eq(model)
        & frame["representation"].eq("level_instability")
        & frame["condition"].eq("ordered")
        & frame["interactions"].eq(interactions)
        & frame["scope"].eq(scope)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one row for model={model}, interactions={interactions}, scope={scope}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def _assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"metric mismatch for {name}: observed={observed:.16g}, "
            f"expected={expected:.16g}, tolerance={tolerance:.3g}"
        )


def verify_run(run_dir: Path, expected_path: Path, tolerance: float = 5e-10) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    pooled = pd.read_csv(run_dir / "pooled_metrics.csv")
    rows = {
        "selected_path": _selected_row(pooled, interactions="on", scope="path"),
        "selected_transition": _selected_row(pooled, interactions="on", scope="transition"),
        "interaction_off_path": _selected_row(pooled, interactions="off", scope="path"),
    }
    for group, row in rows.items():
        for metric, value in expected["expected"][group].items():
            _assert_close(f"{group}.{metric}", float(row[metric]), float(value), tolerance)

    selections = pd.read_csv(run_dir / "readout_selections.csv")
    fold8 = selections.loc[
        selections["fold"].eq(8)
        & selections["model"].eq("palindrome_ordered_on")
        & selections["representation"].eq("level_instability")
        & selections["condition"].eq("ordered")
        & selections["interactions"].eq("on")
    ]
    if len(fold8) != 1:
        raise RuntimeError(f"expected one fold-8 selected readout row, found {len(fold8)}")
    _assert_close("fold8.selected_alpha", float(fold8.iloc[0]["selected_alpha"]), 0.1, 1e-12)
    _assert_close("fold8.selected_lambda", float(fold8.iloc[0]["selected_lambda"]), 0.25, 1e-12)

    diagnostics = pd.read_csv(run_dir / "feature_diagnostics.csv")
    selected = diagnostics.loc[
        diagnostics["model"].eq("palindrome_ordered_on")
        & diagnostics["representation"].eq("level_instability")
        & diagnostics["condition"].eq("ordered")
        & diagnostics["interactions"].eq("on")
    ]
    if selected.empty or not selected["feature_width"].eq(63).all():
        raise RuntimeError("canonical selected QRC must use exactly 63 occupation/pair features")

    report = {
        "schema_version": 1,
        "status": "verified",
        "run_id": run_dir.name,
        "canonical_commit": expected["canonical_commit"],
        "source_run": expected["source_run"],
        "feature_bank": "occupation_pair_raw",
        "feature_width": 63,
        "fold8_selected_alpha": 0.1,
        "fold8_selected_lambda": 0.25,
        "test_rows_used": 0,
        "metric_tolerance": tolerance,
    }
    (run_dir / "case151_reproduction_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    run_dir = args.output_root / args.run_id
    if not args.verify_existing:
        objects = canonical_objects()
        run_dir = run_palindrome_real_task_relevance_assay(
            fold_dir=args.fold_dir,
            results_root=args.output_root,
            config=objects["config"],
            candidate_features=objects["candidate_features"],
            reservoir=objects["reservoir"],
            geometry=objects["geometry"],
            interaction_scale=float(objects["interaction_scale"]),
            drive_phase_rad=float(objects["drive_phase_rad"]),
            run_id=args.run_id,
        )
    report = verify_run(run_dir, args.expected)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
