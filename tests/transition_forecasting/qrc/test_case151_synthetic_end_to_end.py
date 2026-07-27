from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    PalindromeRealTaskConfig,
    run_palindrome_real_task_relevance_assay,
)
from transition_forecasting.qrc.representation_candidates import CandidateFeatureConfig
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _write_synthetic_folds(root: Path) -> Path:
    rng = np.random.default_rng(20260726)
    splits = ["train"] * 32 + ["val"] * 12 + ["test"] * 2
    labels = ([0, 1] * 16) + ([0, 1] * 6) + [0, 1]
    rows = len(splits)
    dates = pd.date_range("2010-01-01", periods=rows, freq="D", tz="UTC")
    base = np.linspace(-0.35, 0.45, rows)
    windows = np.empty((rows, 8, 1), dtype=float)
    for row in range(rows):
        trend = np.linspace(base[row] - 0.08, base[row], 8)
        windows[row, :, 0] = trend + rng.normal(scale=0.01, size=8)

    manifest = pd.DataFrame(
        {
            "sample_id": [f"synthetic_{index:03d}" for index in range(rows)],
            "label": labels,
            "episode_id": [f"episode_{index:03d}" for index in range(rows)],
            "origin_date": dates.astype(str),
            "lead": np.full(rows, 5, dtype=int),
            "fold": np.full(rows, 4, dtype=int),
            "fold_split": splits,
            "level": base,
            "mean5": base - 0.015,
            "mean20": base - 0.035,
        }
    )
    for horizon, column in enumerate(TARGET_COLUMNS, start=1):
        manifest[column] = (
            0.80 * base
            + 0.025 * np.asarray(labels)
            + 0.004 * horizon
            + rng.normal(scale=0.01, size=rows)
        )

    root.mkdir(parents=True)
    manifest.to_csv(root / "rematched_rolling_manifest.csv", index=False)
    np.savez_compressed(
        root / "rematched_rolling_tensors.npz",
        X=windows,
        sample_id=manifest["sample_id"].to_numpy(dtype="U32"),
        fold=manifest["fold"].to_numpy(dtype=int),
        fold_split=manifest["fold_split"].to_numpy(dtype="U8"),
        valid=np.ones(rows, dtype=bool),
        channel_names=np.asarray(["log_volatility_level"], dtype="U32"),
    )
    return root


def test_case151_assay_runs_end_to_end_on_synthetic_chronological_folds(
    tmp_path: Path,
) -> None:
    fold_dir = _write_synthetic_folds(tmp_path / "folds")
    output = run_palindrome_real_task_relevance_assay(
        fold_dir=fold_dir,
        results_root=tmp_path / "results",
        config=PalindromeRealTaskConfig(
            folds=(4,),
            lead=5,
            max_per_class=16,
            sequence_length=8,
            representations=("level_instability",),
            temporal_conditions=("ordered",),
            ridge_alphas=(0.1, 1.0),
            correction_lambdas=(0.0, 0.25),
            prequential_blocks=5,
        ),
        candidate_features=CandidateFeatureConfig(instability_window=3),
        reservoir=TemporalRydbergChainConfig(
            n_atoms=6,
            omega_mod_fraction=0.60,
            step_duration_us=0.02,
            probe_fractions=(0.5, 1.0),
            shots=None,
        ),
        geometry=StaggeredLadderGeometryConfig(),
        interaction_scale=1.25,
        drive_phase_rad=0.0,
        run_id="synthetic_case151",
    )

    expected = {
        "params.json",
        "prediction_cells.csv.gz",
        "fold_metrics.csv",
        "pooled_metrics.csv",
        "readout_selections.csv",
        "readout_candidates.csv.gz",
        "feature_diagnostics.csv",
        "baseline_health.csv",
        "channel_scalers.csv",
        "simulation_metadata.csv",
        "summary.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    cells = pd.read_csv(output / "prediction_cells.csv.gz")
    assert not cells.empty
    assert set(cells["fold"]) == {4}
    assert set(cells["lead"]) == {5}
    assert cells["sample_id"].str.startswith("synthetic_").all()
    selections = pd.read_csv(output / "readout_selections.csv")
    assert selections["fit_intercept"].eq(False).all()
    assert set(selections["selected_alpha"]).issubset({0.1, 1.0})
    assert set(selections["selected_lambda"]).issubset({0.0, 0.25})
