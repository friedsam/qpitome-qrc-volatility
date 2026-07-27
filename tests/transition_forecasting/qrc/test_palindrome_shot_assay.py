from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.transition_forecasting.qrc.run_palindrome_shot_assay import (
    frozen_geometry,
    frozen_reservoir,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    sample_probe_probabilities,
)
from transition_forecasting.qrc.palindrome_shot_assay import (
    PalindromeShotAssayConfig,
    _aggregate_shots,
    transition_control_direction_rows,
)


def _toy_probabilities() -> np.ndarray:
    probabilities = np.zeros((3, 2, 64), dtype=float)
    probabilities[0, :, 0] = 0.75
    probabilities[0, :, 1] = 0.25
    probabilities[1, :, 2] = 0.35
    probabilities[1, :, 3] = 0.65
    probabilities[2, :, 4] = 0.55
    probabilities[2, :, 5] = 0.45
    return probabilities


def test_shot_sampling_is_invariant_to_row_order() -> None:
    exact = _toy_probabilities()
    sample_ids = np.asarray(["sample-a", "sample-b", "sample-c"])
    direct = sample_probe_probabilities(
        exact,
        shots=500,
        base_seed=20260726,
        fold=5,
        sample_ids=sample_ids,
        probe_steps=(10, 40),
    )

    permutation = np.asarray([2, 0, 1])
    permuted = sample_probe_probabilities(
        exact[permutation],
        shots=500,
        base_seed=20260726,
        fold=5,
        sample_ids=sample_ids[permutation],
        probe_steps=(10, 40),
    )
    inverse = np.argsort(permutation)
    assert np.array_equal(direct, permuted[inverse])
    assert np.allclose(direct.sum(axis=2), 1.0)


def test_direction_rows_preserve_transition_control_gap() -> None:
    labels = np.asarray([0, 0, 1, 1])
    exact = np.asarray(
        [
            [-0.20, -0.10],
            [-0.15, -0.05],
            [0.35, 0.45],
            [0.40, 0.50],
        ]
    )
    finite = exact * 0.8
    rows = transition_control_direction_rows(
        labels=labels,
        exact_correction=exact,
        candidate_correction=finite,
        shot_count=1000,
        measurement_seed=7,
    )
    path = next(row for row in rows if row["scope"] == "path_mean")
    assert path["exact_transition_minus_control_gap"] > 0
    assert path["transition_minus_control_gap"] > 0
    assert path["gap_direction_preserved"] is True
    assert path["gap_relative_to_exact"] == pytest.approx(0.8)
    assert path["transition_sign_agreement"] == pytest.approx(1.0)
    assert path["control_sign_agreement"] == pytest.approx(1.0)


def test_direction_rows_detect_reversed_warning_gap() -> None:
    labels = np.asarray([0, 0, 1, 1])
    exact = np.asarray([[-0.2], [-0.1], [0.3], [0.4]])
    reversed_candidate = -exact
    rows = transition_control_direction_rows(
        labels=labels,
        exact_correction=exact,
        candidate_correction=reversed_candidate,
        shot_count=100,
        measurement_seed=9,
    )
    path = next(row for row in rows if row["scope"] == "path_mean")
    assert path["gap_direction_preserved"] is False
    assert path["gap_relative_to_exact"] == pytest.approx(-1.0)


def test_shot_summary_requires_all_seed_gap_preservation() -> None:
    shot_metrics = pd.DataFrame(
        {
            "shot_count": [100, 100, 1000, 1000],
            "measurement_seed": [1, 2, 1, 2],
            "correction_correlation_vs_exact": [0.8, 0.9, 0.98, 0.99],
            "relative_feature_mae": [0.2, 0.1, 0.02, 0.01],
            "qlike_delta_vs_exact": [0.01, -0.01, 0.001, -0.001],
            "rmse_delta_vs_exact": [0.02, -0.02, 0.002, -0.002],
        }
    )
    direction = pd.DataFrame(
        {
            "shot_count": [100, 100, 1000, 1000],
            "measurement_seed": [1, 2, 1, 2],
            "scope": ["path_mean"] * 4,
            "gap_direction_preserved": [True, False, True, True],
            "gap_relative_to_exact": [0.7, -0.2, 0.95, 1.05],
            "gap_absolute_error": [0.3, 1.2, 0.05, 0.05],
            "transition_sign_agreement": [0.7, 0.5, 0.95, 1.0],
            "control_sign_agreement": [0.8, 0.6, 1.0, 0.95],
        }
    )
    summary = _aggregate_shots(shot_metrics, direction).set_index("shot_count")
    assert bool(summary.loc[100, "all_seed_gap_direction_preserved"]) is False
    assert bool(summary.loc[1000, "all_seed_gap_direction_preserved"]) is True
    assert summary.loc[1000, "gap_direction_preservation_rate"] == pytest.approx(1.0)


def test_frozen_shot_contract() -> None:
    assay = PalindromeShotAssayConfig()
    assay.validate()
    reservoir = frozen_reservoir()
    geometry = frozen_geometry()
    assert reservoir.n_atoms == 6
    assert reservoir.shots is None
    assert reservoir.step_duration_us == pytest.approx(0.02)
    assert reservoir.probe_fractions == (0.25, 0.5, 1.0)
    assert reservoir.omega_mod_fraction == pytest.approx(0.60)
    assert geometry.defect_site == 4
    assert assay.feature_bank == "six_mode_density_curvature"
    assert assay.ridge_alpha == pytest.approx(100.0)
    assert assay.shot_counts == (100, 250, 500, 1000, 2000, 5000)


def test_shot_config_rejects_duplicate_counts_and_seeds() -> None:
    with pytest.raises(ValueError, match="shot_counts must be unique"):
        PalindromeShotAssayConfig(shot_counts=(100, 100)).validate()
    with pytest.raises(ValueError, match="measurement_seeds must be unique"):
        PalindromeShotAssayConfig(measurement_seeds=(1, 1)).validate()
