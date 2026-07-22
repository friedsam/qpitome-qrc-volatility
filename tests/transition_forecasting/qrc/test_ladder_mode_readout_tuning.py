from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.ladder_mode_readout_tools import (
    ArchitectureArchive,
    LadderModeReadoutConfig,
    build_readout_matrix,
    choose_global_lambda,
    ladder_mode_weights,
    select_candidate,
)


def _archive(occupation: np.ndarray) -> ArchitectureArchive:
    probes = (10, 20, 40)
    feature_names = tuple(
        f"probe_{probe}_occupation_site_{site}"
        for probe in probes
        for site in range(6)
    )
    features = np.concatenate(
        [occupation.copy() for _ in probes],
        axis=1,
    )
    rows = len(occupation)
    target = np.zeros((rows, 10), dtype=float)
    return ArchitectureArchive(
        feature_matrix=features,
        feature_names=feature_names,
        target_path=target,
        har_prediction_path=target.copy(),
        prequential_residual_path=target.copy(),
        prequential_residual_valid=np.ones(rows, dtype=bool),
        fold=np.ones(rows, dtype=int),
        sample_id=np.asarray([f"s{index}" for index in range(rows)]),
        fold_split=np.asarray(["train"] * rows),
        lead=np.ones(rows, dtype=int),
        label=np.zeros(rows, dtype=int),
        episode_id=np.asarray([f"e{index}" for index in range(rows)]),
        origin_date=np.asarray(
            [f"2026-01-{index + 1:02d}" for index in range(rows)]
        ),
        case=np.asarray(["ladder_ordered_1p00"] * rows),
        architecture=np.asarray(["ladder"] * rows),
        control=np.asarray(["ordered"] * rows),
        interaction_scale=np.ones(rows, dtype=float),
    )


def test_ladder_mode_weights_are_orthonormal() -> None:
    weights = ladder_mode_weights()
    assert weights.shape == (6, 6)
    np.testing.assert_allclose(
        weights.T @ weights,
        np.eye(6),
        atol=1e-12,
        rtol=0.0,
    )


def test_antisymmetric_modes_reject_equal_rows() -> None:
    occupation = np.asarray(
        [
            [0.1, 0.3, 0.7, 0.1, 0.3, 0.7],
            [0.7, 0.2, 0.4, 0.7, 0.2, 0.4],
        ]
    )
    matrix, names, transform = build_readout_matrix(
        _archive(occupation),
        "antisymmetric_modes",
    )
    assert transform == "direct"
    assert len(names) == 9
    np.testing.assert_allclose(matrix, 0.0, atol=1e-12, rtol=0.0)


def test_symmetric_modes_reject_opposite_rows() -> None:
    top = np.asarray([[0.1, 0.4, 0.8], [0.7, 0.2, 0.3]])
    occupation = np.concatenate([top, -top], axis=1)
    matrix, _, _ = build_readout_matrix(
        _archive(occupation),
        "symmetric_modes",
    )
    np.testing.assert_allclose(matrix, 0.0, atol=1e-12, rtol=0.0)


def test_occupation_family_preserves_three_probe_site_matrix() -> None:
    occupation = np.arange(24, dtype=float).reshape(4, 6)
    matrix, names, transform = build_readout_matrix(
        _archive(occupation),
        "occupation_pca4",
    )
    assert matrix.shape == (4, 18)
    assert len(names) == 18
    assert transform == "pca"


def test_global_lambda_prefers_zero_when_correction_is_harmful() -> None:
    rows = 12
    y = np.zeros((rows, 2), dtype=float)
    har = np.zeros_like(y)
    correction = np.ones_like(y)
    selected, payload = choose_global_lambda(
        correction,
        y=y,
        har=har,
        tune_mask=np.ones(rows, dtype=bool),
        lambdas=(0.0, 0.5, 1.0),
    )
    assert selected == 0.0
    assert payload["inner_qlike"] == 0.0


def test_selection_policies_separate_frozen_and_mode_candidates() -> None:
    frame = pd.DataFrame(
        [
            {
                "case": "ladder_ordered_0p75",
                "readout_family": "occupation_pca4",
                "inner_qlike": 0.9,
                "inner_rmse": 0.5,
                "selected_lambda": 1.0,
                "feature_width": 18,
                "interaction_scale": 0.75,
            },
            {
                "case": "ladder_ordered_1p25",
                "readout_family": "symmetric_modes",
                "inner_qlike": 0.8,
                "inner_rmse": 0.5,
                "selected_lambda": 1.0,
                "feature_width": 9,
                "interaction_scale": 1.25,
            },
        ]
    )
    assert (
        select_candidate(frame, policy="frozen")["readout_family"]
        == "occupation_pca4"
    )
    assert (
        select_candidate(frame, policy="modes")["readout_family"]
        == "symmetric_modes"
    )
    assert (
        select_candidate(frame, policy="full")["readout_family"]
        == "symmetric_modes"
    )


def test_default_config_keeps_alpha_fixed() -> None:
    config = LadderModeReadoutConfig()
    config.validate()
    assert config.ridge_alpha == 100.0
    assert config.pca_components == 4
