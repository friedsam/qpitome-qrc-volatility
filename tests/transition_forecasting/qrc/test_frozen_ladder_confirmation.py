from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    FrozenLadderConfirmationConfig,
    confirmation_gate,
    occupation_matrix,
    symmetric_ladder_mode_matrix,
)


def _feature_names() -> tuple[str, ...]:
    return tuple(
        f"probe_{probe}_occupation_site_{site}"
        for probe in (10, 20, 40)
        for site in range(6)
    )


def test_confirmation_config_excludes_development_folds() -> None:
    FrozenLadderConfirmationConfig().validate()
    with pytest.raises(ValueError, match="exclude development folds"):
        FrozenLadderConfirmationConfig(folds=(3, 4)).validate()


def test_occupation_matrix_preserves_probe_and_site_order() -> None:
    features = np.arange(36, dtype=float).reshape(2, 18)
    matrix, names = occupation_matrix(features, _feature_names(), (10, 20, 40))
    np.testing.assert_array_equal(matrix, features)
    assert names == _feature_names()


def test_symmetric_mode_matrix_has_three_modes_per_probe() -> None:
    features = np.arange(54, dtype=float).reshape(3, 18)
    matrix, names = symmetric_ladder_mode_matrix(
        features,
        _feature_names(),
        (10, 20, 40),
    )
    assert matrix.shape == (3, 9)
    assert len(names) == 9
    assert all("symmetric_" in name for name in names)


def test_symmetric_modes_ignore_antisymmetric_row_difference() -> None:
    base = np.zeros((1, 18), dtype=float)
    perturbed = base.copy()
    perturbed[0, :3] = 1.0
    perturbed[0, 3:6] = -1.0
    base_modes, _ = symmetric_ladder_mode_matrix(
        base,
        _feature_names(),
        (10, 20, 40),
    )
    perturbed_modes, _ = symmetric_ladder_mode_matrix(
        perturbed,
        _feature_names(),
        (10, 20, 40),
    )
    np.testing.assert_allclose(base_modes, perturbed_modes, atol=1e-12)


def test_confirmation_gate_requires_all_predeclared_criteria() -> None:
    pooled = pd.DataFrame(
        [
            {"model_name": "har", "qlike": 1.0, "rmse": 0.50},
            {
                "model_name": "frozen_ladder_symmetric_1p25",
                "qlike": 0.95,
                "rmse": 0.505,
            },
        ]
    )
    fold_rows = []
    for fold, har, ladder in zip(
        (4, 5, 6, 7, 8),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (0.9, 0.9, 0.9, 1.1, 1.1),
        strict=True,
    ):
        fold_rows.extend(
            [
                {"fold": fold, "model_name": "har", "qlike": har},
                {
                    "fold": fold,
                    "model_name": "frozen_ladder_symmetric_1p25",
                    "qlike": ladder,
                },
            ]
        )
    result = confirmation_gate(pooled, pd.DataFrame(fold_rows))
    assert result["passed"] is True
    assert result["folds_improving_or_tying_har"] == 3
