from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.final_sparse_qlike_assay import (
    DENSITY_CURVATURE_INDICES,
    FinalSparseQlikeAssayConfig,
    qlike_loss,
)


def test_density_curvature_indices_remove_only_gradients() -> None:
    assert DENSITY_CURVATURE_INDICES == (0, 2, 3, 5, 6, 8)


def test_qlike_loss_is_zero_for_exact_forecast() -> None:
    values = np.asarray([[-4.0, -3.0], [-2.0, -1.0]])
    assert np.isclose(qlike_loss(values, values), 0.0)


def test_qlike_penalizes_underforecast_more_than_equal_overforecast() -> None:
    observed = np.zeros((2, 1))
    under = np.full((2, 1), -0.5)
    over = np.full((2, 1), 0.5)
    assert qlike_loss(observed, under) > qlike_loss(observed, over)


def test_final_assay_defaults_are_frozen() -> None:
    config = FinalSparseQlikeAssayConfig()
    config.validate()
    assert config.folds == tuple(range(1, 9))
    assert config.shot_count == 1000
    assert len(config.shot_seeds) == 5
    assert config.seed == 20260722
