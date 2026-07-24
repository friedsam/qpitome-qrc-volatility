from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.multichannel_ladder_rescue_assay import (
    _causal_har_design,
    fit_select_direct_path,
    fit_select_occurrence,
)


def test_causal_har_design_uses_prequential_training_predictions() -> None:
    y = np.arange(30, dtype=float).reshape(3, 10)
    har = np.full((3, 10), -1.0)
    residuals = np.full((3, 10), 2.0)
    mask = np.array([True, False, True])

    design = _causal_har_design(y, har, residuals, mask)

    np.testing.assert_allclose(design[mask], y[mask] - 2.0)
    np.testing.assert_allclose(design[~mask], har[~mask])


def test_direct_path_selection_recovers_informative_features() -> None:
    rng = np.random.default_rng(5)
    rows = 240
    matrix = rng.normal(size=(rows, 6))
    weights = rng.normal(size=(6, 10))
    y = matrix @ weights + rng.normal(scale=0.03, size=(rows, 10))
    fit = np.zeros(rows, dtype=bool)
    tune = np.zeros(rows, dtype=bool)
    full = np.zeros(rows, dtype=bool)
    fit[:120] = True
    tune[120:180] = True
    full[:180] = True

    prediction, selected, candidates = fit_select_direct_path(
        matrix,
        y,
        fit,
        tune,
        full,
        alphas=(0.1, 1.0, 10.0, 100.0),
    )

    assert prediction.shape == y.shape
    assert selected["alpha"] in {0.1, 1.0, 10.0, 100.0}
    assert len(candidates) == 4
    assert np.sqrt(np.mean((prediction[180:] - y[180:]) ** 2)) < 0.2


def test_occurrence_selection_recovers_separable_signal() -> None:
    rng = np.random.default_rng(9)
    rows = 300
    labels = np.tile([0, 1], rows // 2)
    matrix = np.column_stack(
        [2.0 * labels + rng.normal(scale=0.35, size=rows), rng.normal(size=(rows, 5))]
    )
    fit = np.zeros(rows, dtype=bool)
    tune = np.zeros(rows, dtype=bool)
    full = np.zeros(rows, dtype=bool)
    fit[:160] = True
    tune[160:220] = True
    full[:220] = True

    probability, selected, candidates = fit_select_occurrence(
        matrix,
        labels,
        fit,
        tune,
        full,
        c_values=(0.1, 1.0, 10.0),
    )

    assert selected["c_value"] in {0.1, 1.0, 10.0}
    assert len(candidates) == 3
    assert probability[220:][labels[220:] == 1].mean() > 0.8
    assert probability[220:][labels[220:] == 0].mean() < 0.2
