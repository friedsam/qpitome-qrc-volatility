from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.gated_l5_early_warning_assay import (
    GatedL5EarlyWarningConfig,
    conditional_paths,
    correction_from_candidate,
    fit_gate,
    gate_metrics,
    select_gate_configuration,
)


def test_gate_recovers_separable_transition_signal() -> None:
    rng = np.random.default_rng(20260724)
    rows = 240
    labels = np.tile([0, 1], rows // 2)
    signal = labels.astype(float) + rng.normal(scale=0.20, size=rows)
    matrix = np.column_stack([signal, rng.normal(size=(rows, 5))])
    fit = np.zeros(rows, dtype=bool)
    fit[:160] = True

    gate = fit_gate(
        matrix,
        labels,
        fit,
        c_value=1.0,
        positive_class_weight=1.0,
    )
    probabilities = gate.predict_probability(matrix[~fit])
    metrics = gate_metrics(labels[~fit], probabilities, 0.5)

    assert metrics["recall"] > 0.90
    assert metrics["false_alert_rate"] < 0.10
    assert metrics["average_precision"] > 0.95


def test_gate_selection_respects_false_alert_budget_when_feasible() -> None:
    rng = np.random.default_rng(17)
    rows = 300
    labels = np.tile([0, 1], rows // 2)
    matrix = np.column_stack(
        [1.5 * labels + rng.normal(scale=0.6, size=rows), rng.normal(size=(rows, 5))]
    )
    fit = np.zeros(rows, dtype=bool)
    tune = np.zeros(rows, dtype=bool)
    fit[:200] = True
    tune[200:] = True
    config = GatedL5EarlyWarningConfig(
        gate_cs=(0.1, 1.0, 10.0),
        positive_class_weights=(1.0, 2.0),
        thresholds=(0.3, 0.5, 0.7),
        false_alert_budget=0.25,
    )

    selected, candidates = select_gate_configuration(matrix, labels, fit, tune, config)

    assert not candidates.empty
    assert selected["false_alert_rate"] <= 0.25
    assert selected["recall"] > 0.50


def test_conditional_gate_suppresses_controls_and_activates_transition_path() -> None:
    probabilities = np.array([0.1, 0.8])
    path = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.4, 0.5, 0.5])

    correction = correction_from_candidate(
        probabilities,
        path,
        threshold=0.5,
        activation="hard",
        scale=1.0,
    )

    np.testing.assert_allclose(correction[0], 0.0)
    np.testing.assert_allclose(correction[1], path)


def test_conditional_paths_build_late_positive_candidate_from_training_transitions() -> None:
    residuals = np.array(
        [
            [-0.2, -0.1, 0.0, 0.1, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    labels = np.array([1, 1, 0])
    fit = np.ones(3, dtype=bool)
    current = np.zeros_like(residuals)

    paths = conditional_paths(residuals, labels, fit, current, split_horizon=4)

    np.testing.assert_allclose(paths["late_positive_mean"][:4], 0.0)
    assert np.all(paths["late_positive_mean"][4:] > 0.0)
    np.testing.assert_allclose(paths["incumbent_total"], current)
