"""Focused tests for the exploratory day-5 ESN probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "modeling"
    / "run_cross_market_day5_esn_probe.py"
)
SPEC = importlib.util.spec_from_file_location("day5_esn_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_logit_sigmoid_round_trip() -> None:
    probabilities = np.array([1e-4, 0.1, 0.5, 0.9, 1 - 1e-4])
    reconstructed = probe.sigmoid(probe.logit(probabilities))
    np.testing.assert_allclose(reconstructed, probabilities, atol=1e-12, rtol=0)


def test_sequence_tensor_preserves_day_and_channel_order() -> None:
    rows = 3
    data: dict[str, np.ndarray] = {}
    for day in range(1, 6):
        for channel_index, channel in enumerate(probe.CHANNELS):
            data[f"{channel}_d{day}"] = np.arange(rows) + 100 * day + channel_index
    frame = pd.DataFrame(data)

    tensor = probe.make_sequence_tensor(frame)

    assert tensor.shape == (rows, 5, len(probe.CHANNELS))
    for row in range(rows):
        for day in range(1, 6):
            for channel_index in range(len(probe.CHANNELS)):
                assert tensor[row, day - 1, channel_index] == (
                    row + 100 * day + channel_index
                )


def test_offset_logistic_leaves_zero_correction_when_offset_is_exact() -> None:
    # Four balanced groups with exact empirical probabilities supplied as offsets.
    probabilities = np.repeat([0.25, 0.50, 0.75], 40)
    y = np.concatenate(
        [
            np.tile([0, 0, 0, 1], 10),
            np.tile([0, 1], 20),
            np.tile([0, 1, 1, 1], 10),
        ]
    )
    X = np.zeros((len(y), 4))

    beta, intercept = probe.fit_offset_logistic(
        X=X,
        y=y,
        offset=probe.logit(probabilities),
        l2=10.0,
    )

    np.testing.assert_allclose(beta, 0.0, atol=1e-8, rtol=0)
    assert abs(intercept) < 1e-8


def test_offset_logistic_finds_incremental_signal() -> None:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(500, 1))
    offset = np.zeros(500)
    true_beta = 0.8
    p = probe.sigmoid(true_beta * X[:, 0])
    y = rng.binomial(1, p)

    beta, _ = probe.fit_offset_logistic(X, y, offset, l2=1.0)

    assert beta.shape == (1,)
    assert beta[0] > 0.4


def test_honest_d1_predictions_use_only_rows_before_cluster_start(monkeypatch) -> None:
    n = 36
    dates = pd.date_range("1980-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "landmark_date": dates,
            "cluster_start": dates,
            "y_recovery": np.tile([0, 1], n // 2),
            **{column: np.linspace(0.0, 1.0, n) for column in probe.D1},
        }
    )

    observed_training_maxima: list[pd.Timestamp] = []

    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    def fake_fit_d1(train: pd.DataFrame):
        observed_training_maxima.append(train["landmark_date"].max())
        return FakeModel()

    monkeypatch.setattr(probe, "fit_d1", fake_fit_d1)
    predictions = probe.compute_honest_d1_predictions(frame)

    assert np.isnan(predictions[: probe.MIN_TRAIN]).all()
    assert np.isfinite(predictions[probe.MIN_TRAIN :]).all()
    for row_index, maximum_training_date in zip(
        range(probe.MIN_TRAIN, n), observed_training_maxima
    ):
        assert maximum_training_date < frame.loc[row_index, "cluster_start"]
