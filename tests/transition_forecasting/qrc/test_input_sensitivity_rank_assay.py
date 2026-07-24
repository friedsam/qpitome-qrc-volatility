from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.input_sensitivity_rank_assay import (
    build_perturbation_batch,
    channel_novelty_fraction,
    downside_return_windows,
    finite_difference_sensitivity,
    load_close_panel,
    matched_channel_cosines,
    parse_ticker,
)


def test_parse_ticker_from_transition_sample_id() -> None:
    assert parse_ticker("episode_^GSPC_data_L5_control_001") == "^GSPC"
    assert parse_ticker("prefix_NDX_data_L1_transition") == "NDX"


def test_downside_windows_are_causal_and_end_at_origin(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")
    close = np.asarray([100.0, 102.0, 101.0, 103.0, 99.0, 98.0, 120.0, 121.0])
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "date": dates.astype(str),
            "close": close,
            "ticker": "^GSPC",
        }
    ).to_csv(panel_path, index=False)
    panel = load_close_panel(panel_path)
    frame = pd.DataFrame(
        {
            "sample_id": ["episode_^GSPC_data_L5_control_001"],
            "origin_date": [dates[5].isoformat()],
        }
    )

    windows, valid = downside_return_windows(frame, panel, sequence_length=4)

    expected_history = close[1:6]
    expected = np.minimum(np.diff(np.log(expected_history)), 0.0)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(windows[0], expected)
    # The large future jump at index 6 must not enter the origin-date window.
    assert windows[0, -1] < 0.0


def test_finite_difference_recovers_identity_feature_sensitivity() -> None:
    encoded = np.asarray(
        [
            [[-0.4, 0.2], [-0.2, 0.1], [0.0, -0.1], [0.3, -0.2]],
            [[0.5, -0.4], [0.2, -0.2], [-0.1, 0.2], [-0.3, 0.4]],
        ],
        dtype=float,
    )
    lags = (0, 2)
    batch, blocks = build_perturbation_batch(encoded, lags=lags, epsilon=0.05)
    identity_features = batch.reshape(len(batch), -1)

    sensitivity = finite_difference_sensitivity(
        identity_features,
        samples=len(encoded),
        lags=lags,
        blocks=blocks,
    )

    assert sensitivity.shape == (2, 2, 2, 8)
    for channel in range(2):
        for lag_index, lag in enumerate(lags):
            step = encoded.shape[1] - 1 - lag
            expected_column = 2 * step + channel
            np.testing.assert_allclose(
                sensitivity[:, channel, lag_index, expected_column], 1.0
            )
            other = np.delete(
                sensitivity[:, channel, lag_index], expected_column, axis=1
            )
            np.testing.assert_allclose(other, 0.0)


def test_channel_novelty_distinguishes_collinear_and_orthogonal_responses() -> None:
    reference = np.asarray([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    collinear = np.asarray([[3.0, 0.0], [-2.0, 0.0], [1.0, 0.0]])
    orthogonal = np.asarray([[0.0, 1.0], [0.0, -2.0], [0.0, 0.5]])

    collinear_novelty, rank = channel_novelty_fraction(reference, collinear)
    orthogonal_novelty, _ = channel_novelty_fraction(reference, orthogonal)

    assert rank == 1
    assert collinear_novelty < 1e-12
    assert orthogonal_novelty > 1.0 - 1e-12


def test_matched_channel_cosines_report_parallel_channel_responses() -> None:
    sensitivity = np.zeros((3, 2, 2, 4), dtype=float)
    sensitivity[:, 0, :, 0] = 1.0
    sensitivity[:, 1, :, 0] = -2.0

    mean_absolute, median_absolute, rows = matched_channel_cosines(sensitivity)

    assert rows == 6
    assert np.isclose(mean_absolute, 1.0)
    assert np.isclose(median_absolute, 1.0)
