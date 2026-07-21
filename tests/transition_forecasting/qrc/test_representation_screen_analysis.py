from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.representation_screen_analysis import (
    _parse_mixed_utc,
    _prequential_har_residuals,
)


def test_parse_mixed_utc_accepts_naive_and_offset_timestamps() -> None:
    values = pd.Series(
        [
            "2020-01-01 00:00:00",
            "2020-01-02 00:00:00+00:00",
            "2020-01-03T00:00:00Z",
        ]
    )
    parsed = _parse_mixed_utc(values)
    assert str(parsed.dtype) == "datetime64[ns, UTC]"
    assert parsed.is_monotonic_increasing


def test_prequential_residuals_handle_mixed_origin_dates() -> None:
    dates = pd.date_range("2020-01-01", periods=24, freq="D")
    origin_date = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if index % 2 == 0
        else timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00")
        for index, timestamp in enumerate(dates)
    ]
    level = np.linspace(-5.0, -4.0, len(dates))
    frame = pd.DataFrame(
        {
            "origin_date": origin_date,
            "level": level,
            "mean5": level + 0.01,
            "mean20": level + 0.02,
        }
    )
    targets = np.repeat(level[:, None], 10, axis=1)
    train_mask = np.ones(len(frame), dtype=bool)

    residuals, valid = _prequential_har_residuals(
        frame,
        targets,
        train_mask,
        blocks=4,
    )

    assert residuals.shape == targets.shape
    assert valid.any()
    assert np.isfinite(residuals[valid]).all()
