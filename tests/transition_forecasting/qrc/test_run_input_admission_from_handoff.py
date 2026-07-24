from __future__ import annotations

import pandas as pd

from scripts.transition_forecasting.qrc.run_input_admission_from_handoff import (
    _normalize_origin_dates,
)


def test_normalizes_mixed_naive_and_utc_origin_dates() -> None:
    values = pd.Series(
        [
            "2020-03-01 00:00:00",
            "2020-03-02 00:00:00+00:00",
            "2020-03-03T15:45:12Z",
        ]
    )

    observed = _normalize_origin_dates(values)

    assert observed.tolist() == [
        "2020-03-01T00:00:00Z",
        "2020-03-02T00:00:00Z",
        "2020-03-03T00:00:00Z",
    ]
