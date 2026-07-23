from __future__ import annotations

import pandas as pd

from transition_forecasting.catalogue.transition_events import M
from transition_forecasting.data.fold_datasets import (
    DEFAULT_N_FOLDS,
    _attach_positive_label_end_dates,
)


def test_default_fold_count_is_eight() -> None:
    assert DEFAULT_N_FOLDS == 8


def test_positive_label_end_is_onset_plus_persistence_window_minus_one() -> None:
    dates = pd.bdate_range("2000-01-03", periods=80)
    series = pd.Series(range(len(dates)), index=dates, dtype=float)
    onset = dates[40]
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "P1",
                "label": 1,
                "index": "IDX",
                "event_onset": onset,
                "target_end_date": dates[45],
            },
            {
                "sample_id": "N1",
                "label": 0,
                "index": "IDX",
                "event_onset": onset,
                "target_end_date": dates[30],
            },
        ]
    )

    enriched = _attach_positive_label_end_dates(manifest, {"IDX": series})

    assert pd.Timestamp(enriched.loc[0, "label_end_date"]) == dates[40 + M - 1]
    assert pd.Timestamp(enriched.loc[1, "label_end_date"]) == dates[30]
