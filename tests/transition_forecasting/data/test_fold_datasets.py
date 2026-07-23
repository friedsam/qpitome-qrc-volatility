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
    assert enriched.loc[0, "label_end_date_source"] == "exact_trading_row"
    assert pd.Timestamp(enriched.loc[1, "label_end_date"]) == dates[30]
    assert enriched.loc[1, "label_end_date_source"] == "target_end_nonpositive"


def test_positive_label_end_uses_catalogue_history_end_when_packaged_series_is_short() -> None:
    dates = pd.bdate_range("2000-01-03", periods=50)
    series = pd.Series(range(len(dates)), index=dates, dtype=float)
    onset = dates[45]
    target_end = dates[49]
    source_history_end = pd.Timestamp("2000-04-30")
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "P_EDGE",
                "label": 1,
                "index": "IDX",
                "event_onset": onset,
                "target_end_date": target_end,
            }
        ]
    )
    catalogue = pd.DataFrame(
        [
            {
                "index": "IDX",
                "onset_date": onset,
                "history_end": source_history_end,
            }
        ]
    )

    enriched = _attach_positive_label_end_dates(
        manifest,
        {"IDX": series},
        catalogue,
    )

    assert pd.Timestamp(enriched.loc[0, "label_end_date"]) == source_history_end
    assert (
        enriched.loc[0, "label_end_date_source"]
        == "catalogue_history_end_conservative"
    )
