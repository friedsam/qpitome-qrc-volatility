from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.chronological_rematched_dataset import (
    _candidate_partition,
    _validate_positive_intervals,
)


def test_positive_interval_ends_at_forecast_target() -> None:
    positives = pd.DataFrame(
        [
            {
                "input_start_date": "2000-01-01",
                "origin_date": "2000-02-01",
                "target_end_date": "2000-02-15",
            }
        ]
    )

    validated = _validate_positive_intervals(positives)
    assert list(validated.columns) == list(positives.columns)
    assert validated.loc[0, "origin_date"] < validated.loc[0, "target_end_date"]


def test_candidate_partition_uses_target_end_not_future_label_maturity() -> None:
    fold_frame = pd.DataFrame(
        [
            {
                "fold_split": "train",
                "interval_start": "2000-01-01",
                "interval_end": "2000-01-31",
            },
            {
                "fold_split": "val",
                "interval_start": "2000-02-10",
                "interval_end": "2000-02-29",
            },
            {
                "fold_split": "test",
                "interval_start": "2000-03-10",
                "interval_end": "2000-03-31",
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "train",
                "input_start_date": "1999-12-01",
                "target_end_date": "2000-01-20",
            },
            {
                "candidate_id": "val",
                "input_start_date": "2000-02-10",
                "target_end_date": "2000-02-20",
            },
            {
                "candidate_id": "test",
                "input_start_date": "2000-03-10",
                "target_end_date": "2000-03-20",
            },
        ]
    )

    partitioned = _candidate_partition(candidates, fold_frame).set_index(
        "candidate_id"
    )
    assert partitioned.loc["train", "fold_split"] == "train"
    assert partitioned.loc["val", "fold_split"] == "val"
    assert partitioned.loc["test", "fold_split"] == "test"
