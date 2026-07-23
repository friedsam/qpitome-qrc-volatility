from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.chronological_rematched_dataset import (
    _candidate_partition,
    _validate_positive_intervals,
)


def test_positive_interval_ends_at_persistent_label_maturity() -> None:
    positives = pd.DataFrame(
        [
            {
                "input_start_date": "2000-01-01",
                "origin_date": "2000-02-01",
                "target_end_date": "2000-02-15",
                "label_end_date": "2000-03-01",
            }
        ]
    )

    validated = _validate_positive_intervals(positives)
    assert validated.loc[0, "target_end_date"] < validated.loc[0, "label_end_date"]


def test_control_is_not_train_when_label_matures_after_train_boundary() -> None:
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
                "candidate_id": "train_mature",
                "input_start_date": "1999-12-01",
                "target_end_date": "2000-01-20",
                "future_assessment_end_date": "2000-01-31",
            },
            {
                "candidate_id": "train_target_only",
                "input_start_date": "1999-12-01",
                "target_end_date": "2000-01-20",
                "future_assessment_end_date": "2000-02-05",
            },
            {
                "candidate_id": "val_mature",
                "input_start_date": "2000-02-10",
                "target_end_date": "2000-02-20",
                "future_assessment_end_date": "2000-02-29",
            },
            {
                "candidate_id": "test",
                "input_start_date": "2000-03-10",
                "target_end_date": "2000-03-20",
                "future_assessment_end_date": "2000-04-03",
            },
        ]
    )

    partitioned = _candidate_partition(candidates, fold_frame).set_index(
        "candidate_id"
    )
    assert partitioned.loc["train_mature", "fold_split"] == "train"
    assert partitioned.loc["train_target_only", "fold_split"] == "unused"
    assert partitioned.loc["val_mature", "fold_split"] == "val"
    assert partitioned.loc["test", "fold_split"] == "test"
