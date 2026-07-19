from __future__ import annotations

import pandas as pd

from transition_forecasting.catalogue.transition_events import MATCH_FEATURES
from transition_forecasting.modeling.chronological_control_matching import (
    MatchConfig,
    rematch_controls_within_partition,
)


def _features(offset: float) -> dict[str, float]:
    return {feature: offset + index * 0.01 for index, feature in enumerate(MATCH_FEATURES)}


def test_partition_local_matching_is_complete_and_does_not_reuse_origins() -> None:
    positives = pd.DataFrame(
        [
            {
                "sample_id": f"P{sample}",
                "episode_id": f"E{sample}",
                "market_group": "M",
                "event_onset": pd.Timestamp("2010-01-01") + pd.Timedelta(days=sample),
                "label": 1,
                "fold": 1,
                "fold_split": "train",
                "index": "IDX",
                "lead": 5,
                **_features(float(sample)),
            }
            for sample in range(2)
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "fold": 1,
                "fold_split": "train",
                "index": "IDX",
                "lead": 5,
                "origin_pos": position,
                "origin_date": pd.Timestamp("2009-01-01") + pd.Timedelta(days=position),
                "input_start_date": pd.Timestamp("2008-11-01") + pd.Timedelta(days=position),
                "target_end_date": pd.Timestamp("2009-02-01") + pd.Timedelta(days=position),
                **_features(float(position % 2)),
            }
            for position in range(8)
        ]
    )

    matched, audit = rematch_controls_within_partition(
        positives,
        candidates,
        config=MatchConfig(controls_per_positive=3),
    )

    assert len(matched) == 6
    assert audit["complete_match"].all()
    assert not matched.duplicated(["fold", "fold_split", "index", "origin_date"]).any()
    assert matched.groupby("matched_positive_id").size().eq(3).all()


def test_matching_never_borrows_candidates_from_another_partition() -> None:
    positive = pd.DataFrame(
        [
            {
                "sample_id": "P1",
                "episode_id": "E1",
                "market_group": "M",
                "event_onset": pd.Timestamp("2010-01-01"),
                "label": 1,
                "fold": 1,
                "fold_split": "val",
                "index": "IDX",
                "lead": 1,
                **_features(0.0),
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "fold": 1,
                "fold_split": split,
                "index": "IDX",
                "lead": 1,
                "origin_pos": position,
                "origin_date": pd.Timestamp("2009-01-01") + pd.Timedelta(days=position),
                "input_start_date": pd.Timestamp("2008-11-01") + pd.Timedelta(days=position),
                "target_end_date": pd.Timestamp("2009-02-01") + pd.Timedelta(days=position),
                **_features(0.0),
            }
            for position, split in enumerate(["train", "train", "val"])
        ]
    )

    matched, audit = rematch_controls_within_partition(
        positive,
        candidates,
        config=MatchConfig(controls_per_positive=3),
    )

    assert len(matched) == 1
    assert matched["fold_split"].eq("val").all()
    assert audit.iloc[0]["controls_selected"] == 1
    assert not bool(audit.iloc[0]["complete_match"])
