from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    TRANSITION,
)
from transition_forecasting.modeling.fold_selection import (
    select_balanced_episode_rows,
)


def _rows(episode: int, split: str) -> list[dict[str, object]]:
    positive_id = f"P{episode}"
    base = {
        "fold": 1,
        "fold_split": split,
        "lead": 5,
        "episode_id": f"E{episode}",
        "event_onset": pd.Timestamp("2000-01-01") + pd.Timedelta(days=episode * 30),
        "index": "IDX",
    }
    return [
        {
            **base,
            "sample_id": positive_id,
            "label": 1,
            "origin_date": base["event_onset"] - pd.Timedelta(days=5),
            "evaluation_stratum": TRANSITION,
            "_tensor_row": episode,
        },
        {
            **base,
            "sample_id": f"N_CALM_{positive_id}",
            "label": 0,
            "origin_date": base["event_onset"] - pd.Timedelta(days=100),
            "evaluation_stratum": CALM,
            "matched_positive_id": positive_id,
            "match_distance": 0.1,
            "_tensor_row": 100 + episode,
        },
        {
            **base,
            "sample_id": f"N_HARD_{positive_id}",
            "label": 0,
            "origin_date": base["event_onset"] - pd.Timedelta(days=200),
            "evaluation_stratum": HARD_NEGATIVE,
            "matched_positive_id": positive_id,
            "match_distance": 0.2,
            "_tensor_row": 200 + episode,
        },
    ]


def _manifest() -> pd.DataFrame:
    rows = _rows(1, "train") + _rows(2, "train") + _rows(3, "val")
    rows += _rows(4, "test")
    return pd.DataFrame(rows)


def test_redesigned_manifest_ignores_seed_and_returns_equal_strata() -> None:
    manifest = _manifest()
    first = select_balanced_episode_rows(
        manifest,
        fold=1,
        lead=5,
        max_per_class=2,
        seed=1,
    )
    second = select_balanced_episode_rows(
        manifest.sample(frac=1.0, random_state=99).reset_index(drop=True),
        fold=1,
        lead=5,
        max_per_class=2,
        seed=999999,
    )

    identity = ["fold_split", "evaluation_stratum", "episode_id", "sample_id"]
    pd.testing.assert_frame_equal(first[identity], second[identity])
    assert first.groupby(["fold_split", "evaluation_stratum"]).size().to_dict() == {
        ("train", CALM): 2,
        ("train", HARD_NEGATIVE): 2,
        ("train", TRANSITION): 2,
        ("val", CALM): 1,
        ("val", HARD_NEGATIVE): 1,
        ("val", TRANSITION): 1,
    }
    assert not first["fold_split"].eq("test").any()
