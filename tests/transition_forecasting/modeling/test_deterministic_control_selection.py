from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    TRANSITION,
)
from transition_forecasting.modeling.deterministic_control_selection import (
    select_deterministic_control_panel,
)


def _episode_rows(
    *,
    episode: int,
    lead: int,
    split: str,
    onset: str,
) -> list[dict[str, object]]:
    positive_id = f"P_E{episode:02d}_L{lead}"
    base = {
        "fold": 1,
        "fold_split": split,
        "lead": lead,
        "episode_id": f"E{episode:02d}",
        "event_onset": pd.Timestamp(onset),
        "index": "IDX",
    }
    return [
        {
            **base,
            "sample_id": positive_id,
            "label": 1,
            "origin_date": pd.Timestamp(onset) - pd.Timedelta(days=lead),
            "evaluation_stratum": TRANSITION,
            "_tensor_row": episode * 10 + lead,
        },
        {
            **base,
            "sample_id": f"N_CALM_{positive_id}_1",
            "label": 0,
            "origin_date": pd.Timestamp(onset) - pd.Timedelta(days=100 + episode),
            "evaluation_stratum": CALM,
            "matched_positive_id": positive_id,
            "match_distance": 0.10 + episode * 0.001,
            "_tensor_row": 1000 + episode * 10 + lead,
        },
        {
            **base,
            "sample_id": f"N_CALM_{positive_id}_2",
            "label": 0,
            "origin_date": pd.Timestamp(onset) - pd.Timedelta(days=200 + episode),
            "evaluation_stratum": CALM,
            "matched_positive_id": positive_id,
            "match_distance": 0.30 + episode * 0.001,
            "_tensor_row": 2000 + episode * 10 + lead,
        },
        {
            **base,
            "sample_id": f"N_HARD_{positive_id}_1",
            "label": 0,
            "origin_date": pd.Timestamp(onset) - pd.Timedelta(days=300 + episode),
            "evaluation_stratum": HARD_NEGATIVE,
            "matched_positive_id": positive_id,
            "match_distance": 0.20 + episode * 0.001,
            "_tensor_row": 3000 + episode * 10 + lead,
        },
    ]


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode, onset in enumerate(
        ["2001-01-15", "2002-01-15", "2003-01-15", "2004-01-15"],
        start=1,
    ):
        rows.extend(
            _episode_rows(
                episode=episode,
                lead=5,
                split="train",
                onset=onset,
            )
        )
    rows.extend(
        _episode_rows(
            episode=9,
            lead=5,
            split="val",
            onset="2009-01-15",
        )
    )
    rows.extend(
        _episode_rows(
            episode=10,
            lead=5,
            split="test",
            onset="2010-01-15",
        )
    )
    return pd.DataFrame(rows)


def test_selector_is_deterministic_balanced_and_uses_closest_controls() -> None:
    manifest = _manifest()
    selected_a = select_deterministic_control_panel(
        manifest,
        fold=1,
        leads=(5,),
    )
    selected_b = select_deterministic_control_panel(
        manifest.sample(frac=1.0, random_state=17).reset_index(drop=True),
        fold=1,
        leads=(5,),
    )

    identity = [
        "fold_split",
        "lead",
        "evaluation_stratum",
        "episode_id",
        "sample_id",
    ]
    pd.testing.assert_frame_equal(
        selected_a[identity].reset_index(drop=True),
        selected_b[identity].reset_index(drop=True),
    )
    counts = selected_a.groupby(
        ["fold_split", "evaluation_stratum"], sort=True
    ).size()
    assert counts.loc[("train", TRANSITION)] == 4
    assert counts.loc[("train", CALM)] == 4
    assert counts.loc[("train", HARD_NEGATIVE)] == 4
    assert counts.loc[("val", TRANSITION)] == 1
    assert counts.loc[("val", CALM)] == 1
    assert counts.loc[("val", HARD_NEGATIVE)] == 1
    assert not selected_a["fold_split"].eq("test").any()
    assert not selected_a["sample_id"].str.contains("_2$").any()


def test_selector_applies_chronology_spread_cap_after_full_eligibility() -> None:
    selected = select_deterministic_control_panel(
        _manifest(),
        fold=1,
        leads=(5,),
        splits=("train",),
        max_episodes_per_split_lead=2,
    )

    positives = selected.loc[selected["evaluation_stratum"].eq(TRANSITION)]
    assert positives["episode_id"].tolist() == ["E02", "E04"]
    assert selected["evaluation_stratum"].value_counts().to_dict() == {
        TRANSITION: 2,
        CALM: 2,
        HARD_NEGATIVE: 2,
    }


def test_selector_rejects_test_access() -> None:
    try:
        select_deterministic_control_panel(
            _manifest(),
            fold=1,
            leads=(5,),
            splits=("test",),
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("selector permitted test rows")
