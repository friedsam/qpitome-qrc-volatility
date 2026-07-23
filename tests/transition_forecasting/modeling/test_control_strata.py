from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import MATCH_FEATURES
from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    PERSISTENT_EXCLUDED,
    ControlStrataPolicy,
    classify_control_future,
    rematch_control_strata_within_partition,
)


def _features(offset: float) -> dict[str, float]:
    return {
        feature: float(offset + index * 0.01)
        for index, feature in enumerate(MATCH_FEATURES)
    }


def _positive(sample_id: str, lead: int, offset: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "episode_id": f"E_{sample_id}",
        "market_group": "market",
        "event_onset": pd.Timestamp("2010-01-15"),
        "label": 1,
        "fold": 1,
        "fold_split": "train",
        "index": "IDX",
        "lead": lead,
        **_features(offset),
    }


def _candidate(
    *,
    stratum: str,
    lead: int,
    position: int,
    offset: float,
) -> dict[str, object]:
    origin = pd.Timestamp("2008-01-01") + pd.Timedelta(days=position)
    return {
        "candidate_id": f"C_{stratum}_{lead}_{position}",
        "control_stratum": stratum,
        "fold": 1,
        "fold_split": "train",
        "index": "IDX",
        "lead": lead,
        "origin_pos": position,
        "origin_date": origin,
        "input_start_date": origin - pd.Timedelta(days=60),
        "target_end_date": origin + pd.Timedelta(days=20),
        "target_x_h1": float(1000 + position),
        **_features(offset),
    }


def test_future_strata_follow_the_frozen_persistence_rule() -> None:
    policy = ControlStrataPolicy()
    calm = classify_control_future(
        np.full(policy.persistence_window, -1.0),
        threshold=0.0,
        policy=policy,
    )
    hard = classify_control_future(
        np.asarray([1.0, 1.0, 1.0] + [-1.0] * 12),
        threshold=0.0,
        policy=policy,
    )
    persistent = classify_control_future(
        np.asarray([1.0] * policy.persistence_required + [-1.0] * 5),
        threshold=0.0,
        policy=policy,
    )

    assert calm["control_stratum"] == CALM
    assert calm["future_threshold_crossings"] == 0
    assert hard["control_stratum"] == HARD_NEGATIVE
    assert hard["future_threshold_crossings"] == 3
    assert persistent["control_stratum"] == PERSISTENT_EXCLUDED
    assert persistent["future_persistent"] is True


def test_historical_total_three_freezes_two_calm_and_one_hard() -> None:
    policy = ControlStrataPolicy.from_total_controls(3)
    assert policy.calm_controls_per_positive == 2
    assert policy.hard_controls_per_positive == 1
    assert policy.controls_per_positive == 3


def test_matching_is_deterministic_preorigin_only_and_without_reuse() -> None:
    positives = pd.DataFrame(
        [
            _positive("P_L1", 1, 0.0),
            _positive("P_L5", 5, 1.0),
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate(
                stratum=stratum,
                lead=lead,
                position=position,
                offset=float(position % 3),
            )
            for stratum, positions in ((CALM, range(10, 18)), (HARD_NEGATIVE, range(30, 36)))
            for lead in (1, 5)
            for position in positions
        ]
    )
    policy = ControlStrataPolicy.from_total_controls(3)

    matched_a, audit_a = rematch_control_strata_within_partition(
        positives,
        candidates,
        policy=policy,
    )
    shuffled = candidates.sample(frac=1.0, random_state=991).reset_index(drop=True)
    shuffled["target_x_h1"] = shuffled["target_x_h1"] * -17.0
    matched_b, audit_b = rematch_control_strata_within_partition(
        positives,
        shuffled,
        policy=policy,
    )

    identity = [
        "sample_id",
        "matched_positive_id",
        "control_stratum",
        "lead",
        "origin_pos",
        "match_distance",
    ]
    pd.testing.assert_frame_equal(
        matched_a[identity].sort_values(identity[:-1]).reset_index(drop=True),
        matched_b[identity].sort_values(identity[:-1]).reset_index(drop=True),
    )
    assert audit_a["complete_all_strata"].all()
    assert audit_b["complete_all_strata"].all()
    assert not matched_a.duplicated(["fold", "fold_split", "index", "origin_date"]).any()
    assert matched_a.groupby(["matched_positive_id", "control_stratum"]).size().to_dict() == {
        ("P_L1", CALM): 2,
        ("P_L1", HARD_NEGATIVE): 1,
        ("P_L5", CALM): 2,
        ("P_L5", HARD_NEGATIVE): 1,
    }


def test_positive_is_incomplete_when_one_required_stratum_is_missing() -> None:
    positives = pd.DataFrame([_positive("P1", 1, 0.0)])
    candidates = pd.DataFrame(
        [
            _candidate(stratum=CALM, lead=1, position=position, offset=0.0)
            for position in range(10, 14)
        ]
    )

    matched, audit = rematch_control_strata_within_partition(
        positives,
        candidates,
        policy=ControlStrataPolicy.from_total_controls(3),
    )

    assert not audit["complete_all_strata"].any()
    assert matched["control_stratum"].eq(CALM).all()
