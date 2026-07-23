from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import HORIZON, LEADS, WINDOW
from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    ControlStrataPolicy,
)
from transition_forecasting.modeling.stage_d_candidate_pool import (
    build_candidate_pool_from_series,
)


def test_candidate_pool_uses_exact_trading_row_intervals_and_excludes_events() -> None:
    dates = pd.bdate_range("2010-01-01", periods=180)
    values = np.linspace(-4.0, -3.0, len(dates)) + 0.02 * np.sin(
        np.arange(len(dates)) / 5.0
    )
    series = pd.Series(values, index=dates)
    onset_position = 100
    policy = ControlStrataPolicy()

    frame, tensor = build_candidate_pool_from_series(
        index_name="IDX",
        series=series,
        onset_positions=np.asarray([onset_position]),
        event_exclusion=5,
        threshold=-3.35,
        control_policy=policy,
    )

    assert not frame.empty
    assert tensor.shape == (len(frame), WINDOW, 1)
    assert set(frame["lead"].unique()) == set(LEADS)
    assert not frame["origin_pos"].between(
        onset_position - 5,
        onset_position + 5,
    ).any()
    assert set(frame["control_stratum"].unique()).issubset(
        {CALM, HARD_NEGATIVE}
    )
    assert not frame["future_persistent"].astype(bool).any()
    assert frame.loc[
        frame["control_stratum"].eq(CALM),
        "future_threshold_crossings",
    ].eq(0).all()
    assert frame.loc[
        frame["control_stratum"].eq(HARD_NEGATIVE),
        "future_threshold_crossings",
    ].between(1, 9).all()

    row = frame.iloc[0]
    position = int(row["origin_pos"])
    assert pd.Timestamp(row["input_start_date"]) == dates[position - WINDOW + 1]
    assert pd.Timestamp(row["origin_date"]) == dates[position]
    assert pd.Timestamp(row["target_end_date"]) == dates[position + HORIZON]
    assert pd.Timestamp(row["future_assessment_end_date"]) == dates[
        position + policy.future_assessment_rows
    ]


def test_candidate_ids_and_index_lead_origins_are_unique() -> None:
    dates = pd.bdate_range("2012-01-01", periods=160)
    series = pd.Series(np.linspace(-4.2, -3.2, len(dates)), index=dates)

    frame, _ = build_candidate_pool_from_series(
        index_name="IDX",
        series=series,
        onset_positions=np.asarray([], dtype=int),
        threshold=-3.45,
    )

    assert not frame["candidate_id"].duplicated().any()
    assert not frame.duplicated(["index", "lead", "origin_pos"]).any()
    counts = frame.groupby(["index", "origin_pos"]).size()
    assert counts.eq(len(LEADS)).all()


def test_persistent_future_is_excluded_from_control_candidates() -> None:
    dates = pd.bdate_range("2013-01-01", periods=100)
    values = np.full(len(dates), -2.0)
    series = pd.Series(values, index=dates)

    frame, tensor = build_candidate_pool_from_series(
        index_name="IDX",
        series=series,
        onset_positions=np.asarray([], dtype=int),
        threshold=-3.0,
    )

    assert frame.empty
    assert tensor.shape == (0, WINDOW, 1)
