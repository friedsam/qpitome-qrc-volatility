from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    _candidate_category,
    chronological_inner_split,
    gate_values,
    observable_indices,
    parse_pc_band,
)


def test_parse_pc_band_uses_one_based_inclusive_notation() -> None:
    assert parse_pc_band("5", 12) == (4,)
    assert parse_pc_band("2-5", 12) == (1, 2, 3, 4)
    assert parse_pc_band("9-20", 12) == (8, 9, 10, 11)


def test_observable_indices_select_occupation_columns() -> None:
    names = (
        "probe_10_occupation_site_0",
        "probe_10_nearest_pair_0_1",
        "probe_10_nearest_connected_0_1",
        "probe_20_occupation_site_0",
    )
    assert observable_indices(names, "occupation").tolist() == [0, 3]
    assert observable_indices(names, "connected").tolist() == [2]


def test_chronological_inner_split_preserves_time_order() -> None:
    dates = np.asarray(
        [f"2026-01-{day:02d} 00:00:00" for day in range(1, 21)]
    )
    eligible = np.ones(len(dates), dtype=bool)
    fit, tune = chronological_inner_split(
        dates,
        eligible,
        holdout_fraction=0.25,
    )
    parsed = pd.to_datetime(dates, utc=True)
    assert fit.sum() == 15
    assert tune.sum() == 5
    assert parsed[fit].max() < parsed[tune].min()


def test_gate_thresholds_are_fit_only_on_selected_rows() -> None:
    encoded = np.zeros((6, 5, 2), dtype=float)
    encoded[:, :, 1] = np.arange(6, dtype=float)[:, None]
    har = np.tile(np.linspace(0.0, 1.0, 10), (6, 1))
    fit = np.asarray([True, True, True, True, False, False])
    gate = gate_values(
        "instability_high_65",
        encoded_sequence=encoded,
        har=har,
        fit_mask=fit,
    )
    assert gate[:2].sum() == 0.0
    assert gate[-1] == 1.0


def test_candidate_categories_keep_shrinkage_ungated() -> None:
    assert "shrinkage" in _candidate_category(
        "pca",
        "prefix_4",
        "none",
    )
    assert "shrinkage" not in _candidate_category(
        "pca",
        "prefix_4",
        "instability_ramp_50_90",
    )
    assert "gating" in _candidate_category(
        "pca",
        "prefix_4",
        "instability_ramp_50_90",
    )
