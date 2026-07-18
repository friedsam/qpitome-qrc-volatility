from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/build_cross_market_compact_tensors.py")
SPEC = importlib.util.spec_from_file_location("build_cross_market_compact_tensors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_panel_channels_are_bounded_where_expected() -> None:
    dates = pd.date_range("2020-01-01", periods=4, tz="UTC")
    wide = pd.DataFrame({
        "A": [1.0, 2.0, 1.0, 3.0],
        "B": [2.0, 3.0, 4.0, 2.0],
    }, index=dates)
    agg = MODULE.aggregate_panel(wide)
    assert list(agg.columns) == MODULE.CHANNELS[2:8]
    assert agg["availability_fraction"].eq(1.0).all()
    assert agg["fraction_rising"].dropna().between(0.0, 1.0).all()
    assert agg["directional_agreement"].dropna().between(0.5, 1.0).all()


def test_normalize_fold_uses_train_only_statistics() -> None:
    raw = np.zeros((4, 2, len(MODULE.CHANNELS)), dtype=float)
    raw[0:2, :, 0] = [[1.0, 3.0], [5.0, 7.0]]
    raw[2:4, :, 0] = 1000.0
    raw[:, :, 1] = 2.0
    raw[0:2, :, 2:7] = 1.0
    raw[2:4, :, 2:7] = 1000.0
    manifest = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "market_group": ["united_states"] * 4,
    })
    fold_rows = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "fold_split": ["train", "train", "val", "test"],
    })
    normalized, stats = MODULE.normalize_fold(raw, manifest, fold_rows)
    assert abs(normalized[:2, :, 0].mean()) < 1e-7
    assert stats["market_stats"]["united_states"]["mean"] == 4.0
    assert normalized[2, 0, 0] > 100.0
