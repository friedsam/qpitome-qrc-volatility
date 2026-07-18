from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_cross_market_transform_match.py")
SPEC = importlib.util.spec_from_file_location("run_cross_market_transform_match", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mapping_covers_manifest_market_names() -> None:
    expected = {
        "argentina", "australia", "belgium", "brazil", "canada", "chile",
        "china_mainland", "europe_regional", "eurozone_regional", "france",
        "germany", "hong_kong", "india", "indonesia", "israel", "japan",
        "malaysia", "mexico", "new_zealand", "singapore", "south_korea",
        "taiwan", "united_kingdom", "united_states",
    }
    assert expected == set(MODULE.MARKET_GROUP_TO_TICKER)


def test_compare_transforms_identifies_log_rv20() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    close = np.exp(np.cumsum(np.sin(np.arange(100) / 5.0) * 0.01 + 0.001))
    panel = pd.DataFrame({"date": dates, "close": close, "ticker": "^GSPC"})
    signals = MODULE.build_signals(panel)
    target = signals["^GSPC"]["log_rv20"].dropna().tail(40).to_numpy()
    manifest = pd.DataFrame({
        "sample_id": ["s1"],
        "market_group": ["united_states"],
        "origin_date": [dates[-1]],
    })
    results = MODULE.compare_transforms(manifest, target[None, :], signals)
    assert results.iloc[0]["candidate"] == "log_rv20"
    assert results.iloc[0]["correlation"] > 0.999999
