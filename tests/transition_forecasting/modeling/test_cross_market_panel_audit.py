from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_cross_market_panel_audit.py")
SPEC = importlib.util.spec_from_file_location("run_cross_market_panel_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_csv_streams_all_markets(tmp_path: Path) -> None:
    panel = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "Index": ["A", "A", "B", "B"],
        "Close": [1.0, 1.1, 2.0, 2.1],
    })
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)

    per_market, summary = MODULE.audit_csv(path, chunksize=2)

    assert summary["markets"] == 2
    assert summary["union_dates"] == 2
    assert summary["suggested_value_column"] == "Close"
    assert set(per_market["market"]) == {"A", "B"}
    assert (per_market["coverage_of_union"] == 1.0).all()
