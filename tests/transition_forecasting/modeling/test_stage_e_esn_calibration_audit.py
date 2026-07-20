from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_esn_calibration_audit.py")
ROLLING = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
if not SCRIPT.is_file() or not ROLLING.is_file():
    pytest.skip(
        "Legacy Stage E calibration implementation or rolling-origin dependency is absent",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("stage_e_esn_calibration_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_functions_return_expected_shapes() -> None:
    y = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    pred = 0.5 * y + 0.25
    affine, intercept, slope = MODULE._global_affine(y, pred, pred)
    scaled, scales = MODULE._horizon_scale(y, pred, pred)
    assert affine.shape == y.shape
    assert scaled.shape == y.shape
    assert np.isfinite([intercept, slope]).all()
    assert np.isfinite(scales).all()


def test_cross_market_audit_finds_candidate(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
        "market": ["A", "B", "A", "B"],
        "value": [1.0, 2.0, 1.5, 2.5],
    })
    frame.to_csv(tmp_path / "panel.csv", index=False)
    audit = MODULE.audit_cross_market_files((tmp_path,))
    row = audit.loc[audit["path"].str.endswith("panel.csv")].iloc[0]
    assert bool(row["cross_market_candidate"])
    assert int(row["markets_sampled"]) == 2
    assert int(row["dates_sampled"]) == 2
