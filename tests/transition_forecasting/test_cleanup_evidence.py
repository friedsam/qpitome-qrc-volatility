from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/transition_forecasting/analyze_cleanup_evidence.py"
SPEC = importlib.util.spec_from_file_location("analyze_cleanup_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_robust_z_is_centered_and_detects_extreme() -> None:
    values = pd.Series([0.0, 0.1, -0.1, 0.05, 20.0])
    scores = MODULE.robust_z(values)
    assert np.isfinite(scores).all()
    assert abs(scores.iloc[0]) < 1.0
    assert scores.iloc[-1] > 8.0


def test_residual_extremes_marks_event_overlap() -> None:
    daily = pd.DataFrame(
        {
            "index": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "log_parkinson_volatility": [0.0, 0.1, -0.1, 0.05, 20.0, 0.0],
            "effective_start": ["2020-01-01"] * 6,
        }
    )
    catalogue = pd.DataFrame({"index": ["A"], "onset_date": ["2020-01-05"]})
    result = MODULE.residual_extremes(daily, catalogue, z_threshold=8.0)
    assert len(result) == 1
    assert bool(result.iloc[0]["within_60d_of_event"])
    assert result.iloc[0]["nearest_event_days"] == 0.0


def test_matching_summary_reports_quantiles() -> None:
    manifest = pd.DataFrame(
        {
            "label": [0, 0, 0, 1],
            "index": ["A", "A", "A", "A"],
            "lead": [1, 1, 1, 1],
            "split": ["train"] * 4,
            "match_distance": [1.0, 2.0, 3.0, np.nan],
        }
    )
    result = MODULE.matching_summary(manifest)
    assert len(result) == 1
    assert int(result.iloc[0]["controls"]) == 3
    assert result.iloc[0]["p50"] == 2.0
    assert result.iloc[0]["max"] == 3.0


def test_scenarios_are_evidence_only() -> None:
    daily = pd.DataFrame(
        {
            "index": ["A"] * 1000,
            "log_parkinson_volatility": np.r_[np.zeros(999), 100.0],
        }
    )
    result = MODULE.scenario_summary(daily)
    assert set(result["scenario"]) == {"winsor_0.1_99.9", "winsor_0.5_99.5"}
    assert not result["applied"].any()
    assert (result["changed_rows"] > 0).all()


def test_analyze_writes_all_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "evidence"
    dataset.mkdir()
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    pd.DataFrame(
        {
            "index": ["A"] * 20,
            "date": dates,
            "log_parkinson_volatility": np.linspace(-1.0, 1.0, 20),
            "effective_start": [dates[0]] * 20,
        }
    ).to_csv(dataset / "daily_volatility.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"index": ["A"], "onset_date": [dates[10]]}).to_csv(
        dataset / "transition_catalogue.csv", index=False
    )
    rows = []
    for positive in range(2):
        pid = f"P{positive}"
        rows.append({"sample_id": pid, "label": 1, "index": "A", "lead": 1, "split": "train", "match_distance": np.nan})
        for control in range(5):
            rows.append({"sample_id": f"N{positive}_{control}", "label": 0, "index": "A", "lead": 1, "split": "train", "match_distance": float(control)})
    pd.DataFrame(rows).to_csv(dataset / "sample_manifest.csv", index=False)

    report = MODULE.analyze(dataset, output)

    assert report["decisions"]["winsorization_applied"] is False
    for name in (
        "residual_extremes.csv",
        "matching_quality.csv",
        "sample_attribution.csv",
        "effective_starts.csv",
        "transformation_scenarios.csv",
        "cleanup_evidence.json",
    ):
        assert (output / name).is_file()
