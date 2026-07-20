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


def test_baseline_sample_delta_reports_removed_and_added(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.csv"
    reference = pd.DataFrame(
        {
            "sample_id": ["shared", "removed"],
            "index": ["A", "A"],
            "lead": [1, 1],
            "split": ["train", "train"],
            "label": [1, 0],
        }
    )
    current = pd.DataFrame(
        {
            "sample_id": ["shared", "added"],
            "index": ["A", "B"],
            "lead": [1, 5],
            "split": ["train", "val"],
            "label": [1, 0],
        }
    )
    reference.to_csv(reference_path, index=False)
    delta, summary = MODULE.baseline_sample_delta(current, reference_path)
    assert summary == {
        "available": True,
        "reference_samples": 2,
        "current_samples": 2,
        "shared_samples": 1,
        "removed_samples": 1,
        "added_samples": 1,
    }
    assert set(delta["status"]) == {"removed_after_cleanup", "added_after_cleanup"}


def test_event_delta_reports_changed_events(tmp_path: Path) -> None:
    reference_path = tmp_path / "events.csv"
    pd.DataFrame(
        {"index": ["A", "A"], "event_onset": ["2020-01-01", "2020-02-01"]}
    ).to_csv(reference_path, index=False)
    current = pd.DataFrame(
        {"index": ["A", "B"], "onset_date": ["2020-01-01", "2020-03-01"]}
    )
    delta, summary = MODULE.event_delta(current, reference_path)
    assert summary["shared_events"] == 1
    assert summary["removed_events"] == 1
    assert summary["added_events"] == 1
    assert len(delta) == 2


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
        rows.append(
            {
                "sample_id": pid,
                "label": 1,
                "index": "A",
                "lead": 1,
                "split": "train",
                "match_distance": np.nan,
            }
        )
        for control in range(5):
            rows.append(
                {
                    "sample_id": f"N{positive}_{control}",
                    "label": 0,
                    "index": "A",
                    "lead": 1,
                    "split": "train",
                    "match_distance": float(control),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(dataset / "sample_manifest.csv", index=False)
    reference_manifest = tmp_path / "reference_manifest.csv"
    manifest.iloc[:-1].to_csv(reference_manifest, index=False)
    reference_catalogue = tmp_path / "reference_catalogue.csv"
    pd.DataFrame({"index": ["A"], "event_onset": [dates[10]]}).to_csv(
        reference_catalogue, index=False
    )

    report = MODULE.analyze(
        dataset,
        output,
        reference_manifest=reference_manifest,
        reference_catalogue=reference_catalogue,
    )

    assert report["decisions"]["structural_bad_print_removal_applied"] is True
    assert report["baseline_sample_delta"]["available"] is True
    assert report["event_delta"]["available"] is True
    for name in (
        "residual_extremes.csv",
        "matching_quality.csv",
        "sample_attribution.csv",
        "effective_starts.csv",
        "transformation_scenarios.csv",
        "baseline_sample_delta.csv",
        "baseline_event_delta.csv",
        "final_cleanup_report.md",
        "cleanup_evidence.json",
    ):
        assert (output / name).is_file()
