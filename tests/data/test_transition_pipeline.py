from pathlib import Path

import numpy as np
import pandas as pd

from data import transition_events as events


def _write_fixture(root: Path) -> None:
    data_root = root / "global index etf return"
    data_root.mkdir(parents=True)
    index = pd.bdate_range("2007-01-01", "2020-12-31")
    rng = np.random.default_rng(7)
    for offset, name in enumerate(events.NAMES):
        base = rng.normal(-4.4, 0.18, len(index))
        for start in [400 + offset, 900 + 2 * offset, 1450 + offset, 2300 + 2 * offset, 3100 + offset]:
            base[start : start + 20] += 1.4
        ratio = np.exp(np.exp(base) * np.sqrt(4 * np.log(2)))
        low = 100 + np.cumsum(rng.normal(0, 0.05, len(index)))
        high = low * ratio
        frame = pd.DataFrame(
            {
                "Date": index.strftime("%m/%d/%Y"),
                "Price": low,
                "Open": low,
                "High": high,
                "Low": low,
            }
        )
        frame.to_csv(data_root / f"{name}.csv", index=False)


def test_pipeline_runs_end_to_end(tmp_path):
    data = tmp_path / "data"
    run_dir = tmp_path / "results" / "transition_forecasting" / "synthetic_integration"
    _write_fixture(data)
    summary = events.run_pipeline(data, run_dir)
    assert summary["pipeline_status"] == "completed_end_to_end"
    for name in [
        "sample_manifest.csv",
        "matching_balance_smd.csv",
        "input_checksums.csv",
        "output_checksums.csv",
        "sequence_tensors.npz",
        "summary_metrics.json",
        "run_manifest.json",
    ]:
        assert (run_dir / name).exists(), name
    manifest = pd.read_csv(run_dir / "sample_manifest.csv")
    assert {"split", "control_reused_across_leads", "matched_positive_id"}.issubset(manifest.columns)
