from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.global_index_ohlc_audit import audit_directory, audit_file


def _write_ohlc(path: Path, *, periods: int = 20, invalid: bool = False) -> None:
    dates = pd.date_range("2000-01-03", periods=periods, freq="365D")
    high = [100.0 + i for i in range(periods)]
    low = [99.0 + i for i in range(periods)]
    if invalid:
        low[3] = high[3] + 1.0
    pd.DataFrame({"Date": dates, "High": high, "Low": low}).to_csv(path, index=False)


def test_audit_file_accepts_long_valid_history(tmp_path: Path) -> None:
    path = tmp_path / "N225.csv"
    _write_ohlc(path)

    result = audit_file(path, min_years=15.0, min_valid_fraction=0.98)

    assert result["eligible"] is True
    assert result["index"] == "N225"
    assert result["invalid_ranges"] == 0


def test_audit_file_rejects_invalid_high_low_range(tmp_path: Path) -> None:
    path = tmp_path / "BAD.csv"
    _write_ohlc(path, invalid=True)

    result = audit_file(path, min_years=15.0, min_valid_fraction=0.90)

    assert result["eligible"] is False
    assert result["invalid_ranges"] == 1
    assert "invalid_ranges" in str(result["failure_reason"])


def test_audit_directory_reports_counts(tmp_path: Path) -> None:
    _write_ohlc(tmp_path / "GOOD.csv")
    _write_ohlc(tmp_path / "SHORT.csv", periods=5)

    inventory, summary = audit_directory(tmp_path, min_years=15.0)

    assert len(inventory) == 2
    assert summary["eligible_files"] == 1
    assert summary["ineligible_files"] == 1
