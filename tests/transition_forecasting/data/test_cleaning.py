from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from transition_forecasting.data.cleaning import build_cleaned_ohlc, parse_ohlc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_fixture(path: Path) -> None:
    pd.DataFrame(
        {
            "Date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "bad-date",
                "2020-01-06",
                "2020-01-07",
            ],
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "High": [101.0, 102.0, 103.0, 104.0, 100.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 101.0, 104.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 103.0, 105.5],
        }
    ).to_csv(path, index=False)


def test_parse_ohlc_records_invalid_range_date_and_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "INDEX.csv"
    _write_fixture(source)

    canonical, corrections = parse_ohlc(source)

    assert canonical["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-07",
    ]
    assert canonical.loc[canonical["date"] == pd.Timestamp("2020-01-03"), "open"].item() == 102.0
    assert set(corrections["reason"]) == {
        "duplicate_date_keep_last",
        "invalid_date",
        "high_below_low",
    }
    assert set(corrections["action"]) == {"drop"}


def test_build_cleaned_ohlc_preserves_raw_and_is_deterministic(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    source = raw_root / "INDEX.csv"
    _write_fixture(source)
    original_hash = _sha256(source)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_cleaned_ohlc(
        raw_root,
        first,
        expected_structural_flags=0,
        expected_affected_indices=0,
    )
    second_manifest = build_cleaned_ohlc(
        raw_root,
        second,
        expected_structural_flags=0,
        expected_affected_indices=0,
    )

    assert _sha256(source) == original_hash
    assert first_manifest["basic_removed_rows"] == 3
    assert first_manifest["structural_removed_rows"] == 0
    assert (first / "row_corrections.csv").read_bytes() == (
        second / "row_corrections.csv"
    ).read_bytes()
    assert (first / "individual_indices_data/INDEX.csv").read_bytes() == (
        second / "individual_indices_data/INDEX.csv"
    ).read_bytes()


def test_build_cleaned_ohlc_refuses_unexpected_structural_count(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_fixture(raw_root / "INDEX.csv")

    with pytest.raises(RuntimeError, match="Structural correction/source-lineage mismatch"):
        build_cleaned_ohlc(
            raw_root,
            tmp_path / "output",
            expected_structural_flags=1,
            expected_affected_indices=1,
        )
