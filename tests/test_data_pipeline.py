from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PROCESSED_DATASETS = [
    REPO_ROOT / "data" / "processed" / "phase2_spy_vix_volatility.csv",
    REPO_ROOT / "data" / "processed" / "phase2_spy_vix_modeling_frame.csv",
    REPO_ROOT / "data" / "processed" / "spy_vix_volatility.csv",
]
TARGET_COLUMN = "future_rv_20d"
SPLIT_BOUNDARIES = {
    "train_end": pd.Timestamp("2014-12-31"),
    "validation_start": pd.Timestamp("2015-01-01"),
    "validation_end": pd.Timestamp("2019-12-31"),
    "test_start": pd.Timestamp("2020-01-01"),
}


def _processed_dataset_path() -> Path:
    for path in CANDIDATE_PROCESSED_DATASETS:
        if path.exists():
            return path
    pytest.skip(
        "No processed Phase 2 dataset found in expected locations. "
        "Generate or restore the processed SPY/VIX modeling frame before running data-pipeline validation tests."
    )


def _load_processed_dataset() -> pd.DataFrame:
    path = _processed_dataset_path()
    frame = pd.read_csv(path)
    assert not frame.empty, f"Processed dataset is empty: {path}"
    return frame


def _date_column(frame: pd.DataFrame) -> str:
    candidates = ["date", "Date", "timestamp", "datetime"]
    for column in candidates:
        if column in frame.columns:
            return column
    raise AssertionError(f"No date-like column found. Columns: {list(frame.columns)}")


def test_processed_phase2_dataset_has_required_columns():
    frame = _load_processed_dataset()

    assert _date_column(frame) in frame.columns
    assert TARGET_COLUMN in frame.columns


def test_processed_phase2_dates_are_parseable_and_sorted():
    frame = _load_processed_dataset()
    dates = pd.to_datetime(frame[_date_column(frame)], errors="coerce")

    assert dates.notna().all()
    assert dates.is_monotonic_increasing


def test_future_realized_volatility_target_is_finite_and_positive():
    frame = _load_processed_dataset()
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    assert target.notna().all()
    assert np.isfinite(target).all()
    assert (target > 0).all()


def test_chronological_phase2_split_ranges_do_not_overlap():
    frame = _load_processed_dataset()
    dates = pd.to_datetime(frame[_date_column(frame)], errors="raise")

    train_mask = dates <= SPLIT_BOUNDARIES["train_end"]
    validation_mask = (dates >= SPLIT_BOUNDARIES["validation_start"]) & (
        dates <= SPLIT_BOUNDARIES["validation_end"]
    )
    test_mask = dates >= SPLIT_BOUNDARIES["test_start"]

    assert train_mask.any(), "Training split is empty."
    assert validation_mask.any(), "Validation split is empty."
    assert test_mask.any(), "Test split is empty."

    assert dates[train_mask].max() < dates[validation_mask].min()
    assert dates[validation_mask].max() < dates[test_mask].min()


def test_processed_phase2_dataset_contains_feature_columns_beyond_date_and_target():
    frame = _load_processed_dataset()
    excluded = {_date_column(frame), TARGET_COLUMN}
    feature_columns = [column for column in frame.columns if column not in excluded]

    assert len(feature_columns) >= 5
