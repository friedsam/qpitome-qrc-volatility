from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.build_volatility_dataset import CORE_COLUMNS


@dataclass(frozen=True)
class ValidationResult:
    rows: int
    columns: int
    date_start: str
    date_end: str
    reference_overlap_rows: int | None = None
    reference_numeric_mae: float | None = None


def validate_volatility_dataset(
    dataset_path: Path,
    *,
    reference_path: Path | None = None,
) -> ValidationResult:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    frame = pd.read_csv(dataset_path, parse_dates=["date"])
    required = {"date", *CORE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Processed dataset missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Processed dataset is empty")
    if frame["date"].duplicated().any():
        raise ValueError("Processed dataset contains duplicate dates")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Processed dataset dates are not sorted")
    if frame[CORE_COLUMNS].isna().any().any():
        raise ValueError("Processed dataset contains null values in core columns")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Processed dataset contains non-finite numeric values")

    overlap_rows: int | None = None
    numeric_mae: float | None = None
    if reference_path is not None:
        reference = pd.read_csv(reference_path, parse_dates=["date"])
        shared_columns = [
            column
            for column in frame.columns
            if column in reference.columns
            and column != "date"
            and pd.api.types.is_numeric_dtype(frame[column])
            and pd.api.types.is_numeric_dtype(reference[column])
        ]
        joined = frame[["date", *shared_columns]].merge(
            reference[["date", *shared_columns]],
            on="date",
            suffixes=("_new", "_reference"),
            how="inner",
        )
        overlap_rows = len(joined)
        if overlap_rows and shared_columns:
            differences = []
            for column in shared_columns:
                differences.append(
                    np.abs(
                        joined[f"{column}_new"].to_numpy(dtype=float)
                        - joined[f"{column}_reference"].to_numpy(dtype=float)
                    )
                )
            numeric_mae = float(np.concatenate(differences).mean())

    return ValidationResult(
        rows=len(frame),
        columns=len(frame.columns),
        date_start=frame["date"].min().date().isoformat(),
        date_end=frame["date"].max().date().isoformat(),
        reference_overlap_rows=overlap_rows,
        reference_numeric_mae=numeric_mae,
    )
