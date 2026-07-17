from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EARLY_END = pd.Timestamp("1950-01-01")
VALUE_COLUMNS = ["open", "high", "low", "close"]


def audit_early_ohlc_file(path: Path, early_end: pd.Timestamp = EARLY_END) -> tuple[dict[str, object], pd.DataFrame]:
    frame = pd.read_csv(path)
    columns = {column.strip().lower(): column for column in frame.columns}
    required = {"date", "high", "low"}
    if not required.issubset(columns):
        raise ValueError(f"{path}: missing date/high/low columns")

    normalized = pd.DataFrame({"date": pd.to_datetime(frame[columns["date"]], errors="coerce")})
    available_values: list[str] = []
    for name in VALUE_COLUMNS:
        if name in columns:
            normalized[name] = pd.to_numeric(frame[columns[name]], errors="coerce")
            available_values.append(name)

    normalized = normalized.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    early = normalized[normalized["date"] < early_end].copy()
    if early.empty:
        summary = {
            "path": str(path),
            "early_end_exclusive": str(early_end.date()),
            "early_rows": 0,
            "exact_duplicate_value_rows": 0,
            "consecutive_identical_value_rows": 0,
            "maximum_identical_run_length": 0,
            "zero_intraday_range_rows": 0,
        }
        return summary, pd.DataFrame()

    duplicate_mask = early.duplicated(subset=available_values, keep=False)
    consecutive_mask = early[available_values].eq(early[available_values].shift()).all(axis=1)
    zero_range_mask = early["high"].eq(early["low"])

    run_ids = (~consecutive_mask).cumsum()
    run_lengths = early.groupby(run_ids).size()
    repeated_runs = []
    for run_id, length in run_lengths.items():
        if length < 2:
            continue
        group = early.loc[run_ids == run_id]
        repeated_runs.append(
            {
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "run_length": int(length),
                **{column: group[column].iloc[0] for column in available_values},
            }
        )

    details = pd.DataFrame(repeated_runs)
    summary = {
        "path": str(path),
        "early_end_exclusive": str(early_end.date()),
        "early_start": str(early["date"].min().date()),
        "early_last": str(early["date"].max().date()),
        "early_rows": int(len(early)),
        "exact_duplicate_value_rows": int(duplicate_mask.sum()),
        "exact_duplicate_value_fraction": float(duplicate_mask.mean()),
        "consecutive_identical_value_rows": int(consecutive_mask.sum()),
        "consecutive_identical_value_fraction": float(consecutive_mask.mean()),
        "maximum_identical_run_length": int(run_lengths.max()),
        "zero_intraday_range_rows": int(zero_range_mask.sum()),
        "zero_intraday_range_fraction": float(zero_range_mask.mean()),
        "available_value_columns": available_values,
    }
    return summary, details


def write_early_ohlc_audit(path: Path, run_dir: Path, early_end: pd.Timestamp = EARLY_END) -> dict[str, object]:
    summary, details = audit_early_ohlc_file(path, early_end)
    details.to_csv(run_dir / "early_identical_value_runs.csv", index=False)
    (run_dir / "early_ohlc_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
