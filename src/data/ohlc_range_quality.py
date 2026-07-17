from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def annual_range_quality(path: Path, minimum_nonzero_fraction: float = 0.95) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    columns = {column.strip().lower(): column for column in frame.columns}
    required = {"date", "high", "low"}
    if not required.issubset(columns):
        raise ValueError(f"{path}: missing date/high/low columns")

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[columns["date"]], errors="coerce"),
            "high": pd.to_numeric(frame[columns["high"]], errors="coerce"),
            "low": pd.to_numeric(frame[columns["low"]], errors="coerce"),
        }
    ).dropna()
    data = data[(data["high"] > 0) & (data["low"] > 0) & (data["high"] >= data["low"])]
    data["year"] = data["date"].dt.year
    data["nonzero_range"] = data["high"] > data["low"]

    annual = (
        data.groupby("year", as_index=False)
        .agg(rows=("date", "size"), nonzero_range_rows=("nonzero_range", "sum"))
    )
    annual["zero_range_rows"] = annual["rows"] - annual["nonzero_range_rows"]
    annual["nonzero_range_fraction"] = annual["nonzero_range_rows"] / annual["rows"]
    annual["meets_quality_threshold"] = annual["nonzero_range_fraction"] >= minimum_nonzero_fraction

    qualifying = annual[annual["meets_quality_threshold"]]
    first_qualifying_year = int(qualifying["year"].min()) if not qualifying.empty else None
    first_nonzero_date = data.loc[data["nonzero_range"], "date"].min()

    zero_mask = ~data["nonzero_range"]
    run_ids = (~zero_mask).cumsum()
    zero_run_lengths = data[zero_mask].groupby(run_ids[zero_mask]).size()
    longest_zero_run = int(zero_run_lengths.max()) if not zero_run_lengths.empty else 0

    summary = {
        "path": str(path),
        "minimum_nonzero_fraction": minimum_nonzero_fraction,
        "first_nonzero_range_date": None if pd.isna(first_nonzero_date) else str(first_nonzero_date.date()),
        "first_qualifying_year": first_qualifying_year,
        "recommended_effective_start": None if first_qualifying_year is None else f"{first_qualifying_year}-01-01",
        "longest_consecutive_zero_range_run": longest_zero_run,
        "total_rows": int(len(data)),
        "total_nonzero_range_rows": int(data["nonzero_range"].sum()),
        "total_zero_range_rows": int((~data["nonzero_range"]).sum()),
    }
    return annual, summary


def write_annual_range_quality(path: Path, run_dir: Path, minimum_nonzero_fraction: float = 0.95) -> dict[str, object]:
    annual, summary = annual_range_quality(path, minimum_nonzero_fraction)
    annual.to_csv(run_dir / "annual_range_quality.csv", index=False)
    (run_dir / "range_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
