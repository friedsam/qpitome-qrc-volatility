from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATE_ALIASES = ("date", "datetime", "timestamp")
HIGH_ALIASES = ("high",)
LOW_ALIASES = ("low",)
OPEN_ALIASES = ("open",)
CLOSE_ALIASES = ("close", "price", "adj close", "adj_close")
VOLUME_ALIASES = ("volume",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _parse_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def audit_file(path: Path, *, min_years: float = 15.0, min_valid_fraction: float = 0.98) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "file": path.name,
        "index": path.stem,
        "sha256": sha256_file(path),
        "eligible": False,
        "failure_reason": None,
    }
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error details vary
        result["failure_reason"] = f"read_error: {exc}"
        return result

    columns = list(frame.columns)
    date_col = _find_column(columns, DATE_ALIASES)
    high_col = _find_column(columns, HIGH_ALIASES)
    low_col = _find_column(columns, LOW_ALIASES)
    result["columns"] = columns
    result["date_column"] = date_col
    result["high_column"] = high_col
    result["low_column"] = low_col
    result["open_column"] = _find_column(columns, OPEN_ALIASES)
    result["close_column"] = _find_column(columns, CLOSE_ALIASES)
    result["volume_column"] = _find_column(columns, VOLUME_ALIASES)

    if not date_col or not high_col or not low_col:
        result["failure_reason"] = "missing_required_columns"
        return result

    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    high = _parse_numeric(frame[high_col])
    low = _parse_numeric(frame[low_col])
    valid_dates = dates.notna()
    valid_ranges = high.notna() & low.notna() & (high > 0) & (low > 0) & (high >= low)
    usable = valid_dates & valid_ranges

    rows = int(len(frame))
    usable_rows = int(usable.sum())
    duplicate_dates = int(dates[valid_dates].duplicated().sum())
    invalid_ranges = int(((high < low) | (high <= 0) | (low <= 0)).fillna(False).sum())
    valid_fraction = usable_rows / rows if rows else 0.0

    start = dates[usable].min() if usable_rows else pd.NaT
    end = dates[usable].max() if usable_rows else pd.NaT
    span_years = float((end - start).days / 365.2425) if usable_rows > 1 else 0.0

    sorted_dates = dates[usable].drop_duplicates().sort_values()
    gaps = sorted_dates.diff().dt.days.dropna()
    maximum_gap_days = int(gaps.max()) if not gaps.empty else 0
    gaps_over_10_days = int((gaps > 10).sum()) if not gaps.empty else 0

    eligible = (
        rows > 0
        and valid_fraction >= min_valid_fraction
        and span_years >= min_years
        and duplicate_dates == 0
        and invalid_ranges == 0
    )
    reasons: list[str] = []
    if valid_fraction < min_valid_fraction:
        reasons.append("insufficient_valid_ohlc")
    if span_years < min_years:
        reasons.append("insufficient_history")
    if duplicate_dates:
        reasons.append("duplicate_dates")
    if invalid_ranges:
        reasons.append("invalid_ranges")

    result.update(
        {
            "rows": rows,
            "usable_rows": usable_rows,
            "valid_fraction": valid_fraction,
            "start": None if pd.isna(start) else str(start.date()),
            "end": None if pd.isna(end) else str(end.date()),
            "span_years": span_years,
            "duplicate_dates": duplicate_dates,
            "invalid_ranges": invalid_ranges,
            "maximum_gap_days": maximum_gap_days,
            "gaps_over_10_days": gaps_over_10_days,
            "eligible": eligible,
            "failure_reason": None if eligible else ";".join(reasons) or "not_eligible",
        }
    )
    return result


def audit_directory(
    root: Path,
    *,
    min_years: float = 15.0,
    min_valid_fraction: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, object]]:
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    records = [
        audit_file(path, min_years=min_years, min_valid_fraction=min_valid_fraction)
        for path in files
    ]
    inventory = pd.DataFrame(records)
    eligible_count = int(inventory["eligible"].sum()) if "eligible" in inventory else 0
    summary: dict[str, object] = {
        "root": str(root),
        "csv_files": len(files),
        "eligible_files": eligible_count,
        "ineligible_files": len(files) - eligible_count,
        "min_years": min_years,
        "min_valid_fraction": min_valid_fraction,
    }
    if not inventory.empty and "span_years" in inventory:
        summary["maximum_span_years"] = float(inventory["span_years"].max())
    return inventory, summary
