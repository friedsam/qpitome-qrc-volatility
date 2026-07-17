from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

NAMES = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dates_strict(values: pd.Series, source: Path) -> pd.DatetimeIndex:
    raw = values.astype(str).str.strip()
    successful: list[tuple[str, pd.Series]] = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().all():
            successful.append((fmt, parsed))
    if not successful:
        raise ValueError(f"{source}: Date does not match accepted formats {DATE_FORMATS}")
    unique = {tuple(parsed.astype("int64")) for _, parsed in successful}
    if len(unique) > 1:
        raise ValueError(f"{source}: ambiguous date interpretation")
    return pd.DatetimeIndex(successful[0][1])


def load_ohlc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", "High", "Low"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["Date"] = parse_dates_strict(frame["Date"], path)
    for column in ["Price", "Open", "High", "Low"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    if frame["Date"].duplicated().any():
        raise ValueError(f"{path}: duplicate dates")
    return frame.set_index("Date").sort_index()


def audit_data(root: Path) -> dict[str, object]:
    audit: dict[str, object] = {"ohlc": {}, "rv_panel": {}}
    ohlc = audit["ohlc"]
    assert isinstance(ohlc, dict)
    for name in NAMES:
        path = root / "global index etf return" / f"{name}.csv"
        frame = load_ohlc(path)
        invalid = int(((frame.High < frame.Low) | (frame.High <= 0) | (frame.Low <= 0)).sum())
        ohlc[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "missing": frame[["High", "Low"]].isna().sum().astype(int).to_dict(),
            "invalid_ranges": invalid,
        }
    rv_path = root / "rv_dataset.csv"
    if rv_path.exists():
        rv = pd.read_csv(rv_path)
        date_columns = [c for c in rv.columns if c.lower() in {"date", "dateid", "datetime", "timestamp"}]
        audit["rv_panel"] = {
            "exists": True,
            "path": str(rv_path),
            "sha256": sha256_file(rv_path),
            "rows": int(len(rv)),
            "columns": list(rv.columns),
            "has_exact_date_column": bool(date_columns),
            "date_columns": date_columns,
            "usable_for_dated_concordance": bool(date_columns),
            "note": "Undated rows must never be mapped to an inferred trading calendar.",
        }
    else:
        audit["rv_panel"] = {"exists": False}
    return audit
