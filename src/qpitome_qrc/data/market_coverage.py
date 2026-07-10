"""Coverage diagnostics for candidate equity-market portability data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_yfinance_history(history: pd.DataFrame) -> pd.DataFrame:
    """Return a simple date/close/adj_close frame from yfinance output."""
    if history.empty:
        return pd.DataFrame(columns=["date", "close", "adj_close"])

    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]) for column in frame.columns]

    frame = frame.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    if "Close" not in frame.columns:
        raise KeyError("Downloaded history missing Close column")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], utc=True)
            .dt.tz_convert(None)
            .dt.normalize(),
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
        }
    )
    if "Adj Close" in frame.columns:
        out["adj_close"] = pd.to_numeric(frame["Adj Close"], errors="coerce")
    else:
        out["adj_close"] = out["close"]

    return out.sort_values("date").reset_index(drop=True)


def summarize_market_coverage(
    frame: pd.DataFrame,
    *,
    market: str,
    symbol: str,
) -> dict[str, object]:
    """Summarize raw daily-price coverage and obvious data-quality problems."""
    required = {"date", "close", "adj_close"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing coverage columns: {sorted(missing)}")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid_dates = dates.dropna().sort_values()
    gaps = valid_dates.diff().dt.days.dropna()

    return {
        "market": market,
        "symbol": symbol,
        "first_date": valid_dates.min().date().isoformat() if len(valid_dates) else None,
        "last_date": valid_dates.max().date().isoformat() if len(valid_dates) else None,
        "rows": int(len(frame)),
        "duplicate_dates": int(dates.duplicated().sum()),
        "missing_dates": int(dates.isna().sum()),
        "missing_close": int(frame["close"].isna().sum()),
        "missing_adj_close": int(frame["adj_close"].isna().sum()),
        "nonpositive_close": int((frame["close"] <= 0).fillna(False).sum()),
        "nonpositive_adj_close": int((frame["adj_close"] <= 0).fillna(False).sum()),
        "longest_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
        "median_calendar_gap_days": float(np.median(gaps)) if len(gaps) else np.nan,
    }
