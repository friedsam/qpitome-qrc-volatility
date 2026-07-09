"""Corrected entry point for long-history HAR residual replication.

Uses the canonical raw S&P 500 OHLCV file when no engineered long-history file
is present, and derives the realized-volatility substrate deterministically.
The underlying replication logic remains in run_long_history_har_residual_replication.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).with_name("run_long_history_har_residual_replication.py")
spec = importlib.util.spec_from_file_location("long_history_har_residual_replication", RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {RUNNER}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


CANONICAL_RAW = Path("data/raw/kaggle_market_ohlcv/SP500.csv")


def detect_long_daily_fixed() -> Path:
    """Prefer engineered long data; otherwise use the canonical raw S&P copy."""
    engineered: list[tuple[int, Path]] = []
    for path in mod._candidate_csvs():
        try:
            df = mod._read_csv(path)
            col = mod._date_column(df)
            if col is None or len(df) < 12000:
                continue
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.notna().sum() < 12000:
                continue
            if dates.min().year > 1965 or dates.max().year < 2024:
                continue
            rv_count = sum(c in df.columns for c in (*mod.HAR_FEATURES, mod.HAR_TARGET))
            if rv_count >= 4:
                engineered.append((len(df), path))
        except Exception:
            continue

    if engineered:
        engineered.sort(reverse=True)
        best = engineered[0][1]
        print(f"Auto-detected engineered long daily file: {best}")
        return best

    if CANONICAL_RAW.exists():
        print(f"Using canonical raw long daily file: {CANONICAL_RAW}")
        return CANONICAL_RAW

    raise FileNotFoundError(
        "No engineered long-history file and canonical raw S&P file is missing. "
        "Pass --data PATH."
    )


def normalize_daily_fixed(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize long daily data and derive RV features from prices when needed."""
    out = mod._normalize_date(raw).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    aliases = {
        "rv_5d": ("rv_5d", "rv5", "RV5", "rv5d"),
        "rv_10d": ("rv_10d", "rv10", "RV10", "rv10d"),
        "rv_20d": ("rv_20d", "rv20", "RV20", "rv20d"),
        "rv_60d": ("rv_60d", "rv60", "RV60", "rv60d"),
        "future_rv_20d": ("future_rv_20d", "future_rv20", "rv20_future", "future_rv20d"),
    }
    for canonical, names in aliases.items():
        if canonical not in out.columns:
            found = next((n for n in names if n in out.columns), None)
            if found is not None:
                out = out.rename(columns={found: canonical})

    price_col = next(
        (
            c
            for c in (
                "Adj Close",
                "adj_close",
                "adjusted_close",
                "Close",
                "close",
                "gspc_adj_close",
                "spy_adj_close",
            )
            if c in out.columns
        ),
        None,
    )
    return_col = next(
        (
            c
            for c in ("log_return", "market_log_return", "gspc_log_return", "spy_log_return")
            if c in out.columns
        ),
        None,
    )

    if return_col is not None:
        out["market_log_return"] = pd.to_numeric(out[return_col], errors="coerce")
    elif price_col is not None:
        price = pd.to_numeric(out[price_col], errors="coerce")
        out["market_log_return"] = np.log(price).diff()
    else:
        raise KeyError("Need a price or log-return column in the long daily file")

    if price_col is None:
        out["market_price"] = np.exp(out["market_log_return"].fillna(0.0).cumsum())
    else:
        out["market_price"] = pd.to_numeric(out[price_col], errors="coerce")

    # If the file is raw OHLCV, derive the same annualized close-to-close RV
    # substrate used throughout the project. future_rv_20d at row t is the
    # realized 20-day volatility ending at t+20, i.e. the next 20 return rows.
    if any(c not in out.columns for c in (*mod.HAR_FEATURES, mod.HAR_TARGET)):
        r = out["market_log_return"]
        annualizer = np.sqrt(252.0)
        out["rv_5d"] = r.rolling(5).std(ddof=1) * annualizer
        out["rv_10d"] = r.rolling(10).std(ddof=1) * annualizer
        out["rv_20d"] = r.rolling(20).std(ddof=1) * annualizer
        out["rv_60d"] = r.rolling(60).std(ddof=1) * annualizer
        out["future_rv_20d"] = out["rv_20d"].shift(-20)
        print("Derived RV5/RV10/RV20/RV60 and future RV20 from raw prices.")

    for col in (*mod.HAR_FEATURES, mod.HAR_TARGET):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


mod.detect_long_daily = detect_long_daily_fixed
mod.normalize_daily = normalize_daily_fixed

if __name__ == "__main__":
    mod.main()
