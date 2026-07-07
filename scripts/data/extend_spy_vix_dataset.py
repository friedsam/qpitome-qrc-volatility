#!/usr/bin/env python3
"""Extend the frozen Phase 2 SPY/VIX dataset without modifying it.

The script downloads a recent raw tail from Yahoo's chart API, combines it with
raw columns preserved in the frozen processed dataset, recomputes features only
to obtain valid rolling context for new dates, and writes a separate extended
CSV. Historical rows through the frozen cutoff are copied verbatim from the
frozen dataset.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from prepare_phase2_spy_vix_dataset import add_features

FROZEN_DEFAULT = Path("data/processed/phase2_spy_vix_volatility.csv")
OUT_DEFAULT = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
RAW_COLS = [
    "date",
    "spy_open", "spy_high", "spy_low", "spy_close", "spy_adj_close", "spy_volume",
    "vix_open", "vix_high", "vix_low", "vix_close",
]
CORE_COLS = [
    "spy_log_return", "spy_abs_log_return", "spy_squared_log_return",
    "spy_log_hl_range", "spy_log_volume", "spy_log_volume_change",
    "rv_5d", "rv_10d", "rv_20d", "rv_60d",
    "future_rv_5d", "future_rv_20d",
    "vix_close", "vix_log_change", "vix_log_hl_range", "vix_ma_5d", "vix_ma_20d",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, default=FROZEN_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--start", default="2024-01-01", help="Recent raw-tail download start")
    return p.parse_args()


def fetch_chart(symbol: str, start: str, end: datetime) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(end.timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r:
        payload = json.load(r)
    result = payload["chart"]["result"][0]
    ts = result.get("timestamp", [])
    quote_block = result["indicators"]["quote"][0]
    adj_block = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "open": quote_block.get("open"),
        "high": quote_block.get("high"),
        "low": quote_block.get("low"),
        "close": quote_block.get("close"),
        "volume": quote_block.get("volume"),
    })
    if adj_block is not None:
        df["adjusted_close"] = adj_block
    return df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")


def recent_raw(start: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc) + timedelta(days=1)
    spy = fetch_chart("SPY", start, end).rename(columns={
        "open":"spy_open", "high":"spy_high", "low":"spy_low", "close":"spy_close",
        "adjusted_close":"spy_adj_close", "volume":"spy_volume",
    })
    vix = fetch_chart("^VIX", start, end).rename(columns={
        "open":"vix_open", "high":"vix_high", "low":"vix_low", "close":"vix_close",
    })
    return spy[["date","spy_open","spy_high","spy_low","spy_close","spy_adj_close","spy_volume"]].merge(
        vix[["date","vix_open","vix_high","vix_low","vix_close"]], on="date", how="inner"
    )


def main():
    a = parse_args()
    frozen = pd.read_csv(a.frozen, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    missing = sorted(set(RAW_COLS) - set(frozen.columns))
    if missing:
        raise ValueError(f"Frozen dataset missing raw columns needed for extension: {missing}")
    cutoff = frozen["date"].max()

    recent = recent_raw(a.start)
    if recent["date"].max() <= cutoff:
        raise RuntimeError("No newer market rows were downloaded")

    historical_raw = frozen[RAW_COLS].copy()
    tail = recent[recent["date"] > cutoff][RAW_COLS].copy()
    combined_raw = pd.concat([historical_raw, tail], ignore_index=True).sort_values("date").drop_duplicates("date", keep="first")
    rebuilt = add_features(combined_raw).dropna(subset=CORE_COLS).reset_index(drop=True)

    new_rows = rebuilt[rebuilt["date"] > cutoff].copy()
    if new_rows.empty:
        raise RuntimeError("No fully labeled new rows after feature/target construction")

    # Preserve every historical value exactly; only append newly constructed rows.
    common = [c for c in frozen.columns if c in new_rows.columns]
    new_rows = new_rows[common]
    extended = pd.concat([frozen[common], new_rows], ignore_index=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(a.out, index=False)

    print(f"Frozen dataset: {a.frozen}")
    print(f"Frozen cutoff:  {cutoff.date()}")
    print(f"Downloaded raw tail through: {recent['date'].max().date()}")
    print(f"New labeled rows: {len(new_rows)}")
    print(f"Extended range: {extended['date'].min().date()} -> {extended['date'].max().date()}")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
