from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class DownloadResult:
    market_path: Path
    volatility_path: Path
    manifest_path: Path
    market_rows: int
    volatility_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_yahoo_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily OHLCV history from Yahoo's public chart endpoint.

    ``end_date`` is exclusive, matching the Yahoo chart API.
    """
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    if end <= start:
        raise ValueError("end_date must be later than start_date")

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo download failed for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_block = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (
        (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert(None)
            .normalize(),
            "open": quote_block.get("open"),
            "high": quote_block.get("high"),
            "low": quote_block.get("low"),
            "close": quote_block.get("close"),
            "volume": quote_block.get("volume"),
        }
    )
    if adjusted is not None:
        frame["adjusted_close"] = adjusted

    frame = (
        frame.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no usable rows for {symbol}")
    return frame


def download_market_data(
    *,
    market_symbol: str,
    volatility_symbol: str,
    start_date: str,
    end_date: str,
    market_output_path: Path,
    volatility_output_path: Path,
    manifest_path: Path,
) -> DownloadResult:
    """Download complete raw market and volatility histories.

    Raw filenames are caller-controlled. The derived dataset name is intentionally
    not part of this function because one set of raw files may feed many outputs.
    """
    market = fetch_yahoo_history(market_symbol, start_date, end_date)
    volatility = fetch_yahoo_history(volatility_symbol, start_date, end_date)

    market_output_path.parent.mkdir(parents=True, exist_ok=True)
    volatility_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    market.to_csv(market_output_path, index=False)
    volatility.to_csv(volatility_output_path, index=False)

    manifest = {
        "provider": "Yahoo Finance chart API",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start_date": start_date,
        "requested_end_date_exclusive": end_date,
        "series": {
            "market": {
                "symbol": market_symbol,
                "path": str(market_output_path),
                "rows": int(len(market)),
                "date_start": market["date"].min().date().isoformat(),
                "date_end": market["date"].max().date().isoformat(),
                "columns": list(market.columns),
                "sha256": _sha256(market_output_path),
            },
            "volatility": {
                "symbol": volatility_symbol,
                "path": str(volatility_output_path),
                "rows": int(len(volatility)),
                "date_start": volatility["date"].min().date().isoformat(),
                "date_end": volatility["date"].max().date().isoformat(),
                "columns": list(volatility.columns),
                "sha256": _sha256(volatility_output_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return DownloadResult(
        market_path=market_output_path,
        volatility_path=volatility_output_path,
        manifest_path=manifest_path,
        market_rows=len(market),
        volatility_rows=len(volatility),
    )
