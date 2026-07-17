from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
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
    market_status: str
    volatility_status: str


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


def _validate_history_file(path: Path, *, require_adjusted_close: bool) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"history file is empty: {path}")

    frame = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if require_adjusted_close:
        required.add("adjusted_close")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    pd.to_datetime(frame["date"], errors="raise")
    return frame


def _acquire_history(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    fallback_path: Path | None,
    require_adjusted_close: bool,
    history_fetcher: Callable[[str, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, str, str | None]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".tmp"

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

    try:
        try:
            frame = history_fetcher(symbol, start_date, end_date)
            frame.to_csv(temporary_path, index=False)
            validated = _validate_history_file(
                temporary_path,
                require_adjusted_close=require_adjusted_close,
            )
            temporary_path.replace(output_path)
            return validated, "downloaded", None
        except Exception as remote_error:
            temporary_path.unlink(missing_ok=True)
            if fallback_path is None:
                raise RuntimeError(
                    f"Yahoo download failed for {symbol}, and no fallback path is configured"
                ) from remote_error

            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.fallback.",
                suffix=suffix,
                delete=False,
            ) as handle:
                fallback_temporary_path = Path(handle.name)

            try:
                if not fallback_path.exists() or not fallback_path.is_file():
                    raise FileNotFoundError(fallback_path)
                if fallback_path.stat().st_size == 0:
                    raise RuntimeError(f"fallback file is empty: {fallback_path}")
                shutil.copy2(fallback_path, fallback_temporary_path)
                validated = _validate_history_file(
                    fallback_temporary_path,
                    require_adjusted_close=require_adjusted_close,
                )
                fallback_temporary_path.replace(output_path)
            except Exception as fallback_error:
                fallback_temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Yahoo download failed for {symbol}, and fallback is unavailable "
                    f"or invalid at {fallback_path}. Remote error: "
                    f"{type(remote_error).__name__}: {remote_error}. Fallback error: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error

            return (
                validated,
                "fallback_copied",
                f"{type(remote_error).__name__}: {remote_error}",
            )
    finally:
        temporary_path.unlink(missing_ok=True)


def download_market_data(
    *,
    market_symbol: str,
    volatility_symbol: str,
    start_date: str,
    end_date: str,
    market_output_path: Path,
    volatility_output_path: Path,
    manifest_path: Path,
    market_fallback_path: Path | None = None,
    volatility_fallback_path: Path | None = None,
    history_fetcher: Callable[[str, str, str], pd.DataFrame] = fetch_yahoo_history,
) -> DownloadResult:
    """Acquire complete raw market and volatility histories.

    Each series is downloaded independently. A failed download copies and
    validates its explicitly configured fallback snapshot.
    """
    market, market_status, market_error = _acquire_history(
        symbol=market_symbol,
        start_date=start_date,
        end_date=end_date,
        output_path=market_output_path,
        fallback_path=market_fallback_path,
        require_adjusted_close=True,
        history_fetcher=history_fetcher,
    )
    volatility, volatility_status, volatility_error = _acquire_history(
        symbol=volatility_symbol,
        start_date=start_date,
        end_date=end_date,
        output_path=volatility_output_path,
        fallback_path=volatility_fallback_path,
        require_adjusted_close=False,
        history_fetcher=history_fetcher,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "provider": "Yahoo Finance chart API",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "remote_first_explicit_fallback_copy",
        "requested_start_date": start_date,
        "requested_end_date_exclusive": end_date,
        "series": {
            "market": {
                "symbol": market_symbol,
                "path": str(market_output_path),
                "fallback_path": (
                    str(market_fallback_path) if market_fallback_path else None
                ),
                "status": market_status,
                "remote_error": market_error,
                "rows": int(len(market)),
                "date_start": pd.to_datetime(market["date"]).min().date().isoformat(),
                "date_end": pd.to_datetime(market["date"]).max().date().isoformat(),
                "columns": list(market.columns),
                "sha256": _sha256(market_output_path),
            },
            "volatility": {
                "symbol": volatility_symbol,
                "path": str(volatility_output_path),
                "fallback_path": (
                    str(volatility_fallback_path)
                    if volatility_fallback_path
                    else None
                ),
                "status": volatility_status,
                "remote_error": volatility_error,
                "rows": int(len(volatility)),
                "date_start": pd.to_datetime(volatility["date"])
                .min()
                .date()
                .isoformat(),
                "date_end": pd.to_datetime(volatility["date"])
                .max()
                .date()
                .isoformat(),
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
        market_status=market_status,
        volatility_status=volatility_status,
    )
