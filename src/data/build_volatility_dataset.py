from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

MARKET_REQUIRED = {"date", "open", "high", "low", "close", "volume", "adjusted_close"}
VOLATILITY_REQUIRED = {"date", "open", "high", "low", "close"}
CORE_COLUMNS = [
    "spy_log_return",
    "spy_abs_log_return",
    "spy_squared_log_return",
    "spy_log_hl_range",
    "spy_log_volume",
    "spy_log_volume_change",
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "future_rv_5d",
    "future_rv_20d",
    "vix_close",
    "vix_log_change",
    "vix_log_hl_range",
    "vix_ma_5d",
    "vix_ma_20d",
]


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    manifest_path: Path
    rows: int
    columns: int
    date_start: str
    date_end: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_output_name(output_name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", output_name):
        raise ValueError(
            "output_name must start with a lowercase letter or digit and contain "
            "only lowercase letters, digits, underscores, and hyphens"
        )
    return output_name


def _read_normalized(
    path: Path,
    *,
    required: set[str],
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if column_mapping:
        normalized_mapping = {
            str(source).strip().lower(): str(target).strip().lower()
            for source, target in column_mapping.items()
        }
        frame = frame.rename(columns=normalized_mapping)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def trailing_rv(log_return: pd.Series, window: int) -> pd.Series:
    return np.sqrt(252.0 / window * log_return.pow(2).rolling(window).sum())


def forward_rv(log_return: pd.Series, horizon: int) -> pd.Series:
    return np.sqrt(
        252.0
        / horizon
        * log_return.pow(2).shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    )


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["spy_log_return"] = np.log(out["spy_adj_close"]).diff()
    out["spy_abs_log_return"] = out["spy_log_return"].abs()
    out["spy_squared_log_return"] = out["spy_log_return"].pow(2)

    out["spy_log_hl_range"] = np.log(out["spy_high"] / out["spy_low"])
    out["spy_log_oc_return"] = np.log(out["spy_close"] / out["spy_open"])
    out["parkinson_var"] = out["spy_log_hl_range"].pow(2) / (4.0 * np.log(2.0))
    out["garman_klass_var"] = (
        0.5 * out["spy_log_hl_range"].pow(2)
        - (2.0 * np.log(2.0) - 1.0) * out["spy_log_oc_return"].pow(2)
    )

    out["spy_log_volume"] = np.log(out["spy_volume"].replace(0, np.nan))
    out["spy_log_volume_change"] = out["spy_log_volume"].diff()
    out["spy_dollar_volume"] = out["spy_adj_close"] * out["spy_volume"]
    out["spy_log_dollar_volume"] = np.log(
        out["spy_dollar_volume"].replace(0, np.nan)
    )

    for window in (5, 10, 20, 60):
        out[f"rv_{window}d"] = trailing_rv(out["spy_log_return"], window)
    for horizon in (5, 20):
        out[f"future_rv_{horizon}d"] = forward_rv(
            out["spy_log_return"], horizon
        )

    out["rv_ratio_5_20"] = out["rv_5d"] / out["rv_20d"]
    out["rv_ratio_20_60"] = out["rv_20d"] / out["rv_60d"]
    out["rv_slope_5_20"] = out["rv_5d"] - out["rv_20d"]
    out["rv_slope_20_60"] = out["rv_20d"] - out["rv_60d"]

    rolling_max_20d = out["spy_adj_close"].rolling(window=20).max()
    out["spy_drawdown_20d"] = out["spy_adj_close"] / rolling_max_20d - 1.0

    out["vix_log_close"] = np.log(out["vix_close"])
    out["vix_log_change"] = out["vix_log_close"].diff()
    out["vix_abs_log_change"] = out["vix_log_change"].abs()
    out["vix_log_hl_range"] = np.log(out["vix_high"] / out["vix_low"])
    for window in (5, 20):
        out[f"vix_ma_{window}d"] = out["vix_close"].rolling(window).mean()
        out[f"vix_std_{window}d"] = out["vix_close"].rolling(window).std()
    return out


def build_volatility_frame(
    market_data_path: Path,
    volatility_data_path: Path,
    *,
    market_column_mapping: Mapping[str, str] | None = None,
    volatility_column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    market = _read_normalized(
        market_data_path,
        required=MARKET_REQUIRED,
        column_mapping=market_column_mapping,
    ).rename(
        columns={
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "volume": "spy_volume",
            "adjusted_close": "spy_adj_close",
        }
    )
    volatility = _read_normalized(
        volatility_data_path,
        required=VOLATILITY_REQUIRED,
        column_mapping=volatility_column_mapping,
    ).rename(
        columns={
            "open": "vix_open",
            "high": "vix_high",
            "low": "vix_low",
            "close": "vix_close",
        }
    )

    market = market[
        [
            "date",
            "spy_open",
            "spy_high",
            "spy_low",
            "spy_close",
            "spy_adj_close",
            "spy_volume",
        ]
    ]
    volatility = volatility[
        ["date", "vix_open", "vix_high", "vix_low", "vix_close"]
    ]
    merged = market.merge(volatility, on="date", how="inner", validate="one_to_one")
    processed = add_features(merged)
    return processed.dropna(subset=CORE_COLUMNS).reset_index(drop=True)


def build_volatility_dataset(
    *,
    market_data_path: Path,
    volatility_data_path: Path,
    output_root: Path,
    output_name: str,
    market_column_mapping: Mapping[str, str] | None = None,
    volatility_column_mapping: Mapping[str, str] | None = None,
) -> BuildResult:
    output_name = _validate_output_name(output_name)
    frame = build_volatility_frame(
        market_data_path,
        volatility_data_path,
        market_column_mapping=market_column_mapping,
        volatility_column_mapping=volatility_column_mapping,
    )
    if frame.empty:
        raise RuntimeError("No processed rows remain after feature construction")

    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.csv"
    manifest_path = output_dir / "manifest.json"
    frame.to_csv(output_path, index=False)

    manifest = {
        "output_name": output_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "src/data/build_volatility_dataset.py",
        "inputs": {
            "market_data_path": str(market_data_path),
            "market_data_sha256": _sha256(market_data_path),
            "volatility_data_path": str(volatility_data_path),
            "volatility_data_sha256": _sha256(volatility_data_path),
        },
        "output": {
            "path": str(output_path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "column_names": list(frame.columns),
            "date_start": frame["date"].min().date().isoformat(),
            "date_end": frame["date"].max().date().isoformat(),
            "sha256": _sha256(output_path),
        },
        "construction": {
            "join": "inner join on date",
            "annualization_days": 252,
            "trailing_rv_windows": [5, 10, 20, 60],
            "forward_rv_horizons": [5, 20],
            "dropna_subset": CORE_COLUMNS,
            "regime_labels_created": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BuildResult(
        output_path=output_path,
        manifest_path=manifest_path,
        rows=len(frame),
        columns=len(frame.columns),
        date_start=frame["date"].min().date().isoformat(),
        date_end=frame["date"].max().date().isoformat(),
    )
