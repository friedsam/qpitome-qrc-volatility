"""End-to-end long-history replication from the Yahoo GSPC raw source.

This wrapper reconstructs the 1950--2026 daily volatility substrate and the
frozen Part B branch episodes, saves both for reproducibility, then runs the
locked HAR-residual replication.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    OutcomeConfig,
    extract_labeled_branch_episodes,
)


RAW_JSON = Path("data/raw/paper_monthly/20260705T170000Z/gspc_daily_yahoo.json")
RECON_DIR = Path("results/regimes/long_history_branch_reconstruction_v1")
DAILY_CSV = RECON_DIR / "gspc_daily_engineered_1950_2026.csv"
EPISODE_CSV = RECON_DIR / "branch_episodes_1950_2026.csv"
OUTPUT_DIR = Path("results/baselines/long_history_har_residual_replication_v1")

BASE_RUNNER = Path(__file__).with_name("run_long_history_har_residual_replication.py")
spec = importlib.util.spec_from_file_location("long_history_replication_base", BASE_RUNNER)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {BASE_RUNNER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def parse_yahoo_json(path: Path) -> pd.DataFrame:
    """Parse Yahoo chart JSON or a simple record-oriented JSON export."""
    payload = json.loads(path.read_text())

    if isinstance(payload, dict) and "chart" in payload:
        result = payload.get("chart", {}).get("result")
        if not result:
            raise ValueError("Yahoo chart JSON has no result payload")
        block = result[0]
        timestamps = block.get("timestamp", [])
        indicators = block.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
        close = adj if adj is not None else quote.get("close")
        if len(timestamps) != len(close):
            raise ValueError("Yahoo timestamp/price lengths differ")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                "Adj Close": pd.to_numeric(close, errors="coerce"),
            }
        )

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        records = payload.get("data") or payload.get("records") or payload.get("prices")
        if not isinstance(records, list):
            raise ValueError(f"Unsupported JSON keys: {sorted(payload.keys())[:20]}")
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported JSON root type: {type(payload).__name__}")

    date_col = next((c for c in ("date", "Date", "timestamp", "datetime") if c in df.columns), None)
    price_col = next((c for c in ("Adj Close", "adjclose", "adj_close", "close", "Close") if c in df.columns), None)
    if date_col is None or price_col is None:
        raise KeyError(f"Could not identify date/price columns in JSON records: {list(df.columns)}")

    dates = df[date_col]
    if pd.api.types.is_numeric_dtype(dates):
        unit = "ms" if pd.to_numeric(dates, errors="coerce").dropna().median() > 1e11 else "s"
        parsed_dates = pd.to_datetime(dates, unit=unit, utc=True, errors="coerce").dt.tz_convert(None)
    else:
        parsed_dates = pd.to_datetime(dates, errors="coerce", utc=True).dt.tz_convert(None)

    return pd.DataFrame(
        {
            "date": parsed_dates.dt.normalize(),
            "Adj Close": pd.to_numeric(df[price_col], errors="coerce"),
        }
    )


def engineer_daily(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.dropna(subset=["date", "Adj Close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    # Match the earlier Part B period rather than using pre-1950 rows.
    out = out[out["date"] >= pd.Timestamp("1950-01-01")].reset_index(drop=True)
    price = out["Adj Close"].astype(float)
    r = np.log(price).diff()
    annualizer = np.sqrt(252.0)

    out["market_price"] = price
    out["market_log_return"] = r
    out["spy_adj_close"] = price
    out["spy_log_return"] = r
    out["rv_5d"] = r.rolling(5).std(ddof=1) * annualizer
    out["rv_10d"] = r.rolling(10).std(ddof=1) * annualizer
    out["rv_20d"] = r.rolling(20).std(ddof=1) * annualizer
    out["rv_60d"] = r.rolling(60).std(ddof=1) * annualizer
    out["future_rv_20d"] = out["rv_20d"].shift(-20)
    return out


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_JSON.exists():
        raise FileNotFoundError(f"Missing expected Yahoo GSPC source: {RAW_JSON}")

    raw = parse_yahoo_json(RAW_JSON)
    daily = engineer_daily(raw)

    # Frozen Part B mid-grid configuration from the earlier long-history audit.
    state_cfg = BranchStateConfig(
        expanding_min_periods=504,
        stress_quantile=0.70,
        drawdown_threshold=-0.06,
        rv_ratio_cap=1.00,
        stabilization_return_floor=-0.005,
        prior_decline_threshold=-0.05,
        prior_decline_lookback=20,
        merge_gap_days=3,
        min_episode_separation=20,
    )
    outcome_cfg = OutcomeConfig(
        horizon=40,
        recovery_scale=0.55,
        relapse_scale=0.70,
    )

    _, episodes = extract_labeled_branch_episodes(daily, state_cfg, outcome_cfg)
    episodes = episodes[episodes["outcome_complete"]].copy().reset_index(drop=True)

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_CSV, index=False)
    episodes.to_csv(EPISODE_CSV, index=False)
    (RECON_DIR / "reconstruction_manifest.json").write_text(
        json.dumps(
            {
                "raw_source": str(RAW_JSON),
                "daily_rows": len(daily),
                "daily_start": str(daily["date"].min().date()),
                "daily_end": str(daily["date"].max().date()),
                "episode_count": len(episodes),
                "outcome_counts": episodes["outcome"].value_counts().to_dict(),
                "state_config": state_cfg.__dict__,
                "outcome_config": outcome_cfg.__dict__,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Reconstructed daily rows: {len(daily)}")
    print(f"Date range: {daily['date'].min().date()}..{daily['date'].max().date()}")
    print(f"Reconstructed complete episodes: {len(episodes)}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Saved reconstruction: {RECON_DIR}")
    return daily, episodes


def main() -> None:
    reconstruct()

    # Reuse the locked replication runner with explicit reconstructed paths.
    sys.argv = [
        str(BASE_RUNNER),
        "--data",
        str(DAILY_CSV),
        "--episodes",
        str(EPISODE_CSV),
        "--output",
        str(OUTPUT_DIR),
    ]
    base.main()


if __name__ == "__main__":
    main()
