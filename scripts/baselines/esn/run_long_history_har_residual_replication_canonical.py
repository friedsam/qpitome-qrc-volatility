"""Canonical long-history HAR-residual replication.

This is the only supported entry point for the 1950--2026 branch replication.
It reconstructs the frozen long-history substrate from the Yahoo GSPC snapshot,
verifies realized-volatility parity against the modern processed dataset, applies
exactly the modern frozen branch definition, asserts the expected reconstruction,
and then delegates model fitting to the locked replication engine.

The previous v2/v3 wrappers were removed because they used the wrong long-history
RV construction (rolling standard deviation instead of project RMS RV).
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
MODERN_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
RECON_DIR = Path("results/regimes/long_history_branch_reconstruction_v2")
DAILY_CSV = RECON_DIR / "gspc_daily_engineered_1950_2026.csv"
EPISODE_CSV = RECON_DIR / "branch_episodes_1950_2026.csv"
OUTPUT_DIR = Path("results/baselines/long_history_har_residual_replication_v2")

EXPECTED_ROWS = 19_246
EXPECTED_START = pd.Timestamp("1950-01-03")
EXPECTED_END = pd.Timestamp("2026-07-02")
EXPECTED_COMPLETE_EPISODES = 80
RV_PARITY_ATOL = 1e-12

STATE_CONFIG = BranchStateConfig(
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
OUTCOME_CONFIG = OutcomeConfig(
    horizon=40,
    recovery_scale=0.55,
    relapse_scale=0.70,
)

ENGINE = Path(__file__).with_name("run_long_history_har_residual_replication.py")
spec = importlib.util.spec_from_file_location("long_history_replication_engine", ENGINE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load replication engine: {ENGINE}")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def parse_yahoo_json(path: Path) -> pd.DataFrame:
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
        if close is None or len(timestamps) != len(close):
            raise ValueError("Yahoo timestamp/price payload is malformed")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True)
                .tz_convert(None)
                .normalize(),
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
        raise KeyError(f"Could not identify date/price columns: {list(df.columns)}")

    dates = df[date_col]
    if pd.api.types.is_numeric_dtype(dates):
        numeric = pd.to_numeric(dates, errors="coerce")
        unit = "ms" if numeric.dropna().median() > 1e11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce").dt.tz_convert(None)
    else:
        parsed = pd.to_datetime(dates, errors="coerce", utc=True).dt.tz_convert(None)

    return pd.DataFrame(
        {
            "date": parsed.dt.normalize(),
            "Adj Close": pd.to_numeric(df[price_col], errors="coerce"),
        }
    )


def rms_rv(log_return: pd.Series, window: int) -> pd.Series:
    """Project realized volatility convention: annualized root mean square."""
    return np.sqrt(log_return.pow(2).rolling(window).mean() * 252.0)


def verify_modern_rv_parity() -> None:
    if not MODERN_DATA.exists():
        raise FileNotFoundError(
            f"Missing modern parity source: {MODERN_DATA}. "
            "The canonical long run refuses to proceed without verifying RV construction."
        )

    modern = pd.read_csv(MODERN_DATA, usecols=["spy_log_return", "rv_5d", "rv_20d"])
    r = pd.to_numeric(modern["spy_log_return"], errors="coerce")
    errors: dict[int, float] = {}
    for window in (5, 20):
        expected = pd.to_numeric(modern[f"rv_{window}d"], errors="coerce")
        rebuilt = rms_rv(r, window)
        mask = expected.notna() & rebuilt.notna()
        max_abs = float((rebuilt[mask] - expected[mask]).abs().max())
        errors[window] = max_abs
        if not np.isfinite(max_abs) or max_abs > RV_PARITY_ATOL:
            raise AssertionError(
                f"RV parity failed for window={window}: max_abs={max_abs:.3e} > {RV_PARITY_ATOL:.1e}"
            )
    print(f"Modern RV parity passed: max abs errors {errors}")


def engineer_daily(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.dropna(subset=["date", "Adj Close"])
        .sort_values("date")
        .drop_duplicates("date")
        .query("date >= '1950-01-01'")
        .reset_index(drop=True)
    )

    price = out["Adj Close"].astype(float)
    r = np.log(price).diff()
    out["market_price"] = price
    out["market_log_return"] = r
    out["spy_adj_close"] = price
    out["spy_log_return"] = r
    out["rv_5d"] = rms_rv(r, 5)
    out["rv_10d"] = rms_rv(r, 10)
    out["rv_20d"] = rms_rv(r, 20)
    out["rv_60d"] = rms_rv(r, 60)
    out["future_rv_20d"] = out["rv_20d"].shift(-20)
    return out


def assert_frozen_reconstruction(daily: pd.DataFrame, episodes: pd.DataFrame) -> None:
    observed = {
        "rows": len(daily),
        "start": daily["date"].min(),
        "end": daily["date"].max(),
        "episodes": len(episodes),
    }
    expected = {
        "rows": EXPECTED_ROWS,
        "start": EXPECTED_START,
        "end": EXPECTED_END,
        "episodes": EXPECTED_COMPLETE_EPISODES,
    }
    if observed != expected:
        raise AssertionError(
            "Frozen long-history reconstruction drifted.\n"
            f"Observed: {observed}\nExpected: {expected}\n"
            "Stop and investigate before fitting models."
        )


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_JSON.exists():
        raise FileNotFoundError(RAW_JSON)

    verify_modern_rv_parity()
    raw = parse_yahoo_json(RAW_JSON)
    daily = engineer_daily(raw)
    _, episodes = extract_labeled_branch_episodes(
        daily,
        STATE_CONFIG,
        OUTCOME_CONFIG,
    )
    episodes = episodes[episodes["outcome_complete"]].copy().reset_index(drop=True)

    assert_frozen_reconstruction(daily, episodes)

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_CSV, index=False)
    episodes.to_csv(EPISODE_CSV, index=False)
    manifest = {
        "status": "canonical long-history reconstruction",
        "raw_source": str(RAW_JSON),
        "rv_definition": "sqrt(rolling_mean(log_return^2) * 252)",
        "modern_rv_parity_atol": RV_PARITY_ATOL,
        "daily_rows": len(daily),
        "daily_start": str(daily["date"].min().date()),
        "daily_end": str(daily["date"].max().date()),
        "episode_count": len(episodes),
        "outcome_counts": episodes["outcome"].value_counts().to_dict(),
        "state_config": STATE_CONFIG.__dict__,
        "outcome_config": OUTCOME_CONFIG.__dict__,
    }
    (RECON_DIR / "reconstruction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Canonical daily rows: {len(daily)}")
    print(f"Date range: {daily['date'].min().date()}..{daily['date'].max().date()}")
    print(f"Canonical complete episodes: {len(episodes)}")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Saved reconstruction: {RECON_DIR}")
    return daily, episodes


def main() -> None:
    reconstruct()

    sys.argv = [
        str(ENGINE),
        "--data",
        str(DAILY_CSV),
        "--episodes",
        str(EPISODE_CSV),
        "--output",
        str(OUTPUT_DIR),
    ]
    engine.main()


if __name__ == "__main__":
    main()
