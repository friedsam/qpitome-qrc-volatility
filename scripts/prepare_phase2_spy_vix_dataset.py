from pathlib import Path

import numpy as np
import pandas as pd


RAW_SPY_PATH = Path("data/raw/kaggle_market_ohlcv/SPY.csv")
RAW_VIX_PATH = Path("data/raw/vix/VIX_History.csv")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "phase2_spy_vix_volatility.csv"


def load_spy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing SPY file: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SPY file missing columns: {sorted(missing)}")

    df = df.rename(
        columns={
            "open": "spy_open",
            "high": "spy_high",
            "low": "spy_low",
            "close": "spy_close",
            "volume": "spy_volume",
            "adjusted_close": "spy_adj_close",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    cols = [
        "date",
        "spy_open",
        "spy_high",
        "spy_low",
        "spy_close",
        "spy_adj_close",
        "spy_volume",
    ]
    return df[cols]


def load_vix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing VIX file: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"VIX file missing columns: {sorted(missing)}")

    df = df.rename(
        columns={
            "open": "vix_open",
            "high": "vix_high",
            "low": "vix_low",
            "close": "vix_close",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]


def trailing_rv(log_return: pd.Series, window: int) -> pd.Series:
    """Annualized realized-volatility proxy from trailing squared daily log returns."""
    return np.sqrt(252.0 / window * log_return.pow(2).rolling(window).sum())


def forward_rv(log_return: pd.Series, horizon: int) -> pd.Series:
    """Annualized realized-volatility target from future squared daily log returns."""
    return np.sqrt(
        252.0
        / horizon
        * log_return.pow(2).shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Return features. Use adjusted close for split/dividend-adjusted close-to-close returns.
    out["spy_log_return"] = np.log(out["spy_adj_close"]).diff()
    out["spy_abs_log_return"] = out["spy_log_return"].abs()
    out["spy_squared_log_return"] = out["spy_log_return"].pow(2)

    # Daily OHLC volatility proxies. These are same-day features available after market close.
    out["spy_log_hl_range"] = np.log(out["spy_high"] / out["spy_low"])
    out["spy_log_oc_return"] = np.log(out["spy_close"] / out["spy_open"])

    out["parkinson_var"] = out["spy_log_hl_range"].pow(2) / (4.0 * np.log(2.0))
    out["garman_klass_var"] = (
        0.5 * out["spy_log_hl_range"].pow(2)
        - (2.0 * np.log(2.0) - 1.0) * out["spy_log_oc_return"].pow(2)
    )

    # Liquidity/activity features.
    out["spy_log_volume"] = np.log(out["spy_volume"].replace(0, np.nan))
    out["spy_log_volume_change"] = out["spy_log_volume"].diff()
    out["spy_dollar_volume"] = out["spy_adj_close"] * out["spy_volume"]
    out["spy_log_dollar_volume"] = np.log(out["spy_dollar_volume"].replace(0, np.nan))

    # Trailing realized-volatility proxies.
    for window in [5, 10, 20, 60]:
        out[f"rv_{window}d"] = trailing_rv(out["spy_log_return"], window)

    # Forward realized-volatility targets. These are targets, not features.
    for horizon in [5, 20]:
        out[f"future_rv_{horizon}d"] = forward_rv(out["spy_log_return"], horizon)

    # Multi-scale volatility state features.
    out["rv_ratio_5_20"] = out["rv_5d"] / out["rv_20d"]
    out["rv_ratio_20_60"] = out["rv_20d"] / out["rv_60d"]
    out["rv_slope_5_20"] = out["rv_5d"] - out["rv_20d"]
    out["rv_slope_20_60"] = out["rv_20d"] - out["rv_60d"]

    # Drawdown over trailing 20 trading days.
    rolling_max_20d = out["spy_adj_close"].rolling(window=20).max()
    out["spy_drawdown_20d"] = out["spy_adj_close"] / rolling_max_20d - 1.0

    # VIX features.
    out["vix_log_close"] = np.log(out["vix_close"])
    out["vix_log_change"] = out["vix_log_close"].diff()
    out["vix_abs_log_change"] = out["vix_log_change"].abs()
    out["vix_log_hl_range"] = np.log(out["vix_high"] / out["vix_low"])

    for window in [5, 20]:
        out[f"vix_ma_{window}d"] = out["vix_close"].rolling(window).mean()
        out[f"vix_std_{window}d"] = out["vix_close"].rolling(window).std()

    # Do not create regime labels here. Regime thresholds must be fit from train split only.
    return out


def build_dataset() -> pd.DataFrame:
    spy = load_spy(RAW_SPY_PATH)
    vix = load_vix(RAW_VIX_PATH)

    df = spy.merge(vix, on="date", how="inner")
    df = add_features(df)

    core_cols = [
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
    df = df.dropna(subset=core_cols).reset_index(drop=True)

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nTarget summary:")
    print(df[["future_rv_5d", "future_rv_20d"]].describe())
    print("\nHead:")
    print(df.head(3))
    print("\nTail:")
    print(df.tail(3))


if __name__ == "__main__":
    main()
