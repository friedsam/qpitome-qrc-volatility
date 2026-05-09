from pathlib import Path

import numpy as np
import pandas as pd


RAW_SPY_PATH = Path("data/raw/kaggle_market_ohlcv/SPY.csv")
RAW_VIX_PATH = Path("data/raw/vix/VIX_History.csv")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "market_stress_v0.csv"


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

    return df[
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


def load_vix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing VIX file: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"VIX file missing columns: {sorted(missing)}")

    df = df.rename(columns={"close": "vix_close"})
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "vix_close"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Use adjusted close for returns to account for splits/dividends.
    out["spy_log_return"] = np.log(out["spy_adj_close"]).diff()
    out["spy_return"] = out["spy_adj_close"].pct_change()
    out["spy_abs_return"] = out["spy_return"].abs()

    # Simple intraday range proxy. Uses raw OHLC, not adjusted.
    out["spy_range"] = (out["spy_high"] - out["spy_low"]) / out["spy_close"]

    # Liquidity/activity proxy.
    out["spy_dollar_volume"] = out["spy_adj_close"] * out["spy_volume"]

    # Rolling realized volatility, annualized using ~252 trading days.
    for window in [5, 10, 20]:
        out[f"rv_{window}d"] = (
            out["spy_log_return"].rolling(window=window).std() * np.sqrt(252)
        )

    # Drawdown over last 20 trading days using adjusted close.
    rolling_max_20d = out["spy_adj_close"].rolling(window=20).max()
    out["spy_drawdown_20d"] = out["spy_adj_close"] / rolling_max_20d - 1.0

    # VIX features.
    out["vix_change"] = out["vix_close"].diff()
    out["vix_pct_change"] = out["vix_close"].pct_change()
    out["vix_ma_5d"] = out["vix_close"].rolling(window=5).mean()

    # Future realized volatility target over next 5 trading days.
    # Shift return backward so each row's target uses future returns only.
    out["future_rv_5d"] = (
        out["spy_log_return"].shift(-1).rolling(window=5).std().shift(-4) * np.sqrt(252)
    )

    return out


def add_labels(df: pd.DataFrame, threshold_quantile: float = 0.80) -> pd.DataFrame:
    out = df.copy()

    # Provisional threshold over available non-missing rows.
    # Later, baseline code should recompute threshold using train split only.
    threshold = out["future_rv_5d"].dropna().quantile(threshold_quantile)

    out["future_high_vol_label"] = (out["future_rv_5d"] > threshold).astype("Int64")
    out.attrs["future_rv_5d_threshold_q80"] = threshold

    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    spy = load_spy(RAW_SPY_PATH)
    vix = load_vix(RAW_VIX_PATH)

    df = spy.merge(vix, on="date", how="inner")
    df = add_features(df)
    df = add_labels(df)

    # Drop rows where core model columns are unavailable due to rolling/future windows.
    core_cols = [
        "spy_log_return",
        "spy_return",
        "spy_range",
        "rv_5d",
        "rv_10d",
        "rv_20d",
        "vix_close",
        "vix_change",
        "vix_ma_5d",
        "future_rv_5d",
        "future_high_vol_label",
    ]
    df_model = df.dropna(subset=core_cols).reset_index(drop=True)

    df_model.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Shape: {df_model.shape}")
    print(f"Date range: {df_model['date'].min()} -> {df_model['date'].max()}")
    print()
    print("Columns:")
    print(list(df_model.columns))
    print()
    print("Label counts:")
    print(df_model["future_high_vol_label"].value_counts(dropna=False))
    print()
    print("Future RV 5d threshold q80:")
    print(df.attrs.get("future_rv_5d_threshold_q80"))
    print()
    print("Head:")
    print(df_model.head(3))
    print()
    print("Tail:")
    print(df_model.tail(3))


if __name__ == "__main__":
    main()
