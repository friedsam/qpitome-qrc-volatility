#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/monthly_market_features")
DATA_START = pd.Timestamp("1950-01-01")
PAPER_START = pd.Timestamp("1950-02-28")
PAPER_END = pd.Timestamp("2017-12-31")
PAPER_ROWS = 815


def read_fred(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col, value_col = frame.columns[:2]
    out = frame[[date_col, value_col]].rename(
        columns={date_col: "date", value_col: value_name}
    )
    out["date"] = (
        pd.to_datetime(out["date"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    if out["date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    return out


def read_french_zip(path: Path, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one member in {path}, found {members}")
        text = archive.read(members[0]).decode("utf-8", errors="replace")

    rows = [
        [part.strip() for part in line.split(",")]
        for line in text.splitlines()
        if re.match(r"^\s*\d{6}\s*,", line)
    ]
    if not rows:
        raise ValueError(f"No monthly rows found in {path}")
    width = max(len(row) for row in rows)
    rows = [row for row in rows if len(row) == width]
    raw = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[0], format="%Y%m", errors="raise")
            .dt.to_period("M")
            .dt.to_timestamp("M")
        }
    )
    for index, name in enumerate(names, start=1):
        out[name] = pd.to_numeric(raw[index], errors="coerce") / 100.0
    if out["date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    return out


def monthly_realized_volatility(
    daily: pd.DataFrame, price_col: str, label: str
) -> pd.DataFrame:
    x = daily[["date", price_col]].dropna().copy()
    x["log_return"] = np.log(x[price_col]).diff()
    x["month"] = x["date"].dt.to_period("M")
    grouped = x.groupby("month", sort=True)["log_return"]
    rv = grouped.apply(
        lambda series: float(np.sqrt(np.square(series.dropna()).sum()))
    )
    out = pd.DataFrame(
        {
            "date": rv.index.to_timestamp("M"),
            f"n_daily_returns_{label}": grouped.count().to_numpy(),
            f"rv_{label}": rv.to_numpy(),
        }
    )
    out[f"log_rv_{label}"] = np.log(
        out[f"rv_{label}"].where(out[f"rv_{label}"] > 0)
    )
    out[f"rv_q3_mean_{label}"] = out[f"rv_{label}"].rolling(
        3, min_periods=3
    ).mean()
    out[f"rv_a12_mean_{label}"] = out[f"rv_{label}"].rolling(
        12, min_periods=12
    ).mean()
    out[f"log_of_rv_q3_mean_{label}"] = np.log(out[f"rv_q3_mean_{label}"])
    out[f"log_of_rv_a12_mean_{label}"] = np.log(out[f"rv_a12_mean_{label}"])
    out[f"log_rv_q3_mean_{label}"] = out[f"log_rv_{label}"].rolling(
        3, min_periods=3
    ).mean()
    out[f"log_rv_a12_mean_{label}"] = out[f"log_rv_{label}"].rolling(
        12, min_periods=12
    ).mean()
    return out


def merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on="date", how="left", validate="one_to_one")


def robust_z(series: pd.Series, window: int = 12) -> pd.Series:
    prior = series.shift(1)
    median = prior.rolling(window, min_periods=window).median()
    mad = prior.rolling(window, min_periods=window).apply(
        lambda values: float(np.median(np.abs(values - np.median(values)))),
        raw=True,
    )
    return (series - median) / (1.4826 * mad).where(mad > 0)


def add_feature(
    catalog: list[dict],
    frame: pd.DataFrame,
    name: str,
    values: pd.Series,
    family: str,
    role: str,
    formula: str,
    timing: str = "available by end of month t",
) -> None:
    frame[name] = values
    catalog.append(
        {
            "feature": name,
            "family": family,
            "role": role,
            "formula": formula,
            "timing": timing,
        }
    )


def main() -> None:
    daily_path = RAW / "monthly_market_features/sp500_yahoo_raw.csv"
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily = daily.loc[daily["date"] >= DATA_START].sort_values("date")
    if daily["date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {daily_path}")

    monthly = monthly_realized_volatility(daily, "close", "close")
    adjusted_col = "adjusted_close" if "adjusted_close" in daily.columns else None
    if adjusted_col:
        monthly = merge(
            monthly,
            monthly_realized_volatility(daily, adjusted_col, "adjclose"),
        )

    fred = RAW / "macro_fred_monthly"
    for filename, name in {
        "fred_TB3MS_three_month_tbill.csv": "tb3ms",
        "fred_CPIAUCSL_cpi.csv": "cpi_index",
        "fred_INDPRO_industrial_production.csv": "ip_index",
        "fred_AAA_aaa_corporate_yield.csv": "aaa_yield",
        "fred_BAA_baa_corporate_yield.csv": "baa_yield",
    }.items():
        monthly = merge(monthly, read_fred(fred / filename, name))

    factors = RAW / "equity_factor_returns_monthly"
    monthly = merge(
        monthly,
        read_french_zip(
            factors / "ff3_monthly.zip",
            ["mkt_excess", "smb", "hml", "rf"],
        ),
    )
    monthly = merge(
        monthly,
        read_french_zip(
            factors / "short_term_reversal_monthly.zip", ["str"]
        ),
    )

    monthly["inflation_log_change"] = 100.0 * np.log(
        monthly["cpi_index"]
    ).diff()
    monthly["ip_log_growth"] = 100.0 * np.log(monthly["ip_index"]).diff()
    monthly["default_spread_baa_minus_aaa__candidate"] = (
        monthly["baa_yield"] - monthly["aaa_yield"]
    )
    monthly["inflation_log_change__avail_lag1"] = monthly[
        "inflation_log_change"
    ].shift(1)
    monthly["ip_log_growth__avail_lag1"] = monthly["ip_log_growth"].shift(1)

    for label in ["close", "adjclose"]:
        source = f"log_rv_{label}"
        if source in monthly:
            monthly[f"target_{source}_t_plus_1"] = monthly[source].shift(-1)
            monthly[f"target_{source}_t_plus_5"] = monthly[source].shift(-5)

    current_month = (
        pd.Timestamp(datetime.now(timezone.utc).date())
        .to_period("M")
        .to_timestamp()
    )
    extended = monthly.loc[monthly["date"] < current_month].copy()
    paper = monthly.loc[
        (monthly["date"] >= PAPER_START) & (monthly["date"] <= PAPER_END)
    ].copy()
    if len(paper) != PAPER_ROWS:
        raise ValueError(
            f"Paper slice has {len(paper)} rows; expected {PAPER_ROWS}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    extended_path = OUT / "extended_complete_market_months.parquet"
    paper_path = OUT / "paper_parity_1950_02_to_2017_12.parquet"
    extended.to_parquet(extended_path, index=False)
    paper.to_parquet(paper_path, index=False)

    y = paper["log_rv_close"]
    features = pd.DataFrame({"date": paper["date"]})
    catalog: list[dict] = []
    add_feature(
        catalog,
        features,
        "vol_state_1m",
        y,
        "volatility_state",
        "current state",
        "log_rv_close",
    )
    add_feature(
        catalog,
        features,
        "vol_state_3m_mean",
        y.rolling(3, min_periods=3).mean(),
        "volatility_state",
        "short memory",
        "mean(log_rv[t-2:t])",
    )
    add_feature(
        catalog,
        features,
        "vol_state_12m_mean",
        y.rolling(12, min_periods=12).mean(),
        "volatility_state",
        "long memory",
        "mean(log_rv[t-11:t])",
    )
    delta = y.diff()
    add_feature(
        catalog,
        features,
        "vol_change_1m",
        delta,
        "volatility_dynamics",
        "rate of change",
        "log_rv[t] - log_rv[t-1]",
    )
    add_feature(
        catalog,
        features,
        "vol_change_3m",
        y - y.shift(3),
        "volatility_dynamics",
        "medium change",
        "log_rv[t] - log_rv[t-3]",
    )
    add_feature(
        catalog,
        features,
        "vol_acceleration_1m",
        delta.diff(),
        "volatility_dynamics",
        "acceleration",
        "delta1[t] - delta1[t-1]",
    )
    add_feature(
        catalog,
        features,
        "vol_short_minus_long",
        features["vol_state_3m_mean"] - features["vol_state_12m_mean"],
        "volatility_dynamics",
        "timescale separation",
        "mean3(log_rv) - mean12(log_rv)",
    )
    add_feature(
        catalog,
        features,
        "vol_prior_12m_robust_z",
        robust_z(y),
        "volatility_dynamics",
        "robust surprise",
        "current log_rv relative to prior-12-month median/MAD",
    )

    for column in ["mkt_excess", "smb", "hml", "str"]:
        add_feature(
            catalog,
            features,
            f"market_{column}",
            paper[column],
            "market_drivers",
            "market shock",
            column,
        )
    for column in [
        "tb3ms",
        "aaa_yield",
        "baa_yield",
        "default_spread_baa_minus_aaa__candidate",
    ]:
        base = column.replace("__candidate", "")
        add_feature(
            catalog,
            features,
            f"credit_{base}_level",
            paper[column],
            "credit_rates",
            "financial conditions state",
            column,
        )
        add_feature(
            catalog,
            features,
            f"credit_{base}_change_1m",
            paper[column].diff(),
            "credit_rates",
            "financial conditions change",
            f"{column}[t] - {column}[t-1]",
        )
    for source, name in {
        "inflation_log_change__avail_lag1": "macro_inflation_growth_lag1",
        "ip_log_growth__avail_lag1": "macro_ip_growth_lag1",
    }.items():
        add_feature(
            catalog,
            features,
            name,
            paper[source],
            "macro_dynamics",
            "lagged macro growth",
            source,
            "one-month availability lag enforced",
        )
        add_feature(
            catalog,
            features,
            f"{name}_change_1m",
            paper[source].diff(),
            "macro_dynamics",
            "macro acceleration",
            f"{source}[t] - {source}[t-1]",
            "one-month availability lag enforced",
        )

    features["target_log_rv_t_plus_1"] = y.shift(-1)
    features["target_log_rv_t_plus_5"] = y.shift(-5)
    feature_dir = OUT / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(
        feature_dir / "preliminary_features.parquet", index=False
    )
    pd.DataFrame(catalog).to_csv(
        feature_dir / "feature_catalog.csv", index=False
    )

    manifest = {
        "producer": "scripts/data/build_monthly_market_features.py",
        "inputs": [str(daily_path), str(fred), str(factors)],
        "outputs": {
            "extended": str(extended_path),
            "paper": str(paper_path),
            "features": str(feature_dir / "preliminary_features.parquet"),
            "catalog": str(feature_dir / "feature_catalog.csv"),
        },
        "paper_rows": len(paper),
        "extended_rows": len(extended),
        "feature_count": len(catalog),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
