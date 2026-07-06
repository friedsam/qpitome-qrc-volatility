#!/usr/bin/env python3
"""Audit whether market temporal dynamics change over historical time.

The goal is to test for dynamic 'batch effects': periods with different memory,
relaxation, and cross-series response structure that may confound model rankings.

For rolling multi-year windows, compute:
- ACF decay and integrated autocorrelation time for volatility-related series
- AR(1)-implied half-life as a secondary memory summary
- shock relaxation time after large return shocks
- lag of strongest return-shock/VIX and VIX/RV cross-correlation
- simple market-state summaries

No model ranking is performed here. This is a diagnostic of the data-generating
context that later model-performance windows can be aligned against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = [
    "spy_abs_log_return",
    "spy_squared_log_return",
    "rv_20d",
    "vix_close",
    "vix_log_change",
    "vix_rv_spread",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase3_spy_vix_volatility_extended.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/diagnostics/temporal_context"))
    p.add_argument("--window", type=int, default=756, help="Rolling window length in trading days (~3y)")
    p.add_argument("--step", type=int, default=63, help="Step in trading days (~3 months)")
    p.add_argument("--max-lag", type=int, default=120)
    p.add_argument("--shock-quantile", type=float, default=0.95)
    p.add_argument("--shock-max-horizon", type=int, default=60)
    return p.parse_args()


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 10 or np.std(x) <= 1e-15:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = np.empty(max_lag + 1, float)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return out


def first_below(a: np.ndarray, threshold: float) -> float:
    idx = np.where(np.isfinite(a[1:]) & (a[1:] < threshold))[0]
    return float(idx[0] + 1) if len(idx) else np.nan


def integrated_time(a: np.ndarray) -> float:
    if len(a) < 2 or not np.isfinite(a[0]):
        return np.nan
    vals = []
    for v in a[1:]:
        if not np.isfinite(v) or v <= 0:
            break
        vals.append(float(v))
    return float(1.0 + 2.0 * np.sum(vals))


def ar1_half_life(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x = x[ok]
    if len(x) < 20:
        return np.nan
    x0, x1 = x[:-1], x[1:]
    if np.std(x0) <= 1e-15:
        return np.nan
    phi = float(np.corrcoef(x0, x1)[0, 1])
    if not (0 < phi < 1):
        return np.nan
    return float(np.log(0.5) / np.log(phi))


def best_lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    best_lag, best_corr = np.nan, np.nan
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xx, yy = x[-lag:], y[:lag]
        elif lag > 0:
            xx, yy = x[:-lag], y[lag:]
        else:
            xx, yy = x, y
        ok = np.isfinite(xx) & np.isfinite(yy)
        if ok.sum() < 30 or np.std(xx[ok]) <= 1e-15 or np.std(yy[ok]) <= 1e-15:
            continue
        c = float(np.corrcoef(xx[ok], yy[ok])[0, 1])
        if not np.isfinite(best_corr) or abs(c) > abs(best_corr):
            best_lag, best_corr = float(lag), c
    return best_lag, best_corr


def shock_relaxation_days(frame: pd.DataFrame, q: float, max_h: int) -> tuple[float, int]:
    shock = frame["spy_abs_log_return"].to_numpy(float)
    rv = frame["rv_20d"].to_numpy(float)
    threshold = float(np.nanquantile(shock, q))
    baseline = float(np.nanmedian(rv))
    event_idx = np.where(shock >= threshold)[0]
    relax = []
    last = -10_000
    for i in event_idx:
        if i - last < 20:
            continue
        last = i
        end = min(len(frame), i + max_h + 1)
        future = rv[i + 1:end]
        hit = np.where(np.isfinite(future) & (future <= baseline))[0]
        if len(hit):
            relax.append(float(hit[0] + 1))
    return (float(np.median(relax)) if relax else np.nan, len(relax))


def main():
    a = parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data).sort_values("date").reset_index(drop=True)

    # The canonical processed dataset stores the raw components but not this
    # QRC-derived feature. Reconstruct it exactly as in the Rydberg pipeline.
    if "vix_rv_spread" not in df.columns:
        base_required = {"vix_close", "rv_20d"}
        missing_base = sorted(base_required - set(df.columns))
        if missing_base:
            raise ValueError(f"Cannot derive vix_rv_spread; missing columns: {missing_base}")
        df["vix_rv_spread"] = np.log(df["vix_close"] / 100.0) - np.log(df["rv_20d"])

    required = {"date", *SERIES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for end in range(a.window, len(df) + 1, a.step):
        start = end - a.window
        w = df.iloc[start:end].copy()
        row = {
            "window_start": str(w["date"].iloc[0].date()),
            "window_end": str(w["date"].iloc[-1].date()),
            "mid_date": str(w["date"].iloc[len(w)//2].date()),
            "n": int(len(w)),
            "median_rv20": float(w["rv_20d"].median()),
            "median_vix": float(w["vix_close"].median()),
            "mean_abs_return": float(w["spy_abs_log_return"].mean()),
            "q95_abs_return": float(w["spy_abs_log_return"].quantile(0.95)),
        }
        for col in SERIES:
            arr = w[col].to_numpy(float)
            aa = acf(arr, a.max_lag)
            row[f"{col}__acf1"] = float(aa[1]) if np.isfinite(aa[1]) else np.nan
            row[f"{col}__tau_int"] = integrated_time(aa)
            row[f"{col}__lag_below_1e"] = first_below(aa, np.exp(-1))
            row[f"{col}__lag_below_0p1"] = first_below(aa, 0.1)
            row[f"{col}__ar1_half_life"] = ar1_half_life(arr)

        lag1, corr1 = best_lag_corr(
            w["spy_abs_log_return"].to_numpy(float),
            w["vix_log_change"].to_numpy(float),
            max_lag=20,
        )
        lag2, corr2 = best_lag_corr(
            w["vix_log_change"].to_numpy(float),
            w["rv_20d"].to_numpy(float),
            max_lag=20,
        )
        row["absret_to_vixchange_best_lag"] = lag1
        row["absret_to_vixchange_best_corr"] = corr1
        row["vixchange_to_rv20_best_lag"] = lag2
        row["vixchange_to_rv20_best_corr"] = corr2

        relax, n_relax = shock_relaxation_days(w, a.shock_quantile, a.shock_max_horizon)
        row["shock_relaxation_days_median"] = relax
        row["shock_relaxation_n_events"] = int(n_relax)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.outdir / "temporal_context_windows.csv", index=False)

    trend_rows = []
    x = pd.to_datetime(out["mid_date"]).map(pd.Timestamp.toordinal).to_numpy(float)
    for col in out.columns:
        if col in {"window_start", "window_end", "mid_date"}:
            continue
        y = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or np.std(y[ok]) <= 1e-15:
            continue
        rho = float(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])
        slope = float(np.polyfit(x[ok], y[ok], 1)[0] * 365.25)
        trend_rows.append({"metric": col, "spearman_time": rho, "linear_change_per_year": slope, "n_windows": int(ok.sum())})
    trends = pd.DataFrame(trend_rows).sort_values("spearman_time", key=lambda s: s.abs(), ascending=False)
    trends.to_csv(a.outdir / "temporal_context_trends.csv", index=False)

    manifest = {
        "data": str(a.data),
        "window": a.window,
        "step": a.step,
        "max_lag": a.max_lag,
        "shock_quantile": a.shock_quantile,
        "shock_max_horizon": a.shock_max_horizon,
        "series": SERIES,
        "derived_features": {
            "vix_rv_spread": "log(vix_close / 100) - log(rv_20d)"
        },
        "interpretation_rule": "Treat strong monotonic drift or sustained era shifts in memory/relaxation metrics as evidence that fixed calendar horizons mix different temporal contexts.",
    }
    (a.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== Temporal context audit ===")
    print(f"Windows: {len(out)}  window={a.window} days  step={a.step} days")
    print(f"Date coverage: {out['window_start'].iloc[0]} -> {out['window_end'].iloc[-1]}")
    print("\n=== Strongest time trends by absolute Spearman correlation ===")
    print(trends.head(20).to_string(index=False))
    print("\n=== Latest 8 windows: selected context metrics ===")
    selected = [
        "window_start", "window_end", "median_rv20", "median_vix",
        "spy_abs_log_return__tau_int", "rv_20d__tau_int", "vix_close__tau_int",
        "shock_relaxation_days_median", "absret_to_vixchange_best_lag",
        "vixchange_to_rv20_best_lag",
    ]
    print(out[selected].tail(8).to_string(index=False))
    print(f"\nWrote outputs to {a.outdir}")


if __name__ == "__main__":
    main()
