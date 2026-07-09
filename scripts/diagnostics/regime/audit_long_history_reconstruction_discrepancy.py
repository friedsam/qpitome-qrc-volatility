"""Forensic audit of the 74-vs-~85 long-history branch discrepancy.

Checks whether the mismatch comes from:
1. realized-volatility convention;
2. branch-threshold choice within the frozen 24-definition audit grid.

No model fitting occurs.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    OutcomeConfig,
    extract_labeled_branch_episodes,
)


MODERN = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
RAW_JSON = Path("data/raw/paper_monthly/20260705T170000Z/gspc_daily_yahoo.json")
OUTPUT = Path("results/regimes/long_history_reconstruction_discrepancy_v1")

CANONICAL = (
    Path(__file__).parents[2]
    / "baselines"
    / "esn"
    / "run_long_history_har_residual_replication_canonical.py"
)
spec = importlib.util.spec_from_file_location("long_history_canonical", CANONICAL)
if spec is None or spec.loader is None:
    raise ImportError(CANONICAL)
canonical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical)

STRESS_QUANTILES = [0.65, 0.70, 0.75]
DRAWDOWN_THRESHOLDS = [-0.05, -0.06]
STABILIZATION_FLOORS = [-0.015, -0.005]
PRIOR_DECLINE_THRESHOLDS = [-0.03, -0.05]
OUTCOME = OutcomeConfig(horizon=40, recovery_scale=0.55, relapse_scale=0.70)


def rv_std(r: pd.Series, window: int, ddof: int) -> pd.Series:
    return r.rolling(window).std(ddof=ddof) * np.sqrt(252.0)


def rv_rms(r: pd.Series, window: int) -> pd.Series:
    return np.sqrt(r.pow(2).rolling(window).mean() * 252.0)


def rv_centered_rms(r: pd.Series, window: int) -> pd.Series:
    # Equivalent to population std, included explicitly for audit readability.
    mu = r.rolling(window).mean()
    return np.sqrt((r.sub(mu).pow(2)).rolling(window).mean() * 252.0)


def compare_modern_conventions() -> pd.DataFrame:
    if not MODERN.exists():
        raise FileNotFoundError(MODERN)
    df = pd.read_csv(MODERN, parse_dates=["date"])
    r = pd.to_numeric(df["spy_log_return"], errors="coerce")
    rows = []
    for name, fn in {
        "std_ddof1": lambda w: rv_std(r, w, 1),
        "std_ddof0": lambda w: rv_std(r, w, 0),
        "rms": lambda w: rv_rms(r, w),
    }.items():
        for window in (5, 20):
            target = pd.to_numeric(df[f"rv_{window}d"], errors="coerce")
            pred = fn(window)
            mask = target.notna() & pred.notna()
            diff = pred[mask] - target[mask]
            rows.append(
                {
                    "convention": name,
                    "window": window,
                    "n": int(mask.sum()),
                    "mae": float(diff.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "max_abs": float(diff.abs().max()),
                    "corr": float(pred[mask].corr(target[mask])),
                    "mean_ratio": float((pred[mask] / target[mask]).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "mae"])


def engineer(raw: pd.DataFrame, convention: str) -> pd.DataFrame:
    out = (
        raw.dropna(subset=["date", "Adj Close"])
        .sort_values("date")
        .drop_duplicates("date")
        .query("date >= '1950-01-01'")
        .reset_index(drop=True)
    )
    p = out["Adj Close"].astype(float)
    r = np.log(p).diff()
    out["spy_adj_close"] = p
    out["spy_log_return"] = r

    if convention == "std_ddof1":
        fn = lambda w: rv_std(r, w, 1)
    elif convention == "std_ddof0":
        fn = lambda w: rv_std(r, w, 0)
    elif convention == "rms":
        fn = lambda w: rv_rms(r, w)
    else:
        raise ValueError(convention)

    out["rv_5d"] = fn(5)
    out["rv_20d"] = fn(20)
    return out


def config_id(q: float, dd: float, stab: float, decline: float) -> str:
    return f"q{int(q*100)}_dd{int(abs(dd)*100):02d}_stab{stab:+.3f}_decl{decline:+.2f}"


def audit_counts(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for convention in ("std_ddof1", "std_ddof0", "rms"):
        daily = engineer(raw, convention)
        for q, dd, stab, decline in itertools.product(
            STRESS_QUANTILES,
            DRAWDOWN_THRESHOLDS,
            STABILIZATION_FLOORS,
            PRIOR_DECLINE_THRESHOLDS,
        ):
            cfg = BranchStateConfig(
                expanding_min_periods=504,
                stress_quantile=q,
                drawdown_threshold=dd,
                rv_ratio_cap=1.0,
                stabilization_return_floor=stab,
                prior_decline_threshold=decline,
                prior_decline_lookback=20,
                merge_gap_days=3,
                min_episode_separation=20,
            )
            _, episodes = extract_labeled_branch_episodes(daily, cfg, OUTCOME)
            complete = episodes[episodes["outcome_complete"]].copy()
            counts = complete["outcome"].value_counts()
            rows.append(
                {
                    "rv_convention": convention,
                    "config_id": config_id(q, dd, stab, decline),
                    "stress_quantile": q,
                    "drawdown_threshold": dd,
                    "stabilization_floor": stab,
                    "prior_decline_threshold": decline,
                    "episodes_total": len(episodes),
                    "episodes_complete": len(complete),
                    "recovery": int(counts.get("recovery", 0)),
                    "relapse": int(counts.get("relapse", 0)),
                    "mixed": int(counts.get("mixed", 0)),
                    "distance_from_85": abs(len(complete) - 85),
                    "first_branch": str(pd.to_datetime(complete["branch_date"]).min().date()) if len(complete) else None,
                    "last_branch": str(pd.to_datetime(complete["branch_date"]).max().date()) if len(complete) else None,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["distance_from_85", "rv_convention", "episodes_complete", "config_id"]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    modern = compare_modern_conventions()
    modern.to_csv(OUTPUT / "modern_rv_convention_match.csv", index=False)
    print("\nModern RV convention match (lowest MAE first):")
    print(modern.to_string(index=False))

    raw = canonical.parse_yahoo_json(RAW_JSON)
    grid = audit_counts(raw)
    grid.to_csv(OUTPUT / "long_history_count_grid.csv", index=False)

    print("\nClosest long-history reconstructions to 85 complete episodes:")
    print(grid.head(20).to_string(index=False))

    mid = grid[
        (grid["stress_quantile"] == 0.70)
        & (grid["drawdown_threshold"] == -0.06)
        & (grid["stabilization_floor"] == -0.005)
        & (grid["prior_decline_threshold"] == -0.05)
    ]
    print("\nCurrent mid-grid under each RV convention:")
    print(mid.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
