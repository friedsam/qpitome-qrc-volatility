#!/usr/bin/env python3
"""
First market-driven Rydberg scalar probe.

This script does not run a new AHS simulation for every market day. It reuses a
previous Rydberg response grid, interpolates numeric reservoir features over the
physical detuning coordinate, and asks whether a simple market stress scalar can
use that nonlinear feature map for volatility-regime warnings.

Purpose
-------
Move one step beyond synthetic x scans:

    market scalar x_t -> detuning window -> interpolated Rydberg features Z_t
    -> simple chronological readouts for future realized-volatility regimes.

This is a bridge diagnostic, not the final temporal QRC model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    data_path: Path
    response_grid_path: Path
    target_col: str
    scalar_col: str
    train_frac: float
    val_frac: float
    delta_min: float
    delta_max: float
    clip_quantile_low: float
    clip_quantile_high: float
    ridge_alpha: float
    out_prefix: Path


def chronological_split(n: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    idx = np.arange(n)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return train, val, test


def robust_scale_train_only(values: np.ndarray, train_idx: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, dict[str, float]]:
    train_values = values[train_idx]
    lo = float(np.nanquantile(train_values, q_low))
    hi = float(np.nanquantile(train_values, q_high))
    med = float(np.nanmedian(train_values))
    half_range = 0.5 * (hi - lo)
    if not np.isfinite(half_range) or half_range <= 0:
        raise ValueError("Invalid robust scale; check scalar distribution")
    scaled = (values - med) / half_range
    clipped = np.clip(scaled, -1.0, 1.0)
    return clipped, {"median": med, "q_low": lo, "q_high": hi, "half_range": half_range}


def add_market_scalar(df: pd.DataFrame, scalar_col: str) -> pd.DataFrame:
    out = df.copy()
    if scalar_col == "rv_accel_log_5_20":
        out[scalar_col] = np.log(out["rv_5d"].clip(lower=1e-12) / out["rv_20d"].clip(lower=1e-12))
    elif scalar_col == "vix_rv_spread":
        # crude implied-vs-realized proxy; VIX is annualized percent, so divide by 100.
        out[scalar_col] = np.log((out["vix_close"].clip(lower=1e-12) / 100.0) / out["rv_20d"].clip(lower=1e-12))
    elif scalar_col in out.columns:
        pass
    else:
        raise ValueError(f"Unknown scalar_col '{scalar_col}'. Provide an existing column or supported derived scalar.")
    return out


def interp_response_features(response: pd.DataFrame, delta_values: np.ndarray) -> pd.DataFrame:
    if "delta_end_over_omega" not in response.columns:
        raise ValueError("response grid must contain delta_end_over_omega; rerun updated rydberg_tfim_response.py")

    response = response.sort_values("delta_end_over_omega").drop_duplicates("delta_end_over_omega")
    grid_x = response["delta_end_over_omega"].to_numpy(dtype=float)
    feature_cols = [
        c for c in response.columns
        if c.startswith("site_") and c.endswith("_excitation")
    ] + [
        "mean_excitation",
        "var_excitation",
        "entropy_bits",
        "nearest_pair_11_mean",
        "next_nearest_pair_11_mean",
        "blockade_contrast_nnn_minus_nn",
        "staggered_even_minus_odd",
        "top1_freq",
        "top2_freq",
        "top3_freq",
    ]
    feature_cols = [c for c in feature_cols if c in response.columns]
    if not feature_cols:
        raise ValueError("No numeric Rydberg response features found")

    data: dict[str, np.ndarray] = {}
    for col in feature_cols:
        grid_y = response[col].to_numpy(dtype=float)
        data[f"rydberg_{col}"] = np.interp(delta_values, grid_x, grid_y)
    return pd.DataFrame(data)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 and np.std(y_true) > 0 else float("nan")
    return {
        "rmse": rmse,
        "corr": corr,
        "actual_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
    }


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": safe_auc(y_true, score),
    }


def evaluate_model(
    name: str,
    X: np.ndarray,
    y_reg: np.ndarray,
    y_q90: np.ndarray,
    y_q95: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []

    reg = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    reg.fit(X[train_idx], y_reg[train_idx])
    reg_pred = reg.predict(X)

    # Warning classifiers. Class weights are important for q95.
    clf90 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf95 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf90.fit(X[train_idx], y_q90[train_idx])
    clf95.fit(X[train_idx], y_q95[train_idx])
    score90 = clf90.predict_proba(X)[:, 1]
    score95 = clf95.predict_proba(X)[:, 1]

    # Thresholds chosen on train predicted-score quantiles to create warning-rate discipline.
    thr90 = float(np.quantile(score90[train_idx], 0.90))
    thr95 = float(np.quantile(score95[train_idx], 0.95))

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        base = {"model": name, "split": split, "n": int(len(idx))}
        rows.append({**base, "task": "regression", **regression_metrics(y_reg[idx], reg_pred[idx])})
        rows.append({**base, "task": "q90_warning", **classification_metrics(y_q90[idx], score90[idx], thr90)})
        rows.append({**base, "task": "q95_crisis", **classification_metrics(y_q95[idx], score95[idx], thr95)})

    pred_df = pd.DataFrame({
        f"{name}_reg_pred": reg_pred,
        f"{name}_q90_score": score90,
        f"{name}_q95_score": score95,
    })
    return rows, pred_df


def plot_diagnostics(df: pd.DataFrame, out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["market_x_scaled"], df["delta_end_over_omega"], s=5, alpha=0.25)
    ax.set_xlabel("scaled market scalar")
    ax.set_ylabel("delta_end_over_omega")
    ax.set_title("Market scalar mapped into Rydberg detuning window")
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_scalar_to_delta.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["delta_end_over_omega"], df["rydberg_entropy_bits"], s=5, alpha=0.25)
    ax.set_xlabel("delta_end_over_omega")
    ax.set_ylabel("interpolated Rydberg entropy")
    ax.set_title("Rydberg response feature sampled by market inputs")
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_delta_entropy.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Market-driven Rydberg scalar feature probe")
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--response-grid", type=Path, default=Path("artifacts/rydberg_tfim_response_zoom_v2.csv"))
    p.add_argument("--target-col", default="future_rv_20d")
    p.add_argument("--scalar-col", default="rv_accel_log_5_20")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--delta-min", type=float, default=0.40)
    p.add_argument("--delta-max", type=float, default=1.10)
    p.add_argument("--clip-quantile-low", type=float, default=0.05)
    p.add_argument("--clip-quantile-high", type=float, default=0.95)
    p.add_argument("--ridge-alpha", type=float, default=1000.0)
    p.add_argument("--out-prefix", type=Path, default=Path("results/tables/phase3_rydberg_market_scalar_probe"))
    args = p.parse_args()

    cfg = Config(
        data_path=args.data,
        response_grid_path=args.response_grid,
        target_col=args.target_col,
        scalar_col=args.scalar_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        clip_quantile_low=args.clip_quantile_low,
        clip_quantile_high=args.clip_quantile_high,
        ridge_alpha=args.ridge_alpha,
        out_prefix=args.out_prefix,
    )

    market = pd.read_csv(cfg.data_path, parse_dates=["date"])
    market = add_market_scalar(market, cfg.scalar_col)
    needed = ["date", cfg.target_col, cfg.scalar_col, "rv_5d", "rv_20d", "rv_60d", "vix_close", "spy_drawdown_20d"]
    market = market[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    train_idx, val_idx, test_idx = chronological_split(len(market), cfg.train_frac, cfg.val_frac)
    market_x, scaling = robust_scale_train_only(
        market[cfg.scalar_col].to_numpy(dtype=float),
        train_idx,
        cfg.clip_quantile_low,
        cfg.clip_quantile_high,
    )
    delta = cfg.delta_min + 0.5 * (market_x + 1.0) * (cfg.delta_max - cfg.delta_min)

    response = pd.read_csv(cfg.response_grid_path)
    rydberg_features = interp_response_features(response, delta)

    model_df = pd.concat([
        market,
        pd.DataFrame({"market_x_scaled": market_x, "delta_end_over_omega": delta}),
        rydberg_features,
    ], axis=1)

    y = model_df[cfg.target_col].to_numpy(dtype=float)
    q90 = float(np.quantile(y[train_idx], 0.90))
    q95 = float(np.quantile(y[train_idx], 0.95))
    y_q90 = (y >= q90).astype(int)
    y_q95 = (y >= q95).astype(int)

    raw_feature_cols = [cfg.scalar_col, "market_x_scaled", "rv_5d", "rv_20d", "rv_60d", "vix_close", "spy_drawdown_20d"]
    rydberg_feature_cols = [c for c in model_df.columns if c.startswith("rydberg_")]

    feature_sets = {
        "raw_scalar_baseline": raw_feature_cols,
        "rydberg_interpolated_features": rydberg_feature_cols,
        "raw_plus_rydberg": raw_feature_cols + rydberg_feature_cols,
    }

    all_rows: list[dict[str, object]] = []
    pred_parts = [model_df[["date", cfg.target_col, cfg.scalar_col, "market_x_scaled", "delta_end_over_omega"]].copy()]
    for name, cols in feature_sets.items():
        X = model_df[cols].to_numpy(dtype=float)
        rows, preds = evaluate_model(name, X, y, y_q90, y_q95, train_idx, val_idx, test_idx, cfg.ridge_alpha)
        all_rows.extend(rows)
        pred_parts.append(preds)

    metrics = pd.DataFrame(all_rows)
    preds = pd.concat(pred_parts, axis=1)

    cfg.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_metrics.csv")
    preds_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_predictions.csv")
    summary_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_summary.json")
    metrics.to_csv(metrics_path, index=False)
    preds.to_csv(preds_path, index=False)

    summary = {
        "config": {**asdict(cfg), "data_path": str(cfg.data_path), "response_grid_path": str(cfg.response_grid_path), "out_prefix": str(cfg.out_prefix)},
        "n_rows": int(len(model_df)),
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "target_thresholds_train": {"q90": q90, "q95": q95},
        "market_scalar_scaling_train_only": scaling,
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "note": "This is a scalar feature-map diagnostic. It reuses a response grid and is not yet a fully temporal Rydberg reservoir.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    plot_diagnostics(model_df, cfg.out_prefix.with_suffix(""))

    print("Saved:")
    print(metrics_path)
    print(preds_path)
    print(summary_path)
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
