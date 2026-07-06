#!/usr/bin/env python3
"""
Temporal Rydberg response probe.

This is the next step after the scalar lookup diagnostic. It still reuses the
simulated Rydberg response grid, but now applies it to a trailing market-state
sequence instead of one scalar per date.

For each date t:
    trailing market scalar window -> detuning path -> Rydberg feature trajectory
    -> trajectory summaries -> warning/regression readouts

This is not a hardware run and not a full AHS recurrent simulation yet. It is a
fast bridge diagnostic for whether the Rydberg response map becomes more useful
when used temporally.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
    lookback: int
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
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def add_market_scalar(df: pd.DataFrame, scalar_col: str) -> pd.DataFrame:
    out = df.copy()
    if scalar_col == "rv_accel_log_5_20":
        out[scalar_col] = np.log(out["rv_5d"].clip(lower=1e-12) / out["rv_20d"].clip(lower=1e-12))
    elif scalar_col == "vix_rv_spread":
        out[scalar_col] = np.log((out["vix_close"].clip(lower=1e-12) / 100.0) / out["rv_20d"].clip(lower=1e-12))
    elif scalar_col in out.columns:
        pass
    else:
        raise ValueError(f"Unknown scalar_col '{scalar_col}'")
    return out


def robust_scale(values: np.ndarray, train_idx: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, dict[str, float]]:
    train_values = values[train_idx]
    lo = float(np.nanquantile(train_values, q_low))
    hi = float(np.nanquantile(train_values, q_high))
    med = float(np.nanmedian(train_values))
    half_range = 0.5 * (hi - lo)
    if not np.isfinite(half_range) or half_range <= 0:
        raise ValueError("Invalid robust scale")
    scaled = np.clip((values - med) / half_range, -1.0, 1.0)
    return scaled, {"median": med, "q_low": lo, "q_high": hi, "half_range": half_range}


def response_feature_grid(response: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if "delta_end_over_omega" not in response.columns:
        raise ValueError("response grid must contain delta_end_over_omega")
    response = response.sort_values("delta_end_over_omega").drop_duplicates("delta_end_over_omega")
    delta_grid = response["delta_end_over_omega"].to_numpy(dtype=float)
    cols = [
        c for c in response.columns if c.startswith("site_") and c.endswith("_excitation")
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
    cols = [c for c in cols if c in response.columns]
    grid = {c: response[c].to_numpy(dtype=float) for c in cols}
    return delta_grid, cols, grid


def trajectory_features(
    delta_seq: np.ndarray,
    delta_grid: np.ndarray,
    feature_cols: list[str],
    grid: dict[str, np.ndarray],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in feature_cols:
        z = np.interp(delta_seq, delta_grid, grid[col])
        prefix = f"rydberg_traj_{col}"
        out[f"{prefix}_last"] = float(z[-1])
        out[f"{prefix}_mean"] = float(np.mean(z))
        out[f"{prefix}_std"] = float(np.std(z))
        out[f"{prefix}_min"] = float(np.min(z))
        out[f"{prefix}_max"] = float(np.max(z))
        out[f"{prefix}_slope"] = float((z[-1] - z[0]) / max(1, len(z) - 1))
    out["delta_last"] = float(delta_seq[-1])
    out["delta_mean"] = float(np.mean(delta_seq))
    out["delta_std"] = float(np.std(delta_seq))
    out["delta_min_window"] = float(np.min(delta_seq))
    out["delta_max_window"] = float(np.max(delta_seq))
    out["delta_slope"] = float((delta_seq[-1] - delta_seq[0]) / max(1, len(delta_seq) - 1))
    return out


def make_temporal_dataset(market: pd.DataFrame, market_x: np.ndarray, cfg: Config, response: pd.DataFrame) -> pd.DataFrame:
    delta_values = cfg.delta_min + 0.5 * (market_x + 1.0) * (cfg.delta_max - cfg.delta_min)
    delta_grid, cols, grid = response_feature_grid(response)

    rows = []
    for t in range(cfg.lookback - 1, len(market)):
        delta_seq = delta_values[t - cfg.lookback + 1:t + 1]
        feats = trajectory_features(delta_seq, delta_grid, cols, grid)
        base = market.iloc[t][["date", cfg.target_col, cfg.scalar_col, "rv_5d", "rv_20d", "rv_60d", "vix_close", "spy_drawdown_20d"]].to_dict()
        base["market_x_scaled_last"] = float(market_x[t])
        rows.append({**base, **feats})
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan"),
        "actual_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
    }


def class_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def evaluate(
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
    rows = []
    reg = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    reg.fit(X[train_idx], y_reg[train_idx])
    reg_pred = reg.predict(X)

    clf90 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf95 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf90.fit(X[train_idx], y_q90[train_idx])
    clf95.fit(X[train_idx], y_q95[train_idx])
    score90 = clf90.predict_proba(X)[:, 1]
    score95 = clf95.predict_proba(X)[:, 1]
    thr90 = float(np.quantile(score90[train_idx], 0.90))
    thr95 = float(np.quantile(score95[train_idx], 0.95))

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        base = {"model": name, "split": split, "n": int(len(idx))}
        rows.append({**base, "task": "regression", **regression_metrics(y_reg[idx], reg_pred[idx])})
        rows.append({**base, "task": "q90_warning", **class_metrics(y_q90[idx], score90[idx], thr90)})
        rows.append({**base, "task": "q95_crisis", **class_metrics(y_q95[idx], score95[idx], thr95)})

    return rows, pd.DataFrame({
        f"{name}_reg_pred": reg_pred,
        f"{name}_q90_score": score90,
        f"{name}_q95_score": score95,
    })


def plot_outputs(df: pd.DataFrame, out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["delta_last"], linewidth=0.8, label="last")
    ax.plot(df["date"], df["delta_mean"], linewidth=0.8, label="window mean")
    ax.set_ylabel("delta_end_over_omega")
    ax.set_title("Temporal market path through Rydberg operating window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_delta_path.png"), dpi=160)
    plt.close(fig)

    entropy_cols = [c for c in df.columns if c.startswith("rydberg_traj_entropy_bits_")]
    if entropy_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in ["rydberg_traj_entropy_bits_last", "rydberg_traj_entropy_bits_mean", "rydberg_traj_entropy_bits_max"]:
            if c in df.columns:
                ax.plot(df["date"], df[c], linewidth=0.8, label=c.replace("rydberg_traj_entropy_bits_", ""))
        ax.set_ylabel("entropy feature")
        ax.set_title("Temporal Rydberg entropy summaries")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_prefix.with_name(out_prefix.name + "_entropy_path.png"), dpi=160)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Temporal Rydberg response probe")
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--response-grid", type=Path, default=Path("artifacts/rydberg_tfim_response_zoom_v2.csv"))
    p.add_argument("--target-col", default="future_rv_20d")
    p.add_argument("--scalar-col", default="rv_accel_log_5_20")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--delta-min", type=float, default=0.40)
    p.add_argument("--delta-max", type=float, default=1.10)
    p.add_argument("--clip-quantile-low", type=float, default=0.05)
    p.add_argument("--clip-quantile-high", type=float, default=0.95)
    p.add_argument("--ridge-alpha", type=float, default=1000.0)
    p.add_argument("--out-prefix", type=Path, default=Path("results/tables/phase3_rydberg_temporal_probe"))
    args = p.parse_args()

    cfg = Config(
        data_path=args.data,
        response_grid_path=args.response_grid,
        target_col=args.target_col,
        scalar_col=args.scalar_col,
        lookback=args.lookback,
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
    keep = ["date", cfg.target_col, cfg.scalar_col, "rv_5d", "rv_20d", "rv_60d", "vix_close", "spy_drawdown_20d"]
    market = market[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    base_train_idx, _, _ = chronological_split(len(market), cfg.train_frac, cfg.val_frac)
    market_x, scaling = robust_scale(market[cfg.scalar_col].to_numpy(dtype=float), base_train_idx, cfg.clip_quantile_low, cfg.clip_quantile_high)
    response = pd.read_csv(cfg.response_grid_path)
    temporal = make_temporal_dataset(market, market_x, cfg, response)

    train_idx, val_idx, test_idx = chronological_split(len(temporal), cfg.train_frac, cfg.val_frac)
    y = temporal[cfg.target_col].to_numpy(dtype=float)
    q90 = float(np.quantile(y[train_idx], 0.90))
    q95 = float(np.quantile(y[train_idx], 0.95))
    y_q90 = (y >= q90).astype(int)
    y_q95 = (y >= q95).astype(int)

    raw_cols = [cfg.scalar_col, "market_x_scaled_last", "rv_5d", "rv_20d", "rv_60d", "vix_close", "spy_drawdown_20d"]
    temporal_cols = [c for c in temporal.columns if c.startswith("rydberg_traj_") or c.startswith("delta_")]
    feature_sets = {
        "raw_last_baseline": raw_cols,
        "rydberg_temporal_features": temporal_cols,
        "raw_plus_rydberg_temporal": raw_cols + temporal_cols,
    }

    all_rows = []
    pred_parts = [temporal[["date", cfg.target_col, cfg.scalar_col, "market_x_scaled_last", "delta_last", "delta_mean"]].copy()]
    for name, cols in feature_sets.items():
        X = temporal[cols].to_numpy(dtype=float)
        rows, pred = evaluate(name, X, y, y_q90, y_q95, train_idx, val_idx, test_idx, cfg.ridge_alpha)
        all_rows.extend(rows)
        pred_parts.append(pred)

    metrics = pd.DataFrame(all_rows)
    preds = pd.concat(pred_parts, axis=1)

    cfg.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_metrics.csv")
    preds_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_predictions.csv")
    features_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_features.csv")
    summary_path = cfg.out_prefix.with_name(cfg.out_prefix.name + "_summary.json")
    metrics.to_csv(metrics_path, index=False)
    preds.to_csv(preds_path, index=False)
    temporal.to_csv(features_path, index=False)

    summary = {
        "config": {**asdict(cfg), "data_path": str(cfg.data_path), "response_grid_path": str(cfg.response_grid_path), "out_prefix": str(cfg.out_prefix)},
        "n_rows": int(len(temporal)),
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "target_thresholds_train": {"q90": q90, "q95": q95},
        "market_scalar_scaling_train_only": scaling,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "note": "Temporal feature probe using interpolated Rydberg response-grid features. Still not a true recurrent AHS sequence simulation.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    plot_outputs(temporal, cfg.out_prefix.with_suffix(""))

    print("Saved:")
    print(metrics_path)
    print(preds_path)
    print(features_path)
    print(summary_path)
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
