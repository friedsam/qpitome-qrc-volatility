#!/usr/bin/env python3
"""Gatekeeper audit: does temporal-path/fingerprint information predict HAR failure?

This script deliberately does NOT run ESN or QRC. It first asks whether a
credible information niche exists beyond the exact canonical HAR state.

Targets (defined fold-by-fold from cross-fitted TRAIN residuals only):
1. dangerous_underprediction: log(y / HAR_pred) above train q90
2. large_absolute_error: abs(log(y / HAR_pred)) above train q90

Feature blocks, nested in increasing complexity:
- har_state: exact HAR inputs + current HAR forecast
- extended_multiscale: HAR state + multiscale ratios/slopes and simple context
- path_stats: extended block + leakage-safe trailing path summaries
- fingerprint: path block + compact local spectral descriptors

Decision rule: fingerprint/path information earns further work only if it adds
stable out-of-sample AP beyond extended_multiscale, not merely beyond HAR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MASTER_PATH = Path(__file__).resolve().parents[2] / "canonical" / "run_master_comparison.py"
TARGET = "future_rv_20d"
HAR = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close"]
EXT = [
    "rv_ratio_5_20", "rv_ratio_20_60", "rv_slope_5_20", "rv_slope_20_60",
    "vix_log_change", "spy_drawdown_20d", "spy_log_hl_range",
]
PATH_COLS = ["spy_log_return", "spy_abs_log_return", "vix_log_change", "rv_ratio_5_20"]
PERIOD_BANDS = [(2, 5), (5, 10), (10, 20), (20, 40)]


def load_master():
    spec = importlib.util.spec_from_file_location("master", MASTER_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/diagnostics/har_failure_fingerprint"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--failure-quantile", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=20260704)
    return p.parse_args()


def fit_har(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[HAR], train[TARGET])
    return model


def crossfit_har_train(train: pd.DataFrame, min_initial: int = 800, n_splits: int = 5):
    n = len(train); pred = np.full(n, np.nan)
    if n <= min_initial + 100: min_initial = max(300, n // 3)
    edges = np.linspace(min_initial, n, n_splits + 1, dtype=int)
    for i in range(n_splits):
        a, b = edges[i], edges[i + 1]
        if b <= a: continue
        m = fit_har(train.iloc[:a])
        pred[a:b] = np.maximum(m.predict(train.iloc[a:b][HAR]), 1e-8)
    return pred


def rolling_stats(df: pd.DataFrame, lookback: int):
    out = pd.DataFrame(index=df.index)
    for col in PATH_COLS:
        s = df[col].astype(float)
        r = s.rolling(lookback, min_periods=lookback)
        out[f"{col}__mean"] = r.mean(); out[f"{col}__std"] = r.std(ddof=0)
        out[f"{col}__min"] = r.min(); out[f"{col}__max"] = r.max()
        out[f"{col}__delta"] = s - s.shift(lookback - 1)
        # OLS slope on equally spaced time points, closed form via centered x.
        x = np.arange(lookback, dtype=float); xc = x - x.mean(); den = float(np.dot(xc, xc))
        out[f"{col}__slope"] = s.rolling(lookback, min_periods=lookback).apply(
            lambda z: float(np.dot(xc, np.asarray(z) - np.mean(z)) / den), raw=True
        )
    return out


def spectral_one(z: np.ndarray):
    z = np.asarray(z, float); z = z - np.mean(z)
    if len(z) < 8 or np.var(z) <= 1e-16: return [0.0] * (len(PERIOD_BANDS) + 3)
    f, p = signal.periodogram(z, fs=1.0, window="hann", detrend="linear", scaling="spectrum")
    keep = f > 0; f, p = f[keep], p[keep]
    total = float(np.sum(p))
    if total <= 0: return [0.0] * (len(PERIOD_BANDS) + 3)
    per = 1.0 / f
    bands = [float(np.sum(p[(per >= lo) & (per < hi)]) / total) for lo, hi in PERIOD_BANDS]
    q = p / total
    entropy = float(-np.sum(q * np.log(np.maximum(q, 1e-15))) / np.log(len(q)))
    centroid = float(np.sum(f * p) / total)
    concentration = float(np.max(p) / total)
    return bands + [entropy, centroid, concentration]


def rolling_fingerprints(df: pd.DataFrame, lookback: int):
    cols = ["spy_abs_log_return", "vix_log_change", "rv_ratio_5_20"]
    names = [f"band_{a}_{b}" for a, b in PERIOD_BANDS] + ["entropy", "centroid", "concentration"]
    rows = {f"{c}__spec_{n}": np.full(len(df), np.nan) for c in cols for n in names}
    for i in range(lookback - 1, len(df)):
        for c in cols:
            vals = spectral_one(df[c].iloc[i - lookback + 1:i + 1].to_numpy(float))
            for n, v in zip(names, vals): rows[f"{c}__spec_{n}"][i] = v
    return pd.DataFrame(rows, index=df.index)


def finite_metrics(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(np.unique(y)) < 2: return dict(ap=np.nan, auc=np.nan, brier=np.nan, prevalence=float(y.mean()), n=len(y), n_pos=int(y.sum()))
    return dict(ap=float(average_precision_score(y, score)), auc=float(roc_auc_score(y, score)), brier=float(brier_score_loss(y, score)), prevalence=float(y.mean()), n=len(y), n_pos=int(y.sum()))


def main():
    a = parse_args(); a.outdir.mkdir(parents=True, exist_ok=True); master = load_master()
    df = pd.read_csv(a.data).sort_values("date").reset_index(drop=True)
    missing = sorted(set([TARGET, "date", *HAR, *EXT, *PATH_COLS]) - set(df.columns))
    if missing: raise ValueError(f"Missing columns: {missing}")

    stats = rolling_stats(df, a.lookback); specs = rolling_fingerprints(df, a.lookback)
    for c in stats.columns: df[c] = stats[c]
    for c in specs.columns: df[c] = specs[c]
    path_cols = list(stats.columns); spec_cols = list(specs.columns)

    folds = master.make_folds(len(df), n_folds=5, min_train=2500, val_size=504, purge=60)
    metric_rows, event_rows, coef_rows = [], [], []
    for fold in folds:
        aligned = master.aligned_frames(df, fold, a.lookback)
        train, val, test = aligned["train"], aligned["val"], aligned["test"]

        cf = crossfit_har_train(train)
        har_model = fit_har(train)
        pred = {
            "train": cf,
            "val": np.maximum(har_model.predict(val[HAR]), 1e-8),
            "test": np.maximum(har_model.predict(test[HAR]), 1e-8),
        }
        frames = {"train": train, "val": val, "test": test}
        err = {s: np.log(np.maximum(frames[s][TARGET].to_numpy(float), 1e-12) / pred[s]) for s in frames}
        valid_cf = np.isfinite(err["train"])
        thresholds = {
            "dangerous_underprediction": float(np.quantile(err["train"][valid_cf], a.failure_quantile)),
            "large_absolute_error": float(np.quantile(np.abs(err["train"][valid_cf]), a.failure_quantile)),
        }

        feature_sets = {
            "har_state": HAR + ["har_forecast"],
            "extended_multiscale": HAR + ["har_forecast"] + EXT,
            "path_stats": HAR + ["har_forecast"] + EXT + path_cols,
            "fingerprint": HAR + ["har_forecast"] + EXT + path_cols + spec_cols,
        }
        work = {}
        for s in frames:
            work[s] = frames[s].copy(); work[s]["har_forecast"] = pred[s]

        for target_name, threshold in thresholds.items():
            labels = {}
            for s in frames:
                e = err[s]
                labels[s] = (e > threshold).astype(int) if target_name == "dangerous_underprediction" else (np.abs(e) > threshold).astype(int)
            # Cross-fitted train rows only; test is never used for fitting/selection.
            train_ok = valid_cf.copy()
            for model_name, fcols in feature_sets.items():
                pipe = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=a.seed),
                )
                Xtr = work["train"].loc[train_ok, fcols]
                ytr = labels["train"][train_ok]
                pipe.fit(Xtr, ytr)
                for split in ("val", "test"):
                    score = pipe.predict_proba(work[split][fcols])[:, 1]
                    row = dict(fold=fold["fold"], target=target_name, model=model_name, split=split, threshold=threshold, **finite_metrics(labels[split], score))
                    metric_rows.append(row)
                    if split == "test":
                        for d, yt, sc, er, hp, yy in zip(work[split]["date"], labels[split], score, err[split], pred[split], work[split][TARGET]):
                            event_rows.append(dict(fold=fold["fold"], date=d, target=target_name, model=model_name, label=int(yt), score=float(sc), har_log_error=float(er), har_forecast=float(hp), y_true=float(yy)))

    metrics = pd.DataFrame(metric_rows)
    tests = metrics[metrics.split == "test"].copy()
    pivot = tests.pivot_table(index=["fold", "target"], columns="model", values="ap")
    for m in ["path_stats", "fingerprint"]:
        if m in pivot and "extended_multiscale" in pivot: pivot[f"delta_{m}_vs_extended"] = pivot[m] - pivot["extended_multiscale"]
    summary = tests.groupby(["target", "model"], as_index=False).agg(ap_median=("ap","median"), ap_mean=("ap","mean"), auc_median=("auc","median"), brier_median=("brier","median"), prevalence_median=("prevalence","median"), n_folds=("fold","nunique"))
    deltas = pivot.reset_index()

    metrics.to_csv(a.outdir / "per_fold_metrics.csv", index=False)
    summary.to_csv(a.outdir / "aggregate_metrics.csv", index=False)
    deltas.to_csv(a.outdir / "paired_ap_deltas.csv", index=False)
    pd.DataFrame(event_rows).to_csv(a.outdir / "test_predictions.csv", index=False)
    manifest = dict(data=str(a.data), lookback=a.lookback, failure_quantile=a.failure_quantile, targets=list(thresholds), feature_sets={k:v for k,v in feature_sets.items()}, decision_rule="Promote fingerprint/path branch only if test AP improves over extended_multiscale on a majority of folds with positive median paired delta.")
    (a.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== HAR failure fingerprint gatekeeper ===")
    print(summary.to_string(index=False))
    print("\n=== Paired test AP deltas ===")
    print(deltas.to_string(index=False))
    print(f"\nWrote outputs to {a.outdir}")

if __name__ == "__main__": main()
