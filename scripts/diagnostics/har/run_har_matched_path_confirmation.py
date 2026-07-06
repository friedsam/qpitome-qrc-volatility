#!/usr/bin/env python3
"""Untouched confirmation of the HAR-matched path hypothesis on post-2024 data.

Training/reference period ends at the frozen Phase 2 cutoff (2024-03-15).
The post-cutoff tail is used once as an external confirmation set. No fold
rebuilding, threshold tuning, or parameter selection is performed.

Frozen parameters inherited from the preregistered matched-path audit:
- lookback = 40
- candidate pool = 50
- neighbors = 10
- HAR state = rv_5d, rv_10d, rv_20d, rv_60d, vix_close, HAR forecast
- path and fingerprint features are imported unchanged from the prior audit
- primary outcome = log future_rv_20d
- promotion criterion = path reranking RMSE < HAR-neighbor RMSE
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINGERPRINT_PATH = Path(__file__).with_name("run_har_failure_fingerprint_audit.py")
MATCHED_PATH = Path(__file__).with_name("run_har_matched_path_audit.py")
TARGET = "future_rv_20d"
HAR = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase3_spy_vix_volatility_extended.csv"))
    p.add_argument("--cutoff", default="2024-03-15")
    p.add_argument("--outdir", type=Path, default=Path("results/diagnostics/har_matched_path_confirmation_2024_2026"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--candidate-pool", type=int, default=50)
    p.add_argument("--neighbors", type=int, default=10)
    return p.parse_args()


def fit_har(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[HAR], train[TARGET])
    return model


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    scaler = StandardScaler().fit(train[cols])
    return scaler.transform(train[cols]), scaler.transform(test[cols])


def row_distances(Xtrain: np.ndarray, x: np.ndarray):
    return np.sqrt(np.sum((Xtrain - x) ** 2, axis=1))


def metrics(y: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(pred)
    y = y[ok]; pred = pred[ok]
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "corr": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0 else np.nan,
    }


def main():
    a = parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    fpmod = load_module(FINGERPRINT_PATH, "fpmod_confirm")
    mmod = load_module(MATCHED_PATH, "matched_confirm")

    df = pd.read_csv(a.data).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(a.cutoff)
    df = mmod.add_forward_outcomes(df)

    path = fpmod.rolling_stats(df, a.lookback)
    spec = fpmod.rolling_fingerprints(df, a.lookback)
    for c in path.columns: df[c] = path[c]
    for c in spec.columns: df[c] = spec[c]
    path_cols, spec_cols = list(path.columns), list(spec.columns)

    train = df[df["date"] <= cutoff].copy().reset_index(drop=True)
    test = df[df["date"] > cutoff].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise RuntimeError("Training or confirmation split is empty")

    har = fit_har(train)
    train["har_forecast"] = np.maximum(har.predict(train[HAR]), 1e-8)
    test["har_forecast"] = np.maximum(har.predict(test[HAR]), 1e-8)
    har_cols = HAR + ["har_forecast"]

    Htr, Hte = standardize(train, test, har_cols)
    Ptr, Pte = standardize(train, test, path_cols)
    Ftr, Fte = standardize(train, test, spec_cols)

    outcomes = ["log_future_rv_20d", "fwd_return_20d", "fwd_max_drawdown_20d"]
    predictions = {o: {"har_neighbors": [], "path_rerank": [], "fingerprint_rerank": []} for o in outcomes}
    truths = {o: [] for o in outcomes}
    rows = []

    for i in range(len(test)):
        dh = row_distances(Htr, Hte[i])
        pool = np.argsort(dh)[: min(a.candidate_pool, len(train))]
        base = pool[: min(a.neighbors, len(pool))]
        dp = row_distances(Ptr[pool], Pte[i])
        path_idx = pool[np.argsort(dp)[: min(a.neighbors, len(pool))]]
        dfp = row_distances(Ftr[pool], Fte[i])
        fp_idx = pool[np.argsort(dfp)[: min(a.neighbors, len(pool))]]

        for o in outcomes:
            yt = float(test.iloc[i][o])
            truths[o].append(yt)
            vals = {
                "har_neighbors": float(train.iloc[base][o].mean()),
                "path_rerank": float(train.iloc[path_idx][o].mean()),
                "fingerprint_rerank": float(train.iloc[fp_idx][o].mean()),
            }
            for method, pred in vals.items():
                predictions[o][method].append(pred)
                rows.append({"date": test.iloc[i]["date"], "outcome": o, "method": method, "y_true": yt, "y_pred": pred})

    metric_rows = []
    for o in outcomes:
        y = np.asarray(truths[o], float)
        for method, values in predictions[o].items():
            metric_rows.append({"outcome": o, "method": method, **metrics(y, np.asarray(values, float))})

    metrics_df = pd.DataFrame(metric_rows)
    primary = metrics_df[metrics_df["outcome"] == "log_future_rv_20d"].set_index("method")
    path_improvement = float(primary.loc["har_neighbors", "rmse"] - primary.loc["path_rerank", "rmse"])
    fp_improvement = float(primary.loc["har_neighbors", "rmse"] - primary.loc["fingerprint_rerank", "rmse"])

    pd.DataFrame(rows).to_csv(a.outdir / "confirmation_predictions.csv", index=False)
    metrics_df.to_csv(a.outdir / "confirmation_metrics.csv", index=False)
    summary = {
        "cutoff": str(cutoff.date()),
        "train_rows": int(len(train)),
        "confirmation_rows": int(len(test)),
        "confirmation_start": str(test["date"].min().date()),
        "confirmation_end": str(test["date"].max().date()),
        "lookback": a.lookback,
        "candidate_pool": a.candidate_pool,
        "neighbors": a.neighbors,
        "path_rmse_improvement_vs_har": path_improvement,
        "fingerprint_rmse_improvement_vs_har": fp_improvement,
        "path_confirmation_pass": bool(path_improvement > 0),
        "fingerprint_confirmation_pass": bool(fp_improvement > 0),
    }
    (a.outdir / "confirmation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Untouched post-2024 HAR-matched path confirmation ===")
    print(f"Train/reference: {train['date'].min().date()} -> {train['date'].max().date()}  n={len(train)}")
    print(f"Confirmation:    {test['date'].min().date()} -> {test['date'].max().date()}  n={len(test)}")
    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Primary-outcome confirmation ===")
    print(f"Path RMSE improvement vs HAR-neighbors:        {path_improvement:+.6f}")
    print(f"Fingerprint RMSE improvement vs HAR-neighbors: {fp_improvement:+.6f}")
    print(f"Path confirmation pass:        {path_improvement > 0}")
    print(f"Fingerprint confirmation pass: {fp_improvement > 0}")
    print(f"\nWrote outputs to {a.outdir}")


if __name__ == "__main__":
    main()
