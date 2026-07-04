#!/usr/bin/env python3
"""Gatekeeper: among HAR-similar states, do temporal paths predict divergent futures?

For each test date, find a pool of historical TRAIN dates that are closest in the
exact canonical HAR state (rv_5d, rv_10d, rv_20d, rv_60d, vix_close, HAR forecast).
Then compare:

1. HAR-neighbor prediction: average outcome of the k closest HAR-state matches.
2. Path-reranked prediction: within the closest HAR-state pool, prefer matches
   with similar trailing path statistics.
3. Fingerprint-reranked prediction: within the same pool, prefer matches with
   similar compact spectral fingerprints.

Primary outcome: log future 20-day realized volatility.
Secondary outcomes: forward 20-day return and worst forward 20-day drawdown.

Promotion rule: path information is interesting only if reranking improves the
primary outcome on a majority of folds with positive median paired improvement.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")
FINGERPRINT_PATH = Path(__file__).with_name("run_har_failure_fingerprint_audit.py")
TARGET = "future_rv_20d"
HAR = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/diagnostics/har_matched_path"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--candidate-pool", type=int, default=50)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260704)
    return p.parse_args()


def add_forward_outcomes(df: pd.DataFrame):
    out = df.copy(); px = out["spy_adj_close"].to_numpy(float); n = len(out)
    ret20 = np.full(n, np.nan); dd20 = np.full(n, np.nan)
    ret20[:n-20] = px[20:] / px[:n-20] - 1.0
    for i in range(n-20):
        path = px[i+1:i+21] / px[i] - 1.0
        dd20[i] = float(np.min(path))
    out["log_future_rv_20d"] = np.log(np.maximum(out[TARGET], 1e-12))
    out["fwd_return_20d"] = ret20
    out["fwd_max_drawdown_20d"] = dd20
    return out


def fit_har(train: pd.DataFrame):
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(train[HAR], train[TARGET]); return m


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    sc = StandardScaler().fit(train[cols])
    return sc.transform(train[cols]), sc.transform(test[cols])


def row_distances(Xtrain: np.ndarray, x: np.ndarray):
    return np.sqrt(np.sum((Xtrain - x) ** 2, axis=1))


def metrics(y: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(pred); y = y[ok]; pred = pred[ok]
    return dict(n=len(y), rmse=float(np.sqrt(np.mean((y-pred)**2))), mae=float(np.mean(np.abs(y-pred))), corr=float(np.corrcoef(y,pred)[0,1]) if len(y)>2 and np.std(y)>0 and np.std(pred)>0 else np.nan)


def main():
    a = parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    master = load_module(MASTER_PATH, "master")
    fpmod = load_module(FINGERPRINT_PATH, "fpmod")
    df = pd.read_csv(a.data).sort_values("date").reset_index(drop=True)
    required = {"date", "spy_adj_close", TARGET, *HAR, *fpmod.PATH_COLS}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"Missing columns: {missing}")
    df = add_forward_outcomes(df)

    path = fpmod.rolling_stats(df, a.lookback); spec = fpmod.rolling_fingerprints(df, a.lookback)
    for c in path.columns: df[c] = path[c]
    for c in spec.columns: df[c] = spec[c]
    path_cols, spec_cols = list(path.columns), list(spec.columns)

    folds = master.make_folds(len(df), n_folds=5, min_train=2500, val_size=504, purge=60)
    outcomes = ["log_future_rv_20d", "fwd_return_20d", "fwd_max_drawdown_20d"]
    metric_rows, pair_rows, pred_rows = [], [], []

    for fold in folds:
        aligned = master.aligned_frames(df, fold, a.lookback)
        train, test = aligned["train"].copy(), aligned["test"].copy()
        har = fit_har(train)
        train["har_forecast"] = np.maximum(har.predict(train[HAR]), 1e-8)
        test["har_forecast"] = np.maximum(har.predict(test[HAR]), 1e-8)
        har_cols = HAR + ["har_forecast"]
        Htr, Hte = standardize(train, test, har_cols)
        Ptr, Pte = standardize(train, test, path_cols)
        Ftr, Fte = standardize(train, test, spec_cols)

        predictions = {o: {"har_neighbors": [], "path_rerank": [], "fingerprint_rerank": []} for o in outcomes}
        truths = {o: [] for o in outcomes}

        for i in range(len(test)):
            dh = row_distances(Htr, Hte[i])
            pool = np.argsort(dh)[: min(a.candidate_pool, len(train))]
            base = pool[: min(a.neighbors, len(pool))]
            dp = row_distances(Ptr[pool], Pte[i]); path_idx = pool[np.argsort(dp)[: min(a.neighbors, len(pool))]]
            dfp = row_distances(Ftr[pool], Fte[i]); fp_idx = pool[np.argsort(dfp)[: min(a.neighbors, len(pool))]]

            nearest = int(pool[0])
            pair = {
                "fold": fold["fold"], "test_date": test.iloc[i]["date"], "match_date": train.iloc[nearest]["date"],
                "har_distance": float(dh[nearest]), "path_distance": float(np.linalg.norm(Pte[i]-Ptr[nearest])),
                "fingerprint_distance": float(np.linalg.norm(Fte[i]-Ftr[nearest])),
            }
            for o in outcomes:
                yt = float(test.iloc[i][o]); ym = float(train.iloc[nearest][o]); pair[f"abs_diff__{o}"] = abs(yt-ym)
                truths[o].append(yt)
                predictions[o]["har_neighbors"].append(float(train.iloc[base][o].mean()))
                predictions[o]["path_rerank"].append(float(train.iloc[path_idx][o].mean()))
                predictions[o]["fingerprint_rerank"].append(float(train.iloc[fp_idx][o].mean()))
                pred_rows.extend([
                    dict(fold=fold["fold"], date=test.iloc[i]["date"], outcome=o, method="har_neighbors", y_true=yt, y_pred=predictions[o]["har_neighbors"][-1]),
                    dict(fold=fold["fold"], date=test.iloc[i]["date"], outcome=o, method="path_rerank", y_true=yt, y_pred=predictions[o]["path_rerank"][-1]),
                    dict(fold=fold["fold"], date=test.iloc[i]["date"], outcome=o, method="fingerprint_rerank", y_true=yt, y_pred=predictions[o]["fingerprint_rerank"][-1]),
                ])
            pair_rows.append(pair)

        for o in outcomes:
            y = np.asarray(truths[o], float)
            for method, values in predictions[o].items():
                metric_rows.append(dict(fold=fold["fold"], outcome=o, method=method, **metrics(y, np.asarray(values,float))))

    met = pd.DataFrame(metric_rows); pairs = pd.DataFrame(pair_rows); preds = pd.DataFrame(pred_rows)
    piv = met.pivot_table(index=["fold","outcome"], columns="method", values="rmse").reset_index()
    for m in ["path_rerank", "fingerprint_rerank"]:
        piv[f"rmse_improvement_{m}_vs_har"] = piv["har_neighbors"] - piv[m]

    assoc_rows = []
    for fold in sorted(pairs.fold.unique()):
        g = pairs[pairs.fold == fold]
        for o in outcomes:
            for dcol, label in [("path_distance","path"),("fingerprint_distance","fingerprint")]:
                rho, p = stats.spearmanr(g[dcol], g[f"abs_diff__{o}"], nan_policy="omit")
                assoc_rows.append(dict(fold=fold, outcome=o, distance_type=label, spearman_rho=float(rho), p_value=float(p)))
    assoc = pd.DataFrame(assoc_rows)
    summary = met.groupby(["outcome","method"], as_index=False).agg(rmse_median=("rmse","median"), rmse_mean=("rmse","mean"), mae_median=("mae","median"), corr_median=("corr","median"), n_folds=("fold","nunique"))

    met.to_csv(a.outdir/"per_fold_metrics.csv", index=False)
    summary.to_csv(a.outdir/"aggregate_metrics.csv", index=False)
    piv.to_csv(a.outdir/"paired_rmse_improvements.csv", index=False)
    assoc.to_csv(a.outdir/"matched_pair_associations.csv", index=False)
    pairs.to_csv(a.outdir/"matched_pairs.csv", index=False)
    preds.to_csv(a.outdir/"test_predictions.csv", index=False)
    manifest = {
        "data": str(a.data), "lookback": a.lookback, "candidate_pool": a.candidate_pool, "neighbors": a.neighbors,
        "har_state": har_cols, "path_features": path_cols, "fingerprint_features": spec_cols, "outcomes": outcomes,
        "primary_outcome": "log_future_rv_20d",
        "promotion_rule": "Path information survives only if path reranking improves primary-outcome test RMSE on a majority of folds with positive median paired improvement; fingerprint requires the same independently."
    }
    (a.outdir/"run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== HAR-matched path divergence gatekeeper ===")
    print(summary.to_string(index=False))
    print("\n=== Paired RMSE improvements (positive = better than HAR-neighbor matching) ===")
    print(piv.to_string(index=False))
    print("\n=== Matched-pair distance/outcome associations ===")
    print(assoc.to_string(index=False))
    print(f"\nWrote outputs to {a.outdir}")

if __name__ == "__main__": main()
