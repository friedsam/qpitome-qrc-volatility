#!/usr/bin/env python3
"""Purged walk-forward ESN ridge-ranking benchmark for Phase 3 comparisons.

ESN states -> ridge regression on log future RV -> ranking scores for q90/q95 AP.
Also reports official Track A metrics: RMSE, QLIKE, and Mincer-Zarnowitz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def make_folds(n, n_folds, min_train, val_size, purge):
    """Backward-compatible wrapper around the shared Phase 3 protocol."""

    return make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )


def make_sequence_arrays(frame, feature_columns, lookback):
    values = frame[feature_columns].to_numpy(float)
    target = frame[TARGET].to_numpy(float)
    X, y = [], []
    for end in range(lookback - 1, len(frame)):
        X.append(values[end - lookback + 1 : end + 1])
        y.append(target[end])
    return np.asarray(X), np.asarray(y)


def spectral_scale(W, radius):
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(n_inputs, n_reservoir, spectral_radius, input_scale, seed):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0.0, input_scale, size=(n_reservoir, n_inputs))
    W = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    W *= rng.random(W.shape) < 0.10
    return W_in, spectral_scale(W, spectral_radius)


def esn_states(X, W_in, W, leak):
    rows = []
    for window in X:
        h = np.zeros(W.shape[0])
        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new
        rows.append(np.concatenate([h, window[-1]]))
    return np.asarray(rows)


def fit_ridge_scores(H, y, alpha):
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    model.fit(scaler.fit_transform(H["train"]), np.log(np.maximum(y["train"], 1e-8)))
    return {split: model.predict(scaler.transform(H[split])) for split in SPLIT_NAMES}


def label_blocks(y, quantile):
    threshold = float(np.quantile(y["train"], quantile))
    return threshold, {split: (vals >= threshold).astype(int) for split, vals in y.items()}


def binary_metrics(labels, scores):
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return {"ap": np.nan, "auc": np.nan, "f1": np.nan, "n_pos": n_pos}
    pred = (scores >= np.median(scores)).astype(int)
    return {
        "ap": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "n_pos": n_pos,
    }


def track_a_metrics(y_true, log_scores):
    m = evaluate_volatility_forecast(y_true, np.exp(log_scores))
    return {"rmse": m.rmse, "qlike": m.qlike, "mz_alpha": m.mz_alpha, "mz_beta": m.mz_beta, "mz_r2": m.mz_r2}


def make_grid(seeds):
    base = [
        {"n": 300, "sr": 0.70, "inp": 0.30, "leak": 0.30, "alpha": 300.0},
        {"n": 300, "sr": 0.90, "inp": 0.30, "leak": 0.30, "alpha": 1000.0},
        {"n": 500, "sr": 0.70, "inp": 0.20, "leak": 0.50, "alpha": 1000.0},
        {"n": 500, "sr": 0.90, "inp": 0.20, "leak": 0.50, "alpha": 3000.0},
    ]
    grid = []
    for cfg in base:
        for seed in seeds:
            row = dict(cfg, seed=seed)
            row["config_id"] = f"esn_n{row['n']}_sr{row['sr']}_inp{row['inp']}_leak{row['leak']}_alpha{row['alpha']}_seed{seed}"
            grid.append(row)
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("scratch/esn_ridge"))
    p.add_argument("--tag", default="esn_ridge")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--pca-components", type=int, default=6)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--only-folds", nargs="*", type=int)
    p.add_argument("--seeds", default="42")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = ["date", TARGET] + list(FEATURE_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    folds = make_folds(len(df), args.n_folds, args.min_train, args.val_size, args.purge)
    if args.only_folds:
        folds = [f for f in folds if f["fold"] in set(args.only_folds)]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = make_grid(seeds)
    fallback_id = grid[0]["config_id"]
    rows, selected_rows = [], []

    for f in folds:
        split_frames = slice_fold_frames(df, f)
        scaler = StandardScaler()
        pca = PCA(n_components=args.pca_components, random_state=42)
        pca.fit(scaler.fit_transform(split_frames["train"][FEATURE_COLUMNS]))
        pca_cols = [f"pca{i+1}" for i in range(args.pca_components)]
        seq = {}
        for split in SPLIT_NAMES:
            z = pca.transform(scaler.transform(split_frames[split][FEATURE_COLUMNS]))
            frame = pd.DataFrame(z, columns=pca_cols)
            frame[TARGET] = split_frames[split][TARGET].to_numpy()
            seq[split] = make_sequence_arrays(frame, pca_cols, args.lookback)
        X = {s: seq[s][0] for s in SPLIT_NAMES}
        y = {s: seq[s][1] for s in SPLIT_NAMES}
        q90_thr, lab90 = label_blocks(y, 0.90)
        q95_thr, lab95 = label_blocks(y, 0.95)
        print(f"=== fold {f['fold']} (train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}, val/test q90 pos {lab90['val'].sum()}/{lab90['test'].sum()}, q95 pos {lab95['val'].sum()}/{lab95['test'].sum()}) ===")

        fold_rows = []
        for cfg in grid:
            W_in, W = make_esn_weights(X["train"].shape[2], cfg["n"], cfg["sr"], cfg["inp"], cfg["seed"])
            H = {s: esn_states(X[s], W_in, W, cfg["leak"]) for s in SPLIT_NAMES}
            scores = fit_ridge_scores(H, y, cfg["alpha"])
            q90_val = binary_metrics(lab90["val"], scores["val"])
            q90_test = binary_metrics(lab90["test"], scores["test"])
            q95_val = binary_metrics(lab95["val"], scores["val"])
            q95_test = binary_metrics(lab95["test"], scores["test"])
            val_track = track_a_metrics(y["val"], scores["val"])
            test_track = track_a_metrics(y["test"], scores["test"])
            row = {
                "fold": f["fold"], "strategy": "esn_ridge", "config_id": cfg["config_id"],
                "q90_threshold": q90_thr, "q95_threshold": q95_thr,
                "q90_val_ap": q90_val["ap"], "q90_val_auc": q90_val["auc"], "q90_val_n_pos": q90_val["n_pos"],
                "q90_test_ap": q90_test["ap"], "q90_test_auc": q90_test["auc"], "q90_test_f1": q90_test["f1"], "q90_test_n_pos": q90_test["n_pos"],
                "q95_val_ap": q95_val["ap"], "q95_val_auc": q95_val["auc"], "q95_val_n_pos": q95_val["n_pos"],
                "q95_test_ap": q95_test["ap"], "q95_test_auc": q95_test["auc"], "q95_test_f1": q95_test["f1"], "q95_test_n_pos": q95_test["n_pos"],
                **{f"val_{k}": v for k, v in val_track.items()}, **{f"test_{k}": v for k, v in test_track.items()}, **cfg,
            }
            fold_rows.append(row)
            rows.append(row)

        fold_df = pd.DataFrame(fold_rows)
        finite = fold_df[np.isfinite(fold_df["q90_val_ap"])]
        if len(finite):
            best = finite.sort_values("q90_val_ap", ascending=False).iloc[0].to_dict()
            best["selection_metric"] = "q90_val_ap"
        else:
            best = fold_df.loc[fold_df["config_id"] == fallback_id].iloc[0].to_dict()
            best["selection_metric"] = "fixed_default_config"
            print(f"  q90 validation degenerate; fixed fallback {fallback_id}")
        best["strategy"] = "esn_ridge_selected"
        selected_rows.append(best)
        q90_text = "nan" if not np.isfinite(best["q90_val_ap"]) else f"{best['q90_val_ap']:.4f}"
        print(f"selected {best['config_id']} via={best['selection_metric']} q90_val_ap={q90_text} q95_test_ap={best['q95_test_ap']:.4f} test_rmse={best['test_rmse']:.4f} test_qlike={best['test_qlike']:.4f}")

    all_df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    grid_path = args.out_dir / f"esn_ridge_grid_{args.tag}.csv"
    selected_path = args.out_dir / f"esn_ridge_selected_{args.tag}.csv"
    summary_path = args.out_dir / f"esn_ridge_summary_{args.tag}.json"
    all_df.to_csv(grid_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    ap_cols = ["q90_val_ap", "q90_test_ap", "q95_test_ap", "q95_test_auc", "q95_test_f1"]
    track_cols = ["test_rmse", "test_qlike", "test_mz_alpha", "test_mz_beta", "test_mz_r2"]
    summary = {"tag": args.tag, "selection": "q90 val AP with fixed default fallback when degenerate", "fallback_config_id": fallback_id, "folds": [f["fold"] for f in folds], "grid_size": len(grid), "median_selected_metrics": selected_df[ap_cols + track_cols].median(numeric_only=True).to_dict()}
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Selected ESN ridge medians: ranking ===")
    print(selected_df[ap_cols].median(numeric_only=True).to_string())
    print("\n=== Selected ESN ridge medians: official Track A ===")
    print(selected_df[track_cols].median(numeric_only=True).to_string())
    print(f"\nWrote {grid_path}\nWrote {selected_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
