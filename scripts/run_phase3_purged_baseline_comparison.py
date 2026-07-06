#!/usr/bin/env python3
"""Purged walk-forward comparison for raw, TFIM-PCA6, and ESN baselines.

This complements the Rydberg walk-forward runner. It uses the same expanding
fold geometry:

  train | validation | purge gap | test

Models:
  raw_phase2     full Phase 2 feature windows summarized by anchors/statistics
  tfim_pca6      train-only scaled full features -> train-only PCA-6 -> TFIM-QRC
  esn_phase2     full Phase 2 feature sequences -> fixed ESN -> ridge/logistic heads

Outputs stay in scratch/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import DEFAULT_FEATURE_COLUMNS, drop_nonfinite_model_rows, scale_splits_train_only
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    fit_qrc_readout,
    fit_tfim_qrc_regressor,
    make_qrc_sequence_splits,
    predict_qrc_readout,
    select_anchor_indices,
)

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--anchors", type=int, default=6)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--variants", nargs="*", default=["raw_phase2", "tfim_pca6", "esn_phase2"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tfim-qubits", type=int, default=6)
    p.add_argument("--tfim-observable-mode", choices=("z", "zx", "zxzz"), default="z")
    p.add_argument("--tfim-evolution-time", type=float, default=0.5)
    p.add_argument("--tfim-ridge-alpha", type=float, default=10.0)
    p.add_argument("--esn-units", type=int, default=200)
    p.add_argument("--esn-spectral-radius", type=float, default=0.9)
    p.add_argument("--esn-input-scale", type=float, default=0.5)
    p.add_argument("--esn-leak", type=float, default=0.3)
    p.add_argument("--esn-ridge-alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/purged_model_comparison"))
    p.add_argument("--tag", default="phase2_tfim_esn")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    """Backward-compatible wrapper around the shared Phase 3 protocol."""
    return make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )


def date_range(df, sl):
    a, b = sl
    return str(df["date"].iloc[a]), str(df["date"].iloc[b - 1])


def make_clean_splits(df, fold):
    frames = slice_fold_frames(df, fold)
    return {k: drop_nonfinite_model_rows(v, feature_columns=list(DEFAULT_FEATURE_COLUMNS), target_columns=[TARGET]) for k, v in frames.items()}


def raw_features(X, anchor_idx):
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def fit_regression_head(H, y_train, alpha):
    scaler = StandardScaler().fit(H["train"])
    model = Ridge(alpha=alpha)
    model.fit(scaler.transform(H["train"]), np.log(np.maximum(y_train, 1e-8)))
    return model, scaler


def predict_regression_head(model, scaler, H):
    return np.exp(model.predict(scaler.transform(H)))


def regression_metrics(H, y, alpha):
    model, scaler = fit_regression_head(H, y["train"], alpha)
    out = {}
    for split in SPLIT_NAMES:
        m = evaluate_volatility_forecast(y[split], predict_regression_head(model, scaler, H[split]))
        out[f"{split}_rmse"] = m.rmse
        out[f"{split}_qlike"] = m.qlike
        out[f"{split}_mz_r2"] = m.mz_r2
    return out


def warning_head(H, y, quantile):
    thr = float(np.quantile(y["train"], quantile))
    lab = {k: (v >= thr).astype(int) for k, v in y.items()}
    out = {"threshold": thr, "train_positive_rate": float(lab["train"].mean())}
    if lab["train"].sum() == 0 or lab["train"].sum() == len(lab["train"]):
        return {**out, "note": "degenerate train labels"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    clf.fit(H["train"], lab["train"])
    for split in ("val", "test"):
        yl = lab[split]
        if yl.sum() == 0 or yl.sum() == len(yl):
            out[split] = {"n": int(len(yl)), "n_pos": int(yl.sum()), "note": "degenerate labels"}
            continue
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int)
        out[split] = dict(n=int(len(yl)), n_pos=int(yl.sum()), auc=float(roc_auc_score(yl, score)),
                          ap=float(average_precision_score(yl, score)), f1=float(f1_score(yl, pred, zero_division=0)),
                          precision=float(precision_score(yl, pred, zero_division=0)),
                          recall=float(recall_score(yl, pred, zero_division=0)))
    return out


def pca6_splits(scaled_splits):
    pca = PCA(n_components=6, random_state=42)
    pca.fit(scaled_splits["train"][DEFAULT_FEATURE_COLUMNS].to_numpy(float))
    out = {}
    for name, split in scaled_splits.items():
        comps = pca.transform(split[DEFAULT_FEATURE_COLUMNS].to_numpy(float))
        frame = split[["date", TARGET]].copy().reset_index(drop=True)
        for i in range(6):
            frame[f"pc{i+1}"] = comps[:, i]
        out[name] = frame
    return out, [float(x) for x in pca.explained_variance_ratio_]


def esn_weights(n_inputs, n_units, spectral_radius, input_scale, seed):
    rng = np.random.default_rng(seed)
    win = rng.uniform(-input_scale, input_scale, size=(n_units, n_inputs + 1))
    w = rng.normal(0.0, 1.0, size=(n_units, n_units))
    mask = rng.random(size=w.shape) < 0.1
    w *= mask
    eig = np.linalg.eigvals(w)
    radius = max(float(np.max(np.abs(eig))), 1e-12)
    w *= spectral_radius / radius
    return win, w


def esn_transform(X, win, w, leak):
    rows = []
    for seq in X:
        state = np.zeros(w.shape[0])
        for x in seq:
            aug = np.concatenate([[1.0], x])
            proposal = np.tanh(win @ aug + w @ state)
            state = (1.0 - leak) * state + leak * proposal
        rows.append(state.copy())
    return np.asarray(rows, float)


def metric_row(fold_id, model, reg, q90, q95, meta):
    row = {"fold": fold_id, "model": model, **meta}
    for k in ("test_rmse", "test_qlike", "test_mz_r2"):
        row[k] = reg.get(k)
    for label, block in (("q90", q90), ("q95", q95)):
        test = block.get("test", {})
        for k in ("auc", "ap", "f1", "precision", "recall", "n_pos"):
            row[f"{label}_{k}"] = test.get(k)
    return row


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = make_folds(len(df), n_folds=args.n_folds, min_train=args.min_train, val_size=args.val_size, purge=args.purge)
    anchor_idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    rows = []
    summary = {"tag": args.tag, "config": {k: getattr(args, k) for k in vars(args) if k not in {"data", "out_dir"}}, "folds": []}

    for f in folds:
        fold_id = f["fold"]
        print(f"\n=== Fold {fold_id}: test {date_range(df, f['test'])} ===")
        clean = make_clean_splits(df, f)
        scaled, _ = scale_splits_train_only(clean, feature_columns=list(DEFAULT_FEATURE_COLUMNS), scaler_name="standard")
        seq_full = make_qrc_sequence_splits(scaled, feature_columns=list(DEFAULT_FEATURE_COLUMNS), target_column=TARGET, lookback_days=args.lookback)
        if args.stride > 1:
            seq_full = {k: (X[::args.stride], y[::args.stride], d[::args.stride].reset_index(drop=True)) for k, (X, y, d) in seq_full.items()}
        y = {k: v[1] for k, v in seq_full.items()}
        meta = {"train_dates": date_range(df, f["train"]), "val_dates": date_range(df, f["val"]),
                "purge_dates": date_range(df, f["purge"]), "test_dates": date_range(df, f["test"]),
                "train_n": int(len(y["train"])), "val_n": int(len(y["val"])), "test_n": int(len(y["test"]))}
        fold_summary = {"fold": fold_id, **meta, "variants": {}}

        for model in args.variants:
            print(f"-- {model}")
            if model == "raw_phase2":
                H = {k: raw_features(seq_full[k][0], anchor_idx) for k in seq_full}
                reg = regression_metrics(H, y, args.tfim_ridge_alpha)
            elif model == "tfim_pca6":
                pc_splits, pca_var = pca6_splits(scaled)
                seq_pc = make_qrc_sequence_splits(pc_splits, feature_columns=[f"pc{i+1}" for i in range(6)], target_column=TARGET, lookback_days=args.lookback)
                if args.stride > 1:
                    seq_pc = {k: (X[::args.stride], yy[::args.stride], d[::args.stride].reset_index(drop=True)) for k, (X, yy, d) in seq_pc.items()}
                cfg = TFIMQRCConfig(qubits=args.tfim_qubits, pca_components=6, lookback_days=args.lookback,
                                    anchor_count=args.anchors, anchor_policy=args.anchor_policy,
                                    observable_mode=args.tfim_observable_mode, evolution_time=args.tfim_evolution_time,
                                    ridge_alpha=args.tfim_ridge_alpha)
                res = fit_tfim_qrc_regressor(seq_pc, config=cfg, target=TARGET, verbose=args.verbose)
                H = {"train": res.train_features, "val": res.val_features, "test": res.test_features}
                reg = {"train_rmse": res.train_metrics.rmse, "val_rmse": res.val_metrics.rmse,
                       "test_rmse": res.test_metrics.rmse, "test_qlike": res.test_metrics.qlike,
                       "test_mz_r2": res.test_metrics.mz_r2, "pca6_explained_variance_sum": float(sum(pca_var))}
            elif model == "esn_phase2":
                win, w = esn_weights(seq_full["train"][0].shape[2], args.esn_units, args.esn_spectral_radius, args.esn_input_scale, args.seed + fold_id)
                H = {k: esn_transform(seq_full[k][0], win, w, args.esn_leak) for k in seq_full}
                reg = regression_metrics(H, y, args.esn_ridge_alpha)
            else:
                raise ValueError(f"Unknown model: {model}")

            q90 = warning_head(H, y, 0.90)
            q95 = warning_head(H, y, 0.95)
            fold_summary["variants"][model] = {"regression": reg, "q90_warning": q90, "q95_crisis": q95}
            rows.append(metric_row(fold_id, model, reg, q90, q95, meta))
            t = q95.get("test", {})
            if "auc" in t:
                print(f"   q95 AUC={t['auc']:.3f} AP={t['ap']:.3f} F1={t['f1']:.3f}")
            else:
                print(f"   q95 skipped: {t.get('note', q95.get('note', 'unknown'))}")
        summary["folds"].append(fold_summary)

    wide = pd.DataFrame(rows)
    out_csv = args.out_dir / f"purged_model_comparison_{args.tag}.csv"
    out_json = args.out_dir / f"purged_model_comparison_{args.tag}.json"
    wide.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Median test metrics by model ===")
    cols = ["test_rmse", "test_qlike", "test_mz_r2", "q95_auc", "q95_ap", "q95_f1"]
    print(wide.groupby("model")[cols].median(numeric_only=True).to_string())
    print(f"\nWrote {out_csv}\nWrote {out_json}")


if __name__ == "__main__":
    main()
