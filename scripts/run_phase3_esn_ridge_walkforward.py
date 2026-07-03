#!/usr/bin/env python3
"""Purged walk-forward ESN ridge-ranking benchmark for Phase 3 AP comparisons.

This script gives the ESN baseline the same key readout upgrade used in the
Rydberg AP-push harness:

  ESN states -> ridge regression on log(future_rv_20d)
  ridge prediction -> ranking score for q90/q95 AP

Model selection is by q90 validation AP because q95 validation labels are
structurally degenerate in some folds. q95 test AP is reported for every fold
where the test labels are nondegenerate.

This is a classical comparator. It does not simulate shot noise or hardware.
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

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    first_test_start = min_train + val_size + purge
    if first_test_start >= n:
        raise ValueError("Not enough rows for requested min_train/val_size/purge")
    test_size = (n - first_test_start) // n_folds
    if test_size < 100:
        raise ValueError(f"test_size too small: {test_size}")

    folds = []
    for i in range(n_folds):
        test_start = first_test_start + i * test_size
        test_end = n if i == n_folds - 1 else test_start + test_size
        val_end = test_start - purge
        val_start = val_end - val_size
        train_end = val_start
        if train_end < min_train:
            raise ValueError("Invalid fold construction")
        folds.append(
            dict(
                fold=i + 1,
                train=(0, train_end),
                val=(val_start, val_end),
                purge=(val_end, test_start),
                test=(test_start, test_end),
            )
        )
    return folds


def make_sequence_arrays(frame: pd.DataFrame, *, feature_columns: list[str], target_column: str, lookback: int):
    X, y, dates = [], [], []
    values = frame[feature_columns].to_numpy(dtype=float)
    target = frame[target_column].to_numpy(dtype=float)
    date_values = pd.to_datetime(frame["date"])
    for end in range(lookback - 1, len(frame)):
        X.append(values[end - lookback + 1 : end + 1])
        y.append(target[end])
        dates.append(date_values.iloc[end])
    return np.asarray(X), np.asarray(y), pd.Series(dates, name="date")


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    eig = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eig)))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(n_inputs: int, n_reservoir: int, spectral_radius: float, input_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0.0, input_scale, size=(n_reservoir, n_inputs))
    W = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    mask = rng.random(W.shape) < 0.10
    W = spectral_scale(W * mask, spectral_radius)
    return W_in, W


def esn_states(X: np.ndarray, W_in: np.ndarray, W: np.ndarray, leak: float) -> np.ndarray:
    rows = []
    for window in X:
        h = np.zeros(W.shape[0])
        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new
        # Append current PCA input as a skip connection, matching the existing ESN export.
        rows.append(np.concatenate([h, window[-1]]))
    return np.asarray(rows)


def fit_ridge_scores(H: dict[str, np.ndarray], y: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    H_train = scaler.fit_transform(H["train"])
    model = Ridge(alpha=alpha)
    model.fit(H_train, np.log(np.maximum(y["train"], 1e-8)))
    return {split: model.predict(scaler.transform(H[split])) for split in SPLIT_NAMES}


def label_blocks(y: dict[str, np.ndarray], quantile: float):
    threshold = float(np.quantile(y["train"], quantile))
    labels = {split: (vals >= threshold).astype(int) for split, vals in y.items()}
    return threshold, labels


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    n_pos = int(labels.sum())
    out = {"n": int(len(labels)), "n_pos": n_pos}
    if n_pos == 0 or n_pos == len(labels):
        return {**out, "auc": np.nan, "ap": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan, "degenerate": True}
    pred = (scores >= np.median(scores)).astype(int)
    # AP/AUC use the raw ranking score. F1 is secondary and threshold-dependent.
    return {
        **out,
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "degenerate": False,
    }


def make_grid(args: argparse.Namespace) -> list[dict]:
    # Compact grid based on the existing ESN export, with optional seed expansion.
    base = [
        {"n": 300, "sr": 0.70, "inp": 0.30, "leak": 0.30, "alpha": 300.0},
        {"n": 300, "sr": 0.90, "inp": 0.30, "leak": 0.30, "alpha": 1000.0},
        {"n": 500, "sr": 0.70, "inp": 0.20, "leak": 0.50, "alpha": 1000.0},
        {"n": 500, "sr": 0.90, "inp": 0.20, "leak": 0.50, "alpha": 3000.0},
    ]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = []
    for cfg in base:
        for seed in seeds:
            row = dict(cfg)
            row["seed"] = seed
            row["config_id"] = f"esn_n{row['n']}_sr{row['sr']}_inp{row['inp']}_leak{row['leak']}_alpha{row['alpha']}_seed{seed}"
            grid.append(row)
    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("scratch/esn_ridge"))
    p.add_argument("--tag", default="esn_ridge")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--pca-components", type=int, default=6)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--seeds", default="42", help="Comma-separated ESN seeds, e.g. 42 or 11,23,42")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    required = ["date", TARGET] + list(FEATURE_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    folds = make_folds(len(df), n_folds=args.n_folds, min_train=args.min_train, val_size=args.val_size, purge=args.purge)
    if args.only_folds:
        keep = set(args.only_folds)
        folds = [f for f in folds if f["fold"] in keep]

    grid = make_grid(args)
    rows = []
    selected_rows = []

    for f in folds:
        fold_id = f["fold"]
        split_frames = {name: df.iloc[f[name][0] : f[name][1]].copy().reset_index(drop=True) for name in SPLIT_NAMES}

        feature_scaler = StandardScaler()
        pca = PCA(n_components=args.pca_components, random_state=42)
        train_scaled = feature_scaler.fit_transform(split_frames["train"][FEATURE_COLUMNS])
        pca.fit(train_scaled)

        pca_cols = [f"pca{i+1}" for i in range(args.pca_components)]
        pca_frames = {}
        for split in SPLIT_NAMES:
            z = pca.transform(feature_scaler.transform(split_frames[split][FEATURE_COLUMNS]))
            frame = pd.DataFrame(z, columns=pca_cols)
            frame["date"] = split_frames[split]["date"].values
            frame[TARGET] = split_frames[split][TARGET].values
            pca_frames[split] = frame

        seq = {split: make_sequence_arrays(pca_frames[split], feature_columns=pca_cols, target_column=TARGET, lookback=args.lookback) for split in SPLIT_NAMES}
        X = {split: seq[split][0] for split in SPLIT_NAMES}
        y = {split: seq[split][1] for split in SPLIT_NAMES}

        q90_thr, lab90 = label_blocks(y, 0.90)
        q95_thr, lab95 = label_blocks(y, 0.95)
        print(
            f"=== fold {fold_id} "
            f"(train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}, "
            f"val/test q95 pos {int(lab95['val'].sum())}/{int(lab95['test'].sum())}) ==="
        )

        fold_rows = []
        for cfg in grid:
            W_in, W = make_esn_weights(X["train"].shape[2], cfg["n"], cfg["sr"], cfg["inp"], cfg["seed"])
            H = {split: esn_states(X[split], W_in, W, cfg["leak"]) for split in SPLIT_NAMES}
            scores = fit_ridge_scores(H, y, cfg["alpha"])

            q90_val = binary_metrics(lab90["val"], scores["val"])
            q90_test = binary_metrics(lab90["test"], scores["test"])
            q95_val = binary_metrics(lab95["val"], scores["val"])
            q95_test = binary_metrics(lab95["test"], scores["test"])

            row = {
                "fold": fold_id,
                "strategy": "esn_ridge",
                "selection_metric": "q90_val_ap",
                "config_id": cfg["config_id"],
                "q90_threshold": q90_thr,
                "q95_threshold": q95_thr,
                "q90_val_ap": q90_val["ap"],
                "q90_val_auc": q90_val["auc"],
                "q90_val_n_pos": q90_val["n_pos"],
                "q90_test_ap": q90_test["ap"],
                "q90_test_auc": q90_test["auc"],
                "q90_test_f1": q90_test["f1"],
                "q90_test_n_pos": q90_test["n_pos"],
                "q95_val_ap": q95_val["ap"],
                "q95_val_auc": q95_val["auc"],
                "q95_val_n_pos": q95_val["n_pos"],
                "q95_test_ap": q95_test["ap"],
                "q95_test_auc": q95_test["auc"],
                "q95_test_f1": q95_test["f1"],
                "q95_test_n_pos": q95_test["n_pos"],
                **cfg,
            }
            fold_rows.append(row)
            rows.append(row)

        fold_df = pd.DataFrame(fold_rows)
        # q90 validation AP is always populated and is the honest fallback selector.
        best = fold_df.sort_values("q90_val_ap", ascending=False).iloc[0].to_dict()
        best["strategy"] = "esn_ridge_selected_by_q90_val"
        selected_rows.append(best)
        print(
            "selected",
            best["config_id"],
            f"q90_val_ap={best['q90_val_ap']:.4f}",
            f"q95_test_ap={best['q95_test_ap']:.4f}",
        )

    all_df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)

    all_path = args.out_dir / f"esn_ridge_grid_{args.tag}.csv"
    selected_path = args.out_dir / f"esn_ridge_selected_{args.tag}.csv"
    summary_path = args.out_dir / f"esn_ridge_summary_{args.tag}.json"

    all_df.to_csv(all_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    med = selected_df[["q90_val_ap", "q90_test_ap", "q95_test_ap", "q95_test_auc", "q95_test_f1"]].median(numeric_only=True).to_dict()
    summary = {
        "tag": args.tag,
        "selection": "per-fold best ESN grid config by q90 validation AP",
        "folds": [int(f["fold"]) for f in folds],
        "grid_size": len(grid),
        "median_selected_metrics": med,
        "outputs": {"grid": str(all_path), "selected": str(selected_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Selected ESN ridge medians ===")
    print(selected_df.groupby("strategy")[["q90_val_ap", "q90_test_ap", "q95_test_ap", "q95_test_auc", "q95_test_f1"]].median(numeric_only=True).to_string())
    print(f"\nWrote {all_path}")
    print(f"Wrote {selected_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
