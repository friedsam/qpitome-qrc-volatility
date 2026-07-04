#!/usr/bin/env python3
"""Canonical Phase 3 model comparison under one purged walk-forward protocol.

This runner exists to answer one question cleanly: how do the serious model
families compare when folds, dates, labels, thresholds, metrics, and output
schema are identical?

Default model families:
- persistence_20d
- har_ridge
- raw_ridge
- esn_selected
- rydberg_temporal
- rydberg_memoryless
- rydberg_shuffled
- rydberg_multi_lb
- rydberg_multi_lb_memoryless
- rydberg_multi_lb_shuffled

Important protocol rules:
- All models are evaluated on the same post-lookback dates within each fold.
- Track A regression metrics use every requested fold.
- q90/q95 labels are defined from the fold's training target quantile.
- Degenerate test labels produce NaN classification metrics, not fold deletion.
- F1 thresholds are selected on validation only. If validation is degenerate,
  the fallback threshold is the corresponding training-score quantile.
- Exact-state, finite-shot simulator, and real hardware results must remain
  distinct protocols.
- Expensive Rydberg feature blocks are cached independently of readout fitting.

Outputs:
- per_fold_metrics.csv
- predictions.csv
- aggregate_metrics.csv
- run_manifest.json
- missing_models.csv (only when a requested model cannot run)

The script can run one model, selected folds, or the full comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
)
from qpitome_qrc.qrc.tfim_reservoir import select_anchor_indices

TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")
DEFAULT_MODELS = (
    "persistence_20d",
    "har_ridge",
    "raw_ridge",
    "esn_selected",
    "rydberg_temporal",
    "rydberg_memoryless",
    "rydberg_shuffled",
    "rydberg_multi_lb",
    "rydberg_multi_lb_memoryless",
    "rydberg_multi_lb_shuffled",
)
RYDBERG_MODELS = {m for m in DEFAULT_MODELS if m.startswith("rydberg_")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/canonical/current"))
    p.add_argument("--cache-dir", type=Path, default=Path("results/canonical/cache"))
    p.add_argument("--tag", default="phase3_current")
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--only-folds", nargs="*", type=int, default=None)

    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--lookback-fast", type=int, default=10)

    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--raw-ridge-alpha", type=float, default=10.0)
    p.add_argument("--har-alpha", type=float, default=1.0)

    p.add_argument("--esn-seeds", default="42")
    p.add_argument("--pca-components", type=int, default=6)

    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--total-time-us", type=float, default=0.55)
    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--row-gap-um", type=float, default=12.0)
    p.add_argument("--rydberg-ridge-alpha", type=float, default=10.0)
    p.add_argument("--train-stride", type=int, default=1)

    p.add_argument(
        "--protocols",
        nargs="*",
        default=["exact"],
        choices=["exact", "exact_train_noisy_test", "noisy_train_noisy_test"],
    )
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--shot-seeds", default="7")
    p.add_argument("--force-recompute", action="store_true")
    return p.parse_args()


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    first_test_start = min_train + val_size + purge
    if first_test_start >= n:
        raise ValueError("Not enough rows for requested fold construction")
    test_size = (n - first_test_start) // n_folds
    if test_size < 100:
        raise ValueError("Fold test size is too small")
    folds = []
    for i in range(n_folds):
        test_start = first_test_start + i * test_size
        test_end = n if i == n_folds - 1 else test_start + test_size
        val_end = test_start - purge
        val_start = val_end - val_size
        folds.append(
            {
                "fold": i + 1,
                "train": (0, val_start),
                "val": (val_start, val_end),
                "test": (test_start, test_end),
            }
        )
    return folds


def aligned_frames(df: pd.DataFrame, fold: dict, lookback: int) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        a, b = fold[split]
        frame = df.iloc[a:b].copy().reset_index(drop=True)
        if len(frame) < lookback:
            raise ValueError(f"Fold {fold['fold']} {split} shorter than lookback")
        out[split] = frame.iloc[lookback - 1 :].reset_index(drop=True)
    return out


def raw_features(X: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def labels_for(y_train: np.ndarray, values: dict[str, np.ndarray], q: float) -> tuple[float, dict[str, np.ndarray]]:
    threshold = float(np.quantile(y_train, q))
    return threshold, {s: (values[s] >= threshold).astype(int) for s in SPLITS}


def finite_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return {"ap": np.nan, "auc": np.nan, "n_pos": n_pos}
    return {
        "ap": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
        "n_pos": n_pos,
    }


def select_f1_threshold(train_labels: np.ndarray, val_labels: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray, q: float) -> tuple[float, str]:
    if 0 < int(val_labels.sum()) < len(val_labels):
        candidates = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 199)))
        best_threshold = float(candidates[0])
        best_f1 = -1.0
        for threshold in candidates:
            pred = (val_scores >= threshold).astype(int)
            value = f1_score(val_labels, pred, zero_division=0)
            if value > best_f1:
                best_f1 = float(value)
                best_threshold = float(threshold)
        return best_threshold, "validation_f1"
    return float(np.quantile(train_scores, q)), "train_score_quantile_fallback"


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    base = finite_binary_metrics(labels, scores)
    if not np.isfinite(base["ap"]):
        return {**base, "f1": np.nan, "precision": np.nan, "recall": np.nan, "called_rate": np.nan}
    pred = (scores >= threshold).astype(int)
    return {
        **base,
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "called_rate": float(pred.mean()),
    }


def track_a_metrics(y_true: np.ndarray, log_scores: np.ndarray) -> dict[str, float]:
    m = evaluate_volatility_forecast(y_true, np.exp(log_scores))
    return {
        "rmse": float(m.rmse),
        "qlike": float(m.qlike),
        "mz_alpha": float(m.mz_alpha),
        "mz_beta": float(m.mz_beta),
        "mz_r2": float(m.mz_r2),
    }


def fit_log_ridge(features: dict[str, np.ndarray], y: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features["train"], np.log(np.clip(y["train"], 1e-12, None)))
    return {s: model.predict(features[s]) for s in SPLITS}


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(n_inputs: int, n_reservoir: int, spectral_radius: float, input_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0.0, input_scale, size=(n_reservoir, n_inputs))
    W = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    W *= rng.random(W.shape) < 0.10
    return W_in, spectral_scale(W, spectral_radius)


def esn_states(X: np.ndarray, W_in: np.ndarray, W: np.ndarray, leak: float) -> np.ndarray:
    rows = []
    for window in X:
        h = np.zeros(W.shape[0])
        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new
        rows.append(np.concatenate([h, window[-1]]))
    return np.asarray(rows)


def make_sequence_arrays(frame: pd.DataFrame, feature_columns: list[str], lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame[feature_columns].to_numpy(float)
    target = frame[TARGET].to_numpy(float)
    dates = frame["date"].to_numpy()
    X, y, d = [], [], []
    for end in range(lookback - 1, len(frame)):
        X.append(values[end - lookback + 1 : end + 1])
        y.append(target[end])
        d.append(dates[end])
    return np.asarray(X), np.asarray(y), np.asarray(d)


def esn_grid(seeds: list[int]) -> list[dict]:
    base = [
        {"n": 300, "sr": 0.70, "inp": 0.30, "leak": 0.30, "alpha": 300.0},
        {"n": 300, "sr": 0.90, "inp": 0.30, "leak": 0.30, "alpha": 1000.0},
        {"n": 500, "sr": 0.70, "inp": 0.20, "leak": 0.50, "alpha": 1000.0},
        {"n": 500, "sr": 0.90, "inp": 0.20, "leak": 0.50, "alpha": 3000.0},
    ]
    return [{**cfg, "seed": seed} for cfg in base for seed in seeds]


def cache_key(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_or_build_feature_block(
    *,
    cache_dir: Path,
    cache_payload: dict,
    arrays: dict[str, np.ndarray],
    config: RydbergQRCConfig,
    force: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    key = cache_key(cache_payload)
    path = cache_dir / f"rydberg_{key}.npz"
    if path.exists() and not force:
        loaded = np.load(path)
        return {s: loaded[s] for s in SPLITS}, 0.0, str(path)
    start = time.perf_counter()
    block = {s: build_rydberg_feature_matrix(arrays[s], config) for s in SPLITS}
    elapsed = time.perf_counter() - start
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **block)
    return block, elapsed, str(path)


def evaluate_model(
    *,
    model_name: str,
    protocol: str,
    fold_id: int,
    y: dict[str, np.ndarray],
    dates: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    q90_threshold: float,
    lab90: dict[str, np.ndarray],
    q95_threshold: float,
    lab95: dict[str, np.ndarray],
    metadata: dict,
) -> tuple[dict, list[dict]]:
    row = {"fold": fold_id, "model": model_name, "protocol": protocol, **metadata}
    row.update({f"test_{k}": v for k, v in track_a_metrics(y["test"], scores["test"]).items()})
    row.update({f"val_{k}": v for k, v in track_a_metrics(y["val"], scores["val"]).items()})
    row["q90_target_threshold"] = q90_threshold
    row["q95_target_threshold"] = q95_threshold

    for q, labs in ((90, lab90), (95, lab95)):
        score_threshold, source = select_f1_threshold(
            labs["train"], labs["val"], scores["train"], scores["val"], q / 100.0
        )
        row[f"q{q}_score_threshold"] = score_threshold
        row[f"q{q}_threshold_source"] = source
        for split in ("val", "test"):
            values = classification_metrics(labs[split], scores[split], score_threshold)
            for key, value in values.items():
                row[f"q{q}_{split}_{key}"] = value

    pred_rows = []
    for split in SPLITS:
        y_pred = np.exp(scores[split])
        for i in range(len(y[split])):
            pred_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "protocol": protocol,
                    "split": split,
                    "date": dates[split][i],
                    "y_true": float(y[split][i]),
                    "log_score": float(scores[split][i]),
                    "y_pred": float(y_pred[i]),
                    "q90_label": int(lab90[split][i]),
                    "q95_label": int(lab95[split][i]),
                }
            )
    return row, pred_rows


def aggregate_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        c
        for c in per_fold.columns
        if c.startswith("test_") or c.startswith("q90_test_") or c.startswith("q95_test_")
    ]
    rows = []
    for (model, protocol), group in per_fold.groupby(["model", "protocol"], dropna=False):
        base = {"model": model, "protocol": protocol, "n_folds_regression": int(group["test_rmse"].notna().sum())}
        base["n_folds_q90_valid"] = int(group["q90_test_ap"].notna().sum())
        base["n_folds_q95_valid"] = int(group["q95_test_ap"].notna().sum())
        for col in numeric:
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            base[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            base[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            base[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    unknown = set(args.models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = {"date", TARGET, "rv_20d", "rv_5d", "rv_60d", args.level_col, args.rate_col, *FEATURE_COLUMNS}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    folds = make_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        keep = set(args.only_folds)
        folds = [f for f in folds if f["fold"] in keep]

    shot_seeds = [int(x) for x in args.shot_seeds.split(",") if x.strip()]
    esn_seeds = [int(x) for x in args.esn_seeds.split(",") if x.strip()]
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    missing_models: list[dict] = []

    base_cfg = RydbergQRCConfig(
        n_atoms_slow=args.n_slow,
        n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um,
        spacing_fast_um=args.spacing_fast_um,
        row_gap_um=args.row_gap_um,
        lookback_days=args.lookback,
        anchor_count=args.anchors,
        anchor_policy="even",
        reverse_anchors=True,
        total_time_us=args.total_time_us,
        shots=None,
    )
    fast_cfg = replace(base_cfg, lookback_days=args.lookback_fast)
    block_cfg = {
        "temporal": base_cfg,
        "fast": fast_cfg,
        "memoryless": replace(base_cfg, memory_mode="memoryless"),
        "fast_memoryless": replace(fast_cfg, memory_mode="memoryless"),
        "shuffled": replace(base_cfg, shuffle_anchors=True),
        "fast_shuffled": replace(fast_cfg, shuffle_anchors=True),
    }

    model_blocks = {
        "rydberg_temporal": ("temporal",),
        "rydberg_memoryless": ("memoryless",),
        "rydberg_shuffled": ("shuffled",),
        "rydberg_multi_lb": ("temporal", "fast"),
        "rydberg_multi_lb_memoryless": ("memoryless", "fast_memoryless"),
        "rydberg_multi_lb_shuffled": ("shuffled", "fast_shuffled"),
    }

    for fold in folds:
        fold_id = fold["fold"]
        print(f"\n=== Canonical fold {fold_id} ===")
        frames = {s: df.iloc[fold[s][0] : fold[s][1]].copy().reset_index(drop=True) for s in SPLITS}
        aligned = aligned_frames(df, fold, args.lookback)
        y = {s: aligned[s][TARGET].to_numpy(float) for s in SPLITS}
        dates = {s: aligned[s]["date"].to_numpy() for s in SPLITS}
        q90_target, lab90 = labels_for(y["train"], y, 0.90)
        q95_target, lab95 = labels_for(y["train"], y, 0.95)
        print(
            f"train/val/test={len(y['train'])}/{len(y['val'])}/{len(y['test'])}; "
            f"q95 val/test positives={lab95['val'].sum()}/{lab95['test'].sum()}"
        )

        # Cheap classical baselines use the same aligned dates as all reservoirs.
        if "persistence_20d" in args.models:
            scores = {s: np.log(np.clip(aligned[s]["rv_20d"].to_numpy(float), 1e-12, None)) for s in SPLITS}
            row, preds = evaluate_model(
                model_name="persistence_20d", protocol="classical", fold_id=fold_id,
                y=y, dates=dates, scores=scores,
                q90_threshold=q90_target, lab90=lab90, q95_threshold=q95_target, lab95=lab95,
                metadata={"quantum_tasks_per_date": 0, "shots": np.nan, "shot_seed": np.nan, "fit_seconds": 0.0, "feature_seconds": 0.0},
            )
            metric_rows.append(row); prediction_rows.extend(preds)

        if "har_ridge" in args.models:
            har_cols = ["rv_5d", "rv_20d", "rv_60d"]
            features = {s: aligned[s][har_cols].to_numpy(float) for s in SPLITS}
            start = time.perf_counter()
            scores = fit_log_ridge(features, y, args.har_alpha)
            fit_seconds = time.perf_counter() - start
            row, preds = evaluate_model(
                model_name="har_ridge", protocol="classical", fold_id=fold_id,
                y=y, dates=dates, scores=scores,
                q90_threshold=q90_target, lab90=lab90, q95_threshold=q95_target, lab95=lab95,
                metadata={"quantum_tasks_per_date": 0, "shots": np.nan, "shot_seed": np.nan, "fit_seconds": fit_seconds, "feature_seconds": 0.0},
            )
            metric_rows.append(row); prediction_rows.extend(preds)

        # Common level/rate sequences for raw and Rydberg models.
        need_sequence = "raw_ridge" in args.models or any(m in RYDBERG_MODELS for m in args.models)
        if need_sequence:
            seq = make_level_rate_sequence_splits(
                frames,
                level_col=args.level_col,
                rate_col=args.rate_col,
                target_column=TARGET,
                lookback_days=args.lookback,
            )
            seq_fast = make_level_rate_sequence_splits(
                frames,
                level_col=args.level_col,
                rate_col=args.rate_col,
                target_column=TARGET,
                lookback_days=args.lookback_fast,
            )
            seq_fast = {
                s: (X[len(X) - len(seq[s][0]) :], yy[len(yy) - len(seq[s][1]) :], d[len(d) - len(seq[s][2]) :])
                for s, (X, yy, d) in seq_fast.items()
            }
            if args.train_stride > 1:
                stride = args.train_stride
                seq = {s: ((X[::stride], yy[::stride], d[::stride]) if s == "train" else (X, yy, d)) for s, (X, yy, d) in seq.items()}
                seq_fast = {s: ((X[::stride], yy[::stride], d[::stride]) if s == "train" else (X, yy, d)) for s, (X, yy, d) in seq_fast.items()}
                y_fit = {s: seq[s][1] for s in SPLITS}
            else:
                y_fit = y

            anchor_idx = select_anchor_indices(args.lookback, args.anchors, "even")
            raw = {s: raw_features(seq[s][0], anchor_idx) for s in SPLITS}
            if "raw_ridge" in args.models:
                start = time.perf_counter()
                raw_scores = fit_log_ridge(raw, y_fit, args.raw_ridge_alpha)
                fit_seconds = time.perf_counter() - start
                row, preds = evaluate_model(
                    model_name="raw_ridge", protocol="classical", fold_id=fold_id,
                    y=y_fit, dates={s: seq[s][2] for s in SPLITS}, scores=raw_scores,
                    q90_threshold=float(np.quantile(y_fit['train'], 0.90)), lab90=labels_for(y_fit['train'], y_fit, 0.90)[1],
                    q95_threshold=float(np.quantile(y_fit['train'], 0.95)), lab95=labels_for(y_fit['train'], y_fit, 0.95)[1],
                    metadata={"quantum_tasks_per_date": 0, "shots": np.nan, "shot_seed": np.nan, "fit_seconds": fit_seconds, "feature_seconds": 0.0},
                )
                metric_rows.append(row); prediction_rows.extend(preds)

        if "esn_selected" in args.models:
            scaler = StandardScaler()
            pca = PCA(n_components=args.pca_components, random_state=42)
            pca.fit(scaler.fit_transform(frames["train"][FEATURE_COLUMNS]))
            pca_cols = [f"pca{i+1}" for i in range(args.pca_components)]
            esn_seq = {}
            for s in SPLITS:
                z = pca.transform(scaler.transform(frames[s][FEATURE_COLUMNS]))
                tmp = pd.DataFrame(z, columns=pca_cols)
                tmp[TARGET] = frames[s][TARGET].to_numpy()
                tmp["date"] = frames[s]["date"].to_numpy()
                esn_seq[s] = make_sequence_arrays(tmp, pca_cols, args.lookback)
            esn_y = {s: esn_seq[s][1] for s in SPLITS}
            esn_dates = {s: esn_seq[s][2] for s in SPLITS}
            _, esn_lab90 = labels_for(esn_y["train"], esn_y, 0.90)
            _, esn_lab95 = labels_for(esn_y["train"], esn_y, 0.95)
            candidates = []
            start_all = time.perf_counter()
            for cfg in esn_grid(esn_seeds):
                W_in, W = make_esn_weights(args.pca_components, cfg["n"], cfg["sr"], cfg["inp"], cfg["seed"])
                H = {s: esn_states(esn_seq[s][0], W_in, W, cfg["leak"]) for s in SPLITS}
                scores = fit_log_ridge(H, esn_y, cfg["alpha"])
                q90_val = finite_binary_metrics(esn_lab90["val"], scores["val"])["ap"]
                candidates.append((q90_val, cfg, scores))
            finite = [x for x in candidates if np.isfinite(x[0])]
            selected = max(finite, key=lambda x: x[0]) if finite else candidates[0]
            feature_fit_seconds = time.perf_counter() - start_all
            _, cfg, scores = selected
            row, preds = evaluate_model(
                model_name="esn_selected", protocol="classical", fold_id=fold_id,
                y=esn_y, dates=esn_dates, scores=scores,
                q90_threshold=float(np.quantile(esn_y['train'], 0.90)), lab90=esn_lab90,
                q95_threshold=float(np.quantile(esn_y['train'], 0.95)), lab95=esn_lab95,
                metadata={
                    "quantum_tasks_per_date": 0, "shots": np.nan, "shot_seed": cfg["seed"],
                    "fit_seconds": np.nan, "feature_seconds": feature_fit_seconds,
                    "selected_config": json.dumps(cfg, sort_keys=True), "selection_metric": "q90_val_ap",
                },
            )
            metric_rows.append(row); prediction_rows.extend(preds)

        requested_rydberg = [m for m in args.models if m in RYDBERG_MODELS]
        if requested_rydberg:
            needed_blocks = sorted({b for m in requested_rydberg for b in model_blocks[m]})
            sources = {name: (seq_fast if name.startswith("fast") else seq) for name in needed_blocks}

            exact_blocks: dict[str, dict[str, np.ndarray]] = {}
            exact_seconds: dict[str, float] = {}
            cache_paths: dict[str, str] = {}
            for name in needed_blocks:
                payload = {
                    "data": str(args.data), "fold": fold_id, "block": name,
                    "config": asdict(block_cfg[name]), "train_stride": args.train_stride,
                }
                exact_blocks[name], exact_seconds[name], cache_paths[name] = load_or_build_feature_block(
                    cache_dir=args.cache_dir,
                    cache_payload=payload,
                    arrays={s: sources[name][s][0] for s in SPLITS},
                    config=block_cfg[name],
                    force=args.force_recompute,
                )

            def features_for(model: str, blocks: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
                names = model_blocks[model]
                return {s: np.column_stack([blocks[name][s] for name in names]) for s in SPLITS}

            for model in requested_rydberg:
                exact_features = features_for(model, exact_blocks)
                if "exact" in args.protocols:
                    start = time.perf_counter()
                    scores = fit_log_ridge(exact_features, y_fit, args.rydberg_ridge_alpha)
                    fit_seconds = time.perf_counter() - start
                    row, preds = evaluate_model(
                        model_name=model, protocol="exact", fold_id=fold_id,
                        y=y_fit, dates={s: seq[s][2] for s in SPLITS}, scores=scores,
                        q90_threshold=float(np.quantile(y_fit['train'], 0.90)), lab90=labels_for(y_fit['train'], y_fit, 0.90)[1],
                        q95_threshold=float(np.quantile(y_fit['train'], 0.95)), lab95=labels_for(y_fit['train'], y_fit, 0.95)[1],
                        metadata={
                            "quantum_tasks_per_date": len(model_blocks[model]), "shots": np.nan, "shot_seed": np.nan,
                            "fit_seconds": fit_seconds,
                            "feature_seconds": sum(exact_seconds[b] for b in model_blocks[model]),
                            "cache_paths": json.dumps([cache_paths[b] for b in model_blocks[model]]),
                        },
                    )
                    metric_rows.append(row); prediction_rows.extend(preds)

                noisy_protocols = [p for p in args.protocols if p != "exact"]
                for seed in shot_seeds if noisy_protocols else []:
                    noisy_blocks = {}
                    noisy_seconds = {}
                    for name in model_blocks[model]:
                        cfg = replace(block_cfg[name], shots=args.shots, shot_seed=seed)
                        payload = {
                            "data": str(args.data), "fold": fold_id, "block": name,
                            "config": asdict(cfg), "train_stride": args.train_stride,
                        }
                        noisy_blocks[name], noisy_seconds[name], _ = load_or_build_feature_block(
                            cache_dir=args.cache_dir,
                            cache_payload=payload,
                            arrays={s: sources[name][s][0] for s in SPLITS},
                            config=cfg,
                            force=args.force_recompute,
                        )
                    noisy_features = features_for(model, noisy_blocks)
                    for protocol in noisy_protocols:
                        if protocol == "exact_train_noisy_test":
                            fit_features = {
                                "train": exact_features["train"],
                                "val": noisy_features["val"],
                                "test": noisy_features["test"],
                            }
                        elif protocol == "noisy_train_noisy_test":
                            fit_features = noisy_features
                        else:
                            raise ValueError(protocol)
                        start = time.perf_counter()
                        scores = fit_log_ridge(fit_features, y_fit, args.rydberg_ridge_alpha)
                        fit_seconds = time.perf_counter() - start
                        row, preds = evaluate_model(
                            model_name=model, protocol=protocol, fold_id=fold_id,
                            y=y_fit, dates={s: seq[s][2] for s in SPLITS}, scores=scores,
                            q90_threshold=float(np.quantile(y_fit['train'], 0.90)), lab90=labels_for(y_fit['train'], y_fit, 0.90)[1],
                            q95_threshold=float(np.quantile(y_fit['train'], 0.95)), lab95=labels_for(y_fit['train'], y_fit, 0.95)[1],
                            metadata={
                                "quantum_tasks_per_date": len(model_blocks[model]), "shots": args.shots, "shot_seed": seed,
                                "fit_seconds": fit_seconds,
                                "feature_seconds": sum(noisy_seconds.values()),
                            },
                        )
                        metric_rows.append(row); prediction_rows.extend(preds)

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = aggregate_metrics(per_fold) if len(per_fold) else pd.DataFrame()
    missing_df = pd.DataFrame(missing_models)

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    pred_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    missing_path = args.out_dir / f"missing_models_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    per_fold.to_csv(per_fold_path, index=False)
    predictions.to_csv(pred_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    manifest = {
        "tag": args.tag,
        "data": str(args.data),
        "models": args.models,
        "protocols": args.protocols,
        "folds": [f["fold"] for f in folds],
        "fold_policy": "all folds for regression; classification metrics NaN when test labels degenerate",
        "f1_policy": "validation-optimal threshold; training-score quantile fallback when validation labels degenerate",
        "q90_valid_fold_count_by_model": per_fold.groupby(["model", "protocol"])["q90_test_ap"].count().to_dict() if len(per_fold) else {},
        "q95_valid_fold_count_by_model": per_fold.groupby(["model", "protocol"])["q95_test_ap"].count().to_dict() if len(per_fold) else {},
        "args": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print("\n=== Canonical aggregate comparison ===")
    if len(aggregate):
        show = [c for c in [
            "model", "protocol", "n_folds_regression", "n_folds_q95_valid",
            "test_rmse_median", "test_qlike_median", "test_mz_r2_median",
            "q90_test_ap_median", "q95_test_ap_median", "q95_test_auc_median", "q95_test_f1_median",
        ] if c in aggregate.columns]
        print(aggregate[show].sort_values(["q95_test_ap_median", "test_rmse_median"], ascending=[False, True]).to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
