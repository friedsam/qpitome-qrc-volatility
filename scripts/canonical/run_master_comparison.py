#!/usr/bin/env python3
"""Canonical classical model comparison under one purged walk-forward protocol.

The retained Stage 1 comparison covers persistence, HAR ridge, raw temporal
ridge, and the selected deterministic NumPy ESN. GARCH and LSTM retain their
standalone runners because their fitting and artifact schemas are materially
different.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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

from baselines.numpy_esn import (
    esn_states as _esn_states,
    historical_numpy_esn_grid as _historical_numpy_esn_grid,
    make_esn_weights as _make_esn_weights,
    spectral_scale as _spectral_scale,
)
from data.features import FEATURE_COLUMNS
from evaluation.metrics import evaluate_volatility_forecast
from evaluation.walkforward import align_fold_frames, make_purged_walkforward_folds
from experiments.runs import begin_run

TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")
DEFAULT_MODELS = ("persistence_20d", "har_ridge", "raw_ridge", "esn_selected")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/spy_vix_volatility/spy_vix_volatility.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/canonical/run_master_comparison"),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default="classical_current")
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--raw-ridge-alpha", type=float, default=10.0)
    p.add_argument("--har-alpha", type=float, default=1.0)
    p.add_argument("--esn-seeds", default="42")
    p.add_argument("--pca-components", type=int, default=6)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--train-stride", type=int, default=1)
    return p.parse_args()


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    shared = make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )
    return [
        {
            "fold": fold["fold"],
            "train": fold["train"],
            "val": fold["val"],
            "test": fold["test"],
        }
        for fold in shared
    ]


def aligned_frames(df: pd.DataFrame, fold: dict, lookback: int) -> dict[str, pd.DataFrame]:
    return align_fold_frames(df, fold, lookback)


def select_anchor_indices(lookback_days: int, anchor_count: int) -> np.ndarray:
    if anchor_count < 1:
        raise ValueError("anchor_count must be >= 1")
    if anchor_count > lookback_days:
        raise ValueError("anchor_count must be <= lookback_days")
    return np.linspace(0, lookback_days - 1, anchor_count).round().astype(int)


def ensure_market_scalar(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in frame.columns:
        return frame
    out = frame.copy()
    if column == "rv_accel_log_5_20":
        out[column] = np.log(
            out["rv_5d"].clip(lower=1e-12) / out["rv_20d"].clip(lower=1e-12)
        )
    elif column == "vix_rv_spread":
        out[column] = np.log(
            (out["vix_close"].clip(lower=1e-12) / 100.0)
            / out["rv_20d"].clip(lower=1e-12)
        )
    else:
        raise ValueError(f"Unknown scalar column {column!r}")
    return out


def _robust_scale_params(values: np.ndarray) -> tuple[float, float]:
    low = float(np.nanquantile(values, 0.01))
    high = float(np.nanquantile(values, 0.99))
    median = float(np.nanmedian(values))
    half = 0.5 * (high - low)
    if not np.isfinite(half) or half <= 0:
        raise ValueError("Invalid robust scale")
    return median, half


def make_level_rate_sequence_splits(
    splits: dict[str, pd.DataFrame],
    *,
    level_col: str,
    rate_col: str,
    target_column: str,
    lookback_days: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    prepared = {
        name: ensure_market_scalar(ensure_market_scalar(frame, level_col), rate_col)
        for name, frame in splits.items()
    }
    train = prepared["train"]
    params = {
        column: _robust_scale_params(train[column].to_numpy(float))
        for column in (level_col, rate_col)
    }

    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, frame in prepared.items():
        channels = []
        for column in (level_col, rate_col):
            median, half = params[column]
            channels.append(
                np.clip((frame[column].to_numpy(float) - median) / half, -1.0, 1.0)
            )
        values = np.column_stack(channels)
        targets = frame[target_column].to_numpy(float)
        dates = frame["date"].to_numpy()
        X, y, output_dates = [], [], []
        for end in range(lookback_days - 1, len(frame)):
            X.append(values[end - lookback_days + 1 : end + 1])
            y.append(targets[end])
            output_dates.append(dates[end])
        output[name] = (np.asarray(X, float), np.asarray(y, float), np.asarray(output_dates))
    return output


def raw_features(X: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for channel in range(X.shape[2]):
        window = X[:, :, channel]
        stats.append(
            np.column_stack(
                [window[:, -1], window.mean(1), window.std(1), window.min(1), window.max(1)]
            )
        )
    return np.column_stack([anchors] + stats)


def labels_for(
    y_train: np.ndarray,
    values: dict[str, np.ndarray],
    quantile: float,
) -> tuple[float, dict[str, np.ndarray]]:
    threshold = float(np.quantile(y_train, quantile))
    return threshold, {
        split: (values[split] >= threshold).astype(int) for split in SPLITS
    }


def finite_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    n_positive = int(labels.sum())
    if n_positive == 0 or n_positive == len(labels):
        return {"ap": np.nan, "auc": np.nan, "n_pos": n_positive}
    return {
        "ap": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
        "n_pos": n_positive,
    }


def select_f1_threshold(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    quantile: float,
) -> tuple[float, str]:
    if 0 < int(val_labels.sum()) < len(val_labels):
        candidates = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 199)))
        best_threshold = float(candidates[0])
        best_f1 = -1.0
        for threshold in candidates:
            prediction = (val_scores >= threshold).astype(int)
            value = f1_score(val_labels, prediction, zero_division=0)
            if value > best_f1:
                best_f1 = float(value)
                best_threshold = float(threshold)
        return best_threshold, "validation_f1"
    return float(np.quantile(train_scores, quantile)), "train_score_quantile_fallback"


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    base = finite_binary_metrics(labels, scores)
    if not np.isfinite(base["ap"]):
        return {
            **base,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "called_rate": np.nan,
        }
    prediction = (scores >= threshold).astype(int)
    return {
        **base,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "called_rate": float(prediction.mean()),
    }


def track_a_metrics(y_true: np.ndarray, log_scores: np.ndarray) -> dict[str, float]:
    metrics = evaluate_volatility_forecast(y_true, np.exp(log_scores))
    return {
        "rmse": float(metrics.rmse),
        "qlike": float(metrics.qlike),
        "mz_alpha": float(metrics.mz_alpha),
        "mz_beta": float(metrics.mz_beta),
        "mz_r2": float(metrics.mz_r2),
    }


def fit_log_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features["train"], np.log(np.clip(targets["train"], 1e-12, None)))
    return {split: model.predict(features[split]) for split in SPLITS}


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    return _spectral_scale(W, radius)


def make_esn_weights(
    n_inputs: int,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _make_esn_weights(
        n_inputs,
        n_reservoir,
        spectral_radius,
        input_scale,
        seed,
    )


def esn_states(
    X: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    leak: float,
) -> np.ndarray:
    return _esn_states(X, W_in, W, leak)


def make_sequence_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame[feature_columns].to_numpy(float)
    targets = frame[TARGET].to_numpy(float)
    dates = frame["date"].to_numpy()
    X, y, output_dates = [], [], []
    for end in range(lookback - 1, len(frame)):
        X.append(values[end - lookback + 1 : end + 1])
        y.append(targets[end])
        output_dates.append(dates[end])
    return np.asarray(X), np.asarray(y), np.asarray(output_dates)


def esn_grid(seeds: list[int]) -> list[dict]:
    return [
        {key: value for key, value in config.items() if key != "config_id"}
        for config in _historical_numpy_esn_grid(seeds)
    ]


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
    row.update(
        {f"test_{key}": value for key, value in track_a_metrics(y["test"], scores["test"]).items()}
    )
    row.update(
        {f"val_{key}": value for key, value in track_a_metrics(y["val"], scores["val"]).items()}
    )
    row["q90_target_threshold"] = q90_threshold
    row["q95_target_threshold"] = q95_threshold

    for q, labels in ((90, lab90), (95, lab95)):
        score_threshold, source = select_f1_threshold(
            labels["train"],
            labels["val"],
            scores["train"],
            scores["val"],
            q / 100.0,
        )
        row[f"q{q}_score_threshold"] = score_threshold
        row[f"q{q}_threshold_source"] = source
        for split in ("val", "test"):
            values = classification_metrics(labels[split], scores[split], score_threshold)
            for key, value in values.items():
                row[f"q{q}_{split}_{key}"] = value

    prediction_rows = []
    for split in SPLITS:
        y_pred = np.exp(scores[split])
        for index in range(len(y[split])):
            prediction_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "protocol": protocol,
                    "split": split,
                    "date": dates[split][index],
                    "y_true": float(y[split][index]),
                    "log_score": float(scores[split][index]),
                    "y_pred": float(y_pred[index]),
                    "q90_label": int(lab90[split][index]),
                    "q95_label": int(lab95[split][index]),
                }
            )
    return row, prediction_rows


def aggregate_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in per_fold.columns
        if column.startswith("test_")
        or column.startswith("q90_test_")
        or column.startswith("q95_test_")
    ]
    rows = []
    for (model, protocol), group in per_fold.groupby(["model", "protocol"], dropna=False):
        base = {
            "model": model,
            "protocol": protocol,
            "n_folds_regression": int(group["test_rmse"].notna().sum()),
            "n_folds_q90_valid": int(group["q90_test_ap"].notna().sum()),
            "n_folds_q95_valid": int(group["q95_test_ap"].notna().sum()),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            base[f"{column}_median"] = float(values.median()) if values.notna().any() else np.nan
            base[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            base[f"{column}_std"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    unknown = sorted(set(args.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.train_stride < 1:
        raise ValueError("train_stride must be >= 1")

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = {
        "date",
        TARGET,
        "rv_20d",
        "rv_5d",
        "rv_60d",
        "vix_close",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    folds = make_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        requested = set(args.only_folds)
        folds = [fold for fold in folds if fold["fold"] in requested]
        if not folds:
            raise ValueError(f"No matching folds for {sorted(requested)}")

    esn_seeds = [int(value) for value in args.esn_seeds.split(",") if value.strip()]
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"\n=== Canonical fold {fold_id} ===")
        split_frames = {
            split: frame.iloc[fold[split][0] : fold[split][1]].copy().reset_index(drop=True)
            for split in SPLITS
        }
        aligned = aligned_frames(frame, fold, args.lookback)
        y = {split: aligned[split][TARGET].to_numpy(float) for split in SPLITS}
        dates = {split: aligned[split]["date"].to_numpy() for split in SPLITS}
        q90_target, lab90 = labels_for(y["train"], y, 0.90)
        q95_target, lab95 = labels_for(y["train"], y, 0.95)

        if "persistence_20d" in args.models:
            scores = {
                split: np.log(
                    np.clip(aligned[split]["rv_20d"].to_numpy(float), 1e-12, None)
                )
                for split in SPLITS
            }
            row, predictions = evaluate_model(
                model_name="persistence_20d",
                protocol="classical",
                fold_id=fold_id,
                y=y,
                dates=dates,
                scores=scores,
                q90_threshold=q90_target,
                lab90=lab90,
                q95_threshold=q95_target,
                lab95=lab95,
                metadata={
                    "quantum_tasks_per_date": 0,
                    "shots": np.nan,
                    "shot_seed": np.nan,
                    "fit_seconds": 0.0,
                    "feature_seconds": 0.0,
                },
            )
            metric_rows.append(row)
            prediction_rows.extend(predictions)

        if "har_ridge" in args.models:
            har_columns = ["rv_5d", "rv_20d", "rv_60d"]
            features = {
                split: aligned[split][har_columns].to_numpy(float) for split in SPLITS
            }
            start = time.perf_counter()
            scores = fit_log_ridge(features, y, args.har_alpha)
            fit_seconds = time.perf_counter() - start
            row, predictions = evaluate_model(
                model_name="har_ridge",
                protocol="classical",
                fold_id=fold_id,
                y=y,
                dates=dates,
                scores=scores,
                q90_threshold=q90_target,
                lab90=lab90,
                q95_threshold=q95_target,
                lab95=lab95,
                metadata={
                    "quantum_tasks_per_date": 0,
                    "shots": np.nan,
                    "shot_seed": np.nan,
                    "fit_seconds": fit_seconds,
                    "feature_seconds": 0.0,
                },
            )
            metric_rows.append(row)
            prediction_rows.extend(predictions)

        if "raw_ridge" in args.models:
            sequences = make_level_rate_sequence_splits(
                split_frames,
                level_col=args.level_col,
                rate_col=args.rate_col,
                target_column=TARGET,
                lookback_days=args.lookback,
            )
            if args.train_stride > 1:
                stride = args.train_stride
                sequences["train"] = tuple(
                    array[::stride] for array in sequences["train"]
                )
            sequence_y = {split: sequences[split][1] for split in SPLITS}
            anchor_indices = select_anchor_indices(args.lookback, args.anchors)
            raw = {
                split: raw_features(sequences[split][0], anchor_indices)
                for split in SPLITS
            }
            start = time.perf_counter()
            scores = fit_log_ridge(raw, sequence_y, args.raw_ridge_alpha)
            fit_seconds = time.perf_counter() - start
            q90_raw, labels90_raw = labels_for(sequence_y["train"], sequence_y, 0.90)
            q95_raw, labels95_raw = labels_for(sequence_y["train"], sequence_y, 0.95)
            row, predictions = evaluate_model(
                model_name="raw_ridge",
                protocol="classical",
                fold_id=fold_id,
                y=sequence_y,
                dates={split: sequences[split][2] for split in SPLITS},
                scores=scores,
                q90_threshold=q90_raw,
                lab90=labels90_raw,
                q95_threshold=q95_raw,
                lab95=labels95_raw,
                metadata={
                    "quantum_tasks_per_date": 0,
                    "shots": np.nan,
                    "shot_seed": np.nan,
                    "fit_seconds": fit_seconds,
                    "feature_seconds": 0.0,
                },
            )
            metric_rows.append(row)
            prediction_rows.extend(predictions)

        if "esn_selected" in args.models:
            scaler = StandardScaler()
            pca = PCA(n_components=args.pca_components, random_state=42)
            pca.fit(scaler.fit_transform(split_frames["train"][FEATURE_COLUMNS]))
            pca_columns = [f"pca{index + 1}" for index in range(args.pca_components)]
            esn_sequences = {}
            for split in SPLITS:
                transformed = pca.transform(
                    scaler.transform(split_frames[split][FEATURE_COLUMNS])
                )
                temporary = pd.DataFrame(transformed, columns=pca_columns)
                temporary[TARGET] = split_frames[split][TARGET].to_numpy()
                temporary["date"] = split_frames[split]["date"].to_numpy()
                esn_sequences[split] = make_sequence_arrays(
                    temporary, pca_columns, args.lookback
                )
            esn_y = {split: esn_sequences[split][1] for split in SPLITS}
            esn_dates = {split: esn_sequences[split][2] for split in SPLITS}
            q90_esn, esn_lab90 = labels_for(esn_y["train"], esn_y, 0.90)
            q95_esn, esn_lab95 = labels_for(esn_y["train"], esn_y, 0.95)
            candidates = []
            start = time.perf_counter()
            for config in esn_grid(esn_seeds):
                W_in, W = make_esn_weights(
                    args.pca_components,
                    config["n"],
                    config["sr"],
                    config["inp"],
                    config["seed"],
                )
                states = {
                    split: esn_states(
                        esn_sequences[split][0], W_in, W, config["leak"]
                    )
                    for split in SPLITS
                }
                scores = fit_log_ridge(states, esn_y, config["alpha"])
                q90_val = finite_binary_metrics(esn_lab90["val"], scores["val"])["ap"]
                candidates.append((q90_val, config, scores))
            finite = [candidate for candidate in candidates if np.isfinite(candidate[0])]
            selected = max(finite, key=lambda item: item[0]) if finite else candidates[0]
            feature_seconds = time.perf_counter() - start
            _, selected_config, selected_scores = selected
            row, predictions = evaluate_model(
                model_name="esn_selected",
                protocol="classical",
                fold_id=fold_id,
                y=esn_y,
                dates=esn_dates,
                scores=selected_scores,
                q90_threshold=q90_esn,
                lab90=esn_lab90,
                q95_threshold=q95_esn,
                lab95=esn_lab95,
                metadata={
                    "quantum_tasks_per_date": 0,
                    "shots": np.nan,
                    "shot_seed": selected_config["seed"],
                    "fit_seconds": np.nan,
                    "feature_seconds": feature_seconds,
                    "selected_config": json.dumps(selected_config, sort_keys=True),
                    "selection_metric": "q90_val_ap",
                },
            )
            metric_rows.append(row)
            prediction_rows.extend(predictions)

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = aggregate_metrics(per_fold)

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    per_fold.to_csv(per_fold_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "data": str(args.data),
                "models": args.models,
                "folds": [int(fold["fold"]) for fold in folds],
                "fold_policy": (
                    "all folds for regression; classification metrics NaN when "
                    "test labels are degenerate"
                ),
                "f1_policy": (
                    "validation-optimal threshold; training-score quantile fallback "
                    "when validation labels are degenerate"
                ),
                "scope": "classical Stage 1 models only",
                "args": vars(args),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== Canonical aggregate comparison ===")
    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
