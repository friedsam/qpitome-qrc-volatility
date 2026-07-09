"""Run the first cheap Phase 3 residual-front baselines.

Purpose
-------
Establish a versioned out-of-fold prediction artifact before any ESN/QRC
residual model is trained. This runner intentionally does only two things:

1. persistence / zero innovation;
2. a compact HAR-style Ridge on causal multiscale RV features.

Historical canonical runners are not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.targets import (
    DEFAULT_INNOVATION_TARGET,
    add_log_rv_innovation,
    reconstruct_future_rv,
)
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds


DEFAULT_EXTENDED_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_FROZEN_DATA = Path("data/processed/phase2_spy_vix_volatility.csv")
DEFAULT_OUTPUT = Path("results/baselines/residual_front_v1")

HAR_FEATURES = ["rv_5d", "rv_20d", "rv_60d"]
DEFAULT_ALPHAS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--min-test-size", type=int, default=100)
    return parser.parse_args()


def resolve_data_path(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(requested)
        return requested
    if DEFAULT_EXTENDED_DATA.exists():
        return DEFAULT_EXTENDED_DATA
    if DEFAULT_FROZEN_DATA.exists():
        return DEFAULT_FROZEN_DATA
    raise FileNotFoundError(
        f"Neither {DEFAULT_EXTENDED_DATA} nor {DEFAULT_FROZEN_DATA} exists"
    )


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.maximum(np.asarray(y_true, dtype=float), 1e-12)
    pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-12)
    ratio = truth / pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    model = LinearRegression().fit(pred, truth)
    return {
        "mz_intercept": float(model.intercept_),
        "mz_slope": float(model.coef_[0]),
        "mz_r2": float(model.score(pred, truth)),
    }


def score_predictions(
    *,
    y_innovation: np.ndarray,
    innovation_pred: np.ndarray,
    current_rv: np.ndarray,
    future_rv: np.ndarray,
) -> dict[str, float]:
    reconstructed = reconstruct_future_rv(current_rv, innovation_pred)
    scores = {
        "innovation_r2": float(r2_score(y_innovation, innovation_pred)),
        "rv_rmse": float(np.sqrt(mean_squared_error(future_rv, reconstructed))),
        "rv_qlike": qlike(future_rv, reconstructed),
    }
    scores.update(mz_calibration(future_rv, reconstructed))
    return scores


def fit_har_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, float, float]:
    best_alpha = None
    best_val_mse = np.inf

    for alpha in DEFAULT_ALPHAS:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        if val_mse < best_val_mse:
            best_alpha = alpha
            best_val_mse = float(val_mse)

    assert best_alpha is not None
    final_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha)),
        ]
    )
    final_model.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    return final_model, float(best_alpha), best_val_mse


def prepare_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = add_log_rv_innovation(df)

    missing = set(HAR_FEATURES + ["rv_20d", "future_rv_20d", "date"]) - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {sorted(missing)}")

    out = df.dropna(
        subset=HAR_FEATURES
        + [DEFAULT_INNOVATION_TARGET, "rv_20d", "future_rv_20d"]
    ).copy()
    out = out.sort_values("date").reset_index(drop=True)

    for col in HAR_FEATURES:
        out[f"log_{col}"] = np.log(np.maximum(out[col].to_numpy(dtype=float), 1e-12))
    return out


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(data_path)
    folds = make_purged_walkforward_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
        min_test_size=args.min_test_size,
    )

    feature_cols = [f"log_{col}" for col in HAR_FEATURES]
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        train_start, train_end = fold["train"]
        val_start, val_end = fold["val"]
        test_start, test_end = fold["test"]

        train = df.iloc[train_start:train_end]
        val = df.iloc[val_start:val_end]
        test = df.iloc[test_start:test_end]

        y_train = train[DEFAULT_INNOVATION_TARGET].to_numpy(dtype=float)
        y_val = val[DEFAULT_INNOVATION_TARGET].to_numpy(dtype=float)
        y_test = test[DEFAULT_INNOVATION_TARGET].to_numpy(dtype=float)
        current_test = test["rv_20d"].to_numpy(dtype=float)
        future_test = test["future_rv_20d"].to_numpy(dtype=float)

        predictions: dict[str, tuple[np.ndarray, float | None, float | None]] = {
            "persistence": (np.zeros(len(test)), None, None)
        }

        har_model, alpha, val_mse = fit_har_ridge(
            train[feature_cols].to_numpy(dtype=float),
            y_train,
            val[feature_cols].to_numpy(dtype=float),
            y_val,
        )
        har_pred = har_model.predict(test[feature_cols].to_numpy(dtype=float))
        predictions["har_ridge"] = (har_pred, alpha, val_mse)

        for model_name, (innovation_pred, selected_alpha, selected_val_mse) in predictions.items():
            reconstructed = reconstruct_future_rv(current_test, innovation_pred)
            scores = score_predictions(
                y_innovation=y_test,
                innovation_pred=innovation_pred,
                current_rv=current_test,
                future_rv=future_test,
            )
            metric_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "n_test": len(test),
                    "selected_alpha": selected_alpha,
                    "selected_val_mse": selected_val_mse,
                    **scores,
                }
            )

            artifact = pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "fold": fold_id,
                    "model": model_name,
                    "y_true_innovation": y_test,
                    "oof_innovation_prediction": innovation_pred,
                    "residual_innovation": y_test - innovation_pred,
                    "current_rv_20d": current_test,
                    "future_rv_20d": future_test,
                    "reconstructed_future_rv_20d": reconstructed,
                }
            )
            prediction_rows.append(artifact)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            median_innovation_r2=("innovation_r2", "median"),
            mean_rv_rmse=("rv_rmse", "mean"),
            mean_rv_qlike=("rv_qlike", "mean"),
            mean_mz_intercept=("mz_intercept", "mean"),
            mean_mz_slope=("mz_slope", "mean"),
            mean_mz_r2=("mz_r2", "mean"),
        )
        .sort_values("mean_rv_rmse")
    )

    predictions.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    manifest = {
        "data_path": str(data_path),
        "n_rows": len(df),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "target": DEFAULT_INNOVATION_TARGET,
        "target_definition": "log(future_rv_20d / rv_20d)",
        "har_features": feature_cols,
        "alpha_grid": DEFAULT_ALPHAS,
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "min_test_size": args.min_test_size,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Data: {data_path}")
    print(f"Output: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
