from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS, make_regression_arrays
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast


RESULTS_DIR = Path("results/tables")
OUTPUT_PATH = RESULTS_DIR / "phase2_classical_baselines.csv"

TARGET_COLUMNS = ["future_rv_5d", "future_rv_20d"]

PERSISTENCE_SPECS = [
    ("persistence_5d_to_5d", "future_rv_5d", "rv_5d"),
    ("persistence_10d_to_5d", "future_rv_5d", "rv_10d"),
    ("persistence_20d_to_20d", "future_rv_20d", "rv_20d"),
    ("persistence_60d_to_20d", "future_rv_20d", "rv_60d"),
]

HAR_FEATURES_5D = [
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "vix_close",
]

HAR_FEATURES_20D = [
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_60d",
    "vix_close",
]


def metrics_row(
    *,
    split: str,
    model: str,
    target: str,
    predictor: str,
    n: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = evaluate_volatility_forecast(y_true, y_pred)
    row = asdict(metrics)
    row.update(
        {
            "split": split,
            "model": model,
            "target": target,
            "predictor": predictor,
            "n": n,
        }
    )
    return row


def evaluate_persistence(splits: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split_name, split_df in splits.items():
        for model_name, target_col, predictor_col in PERSISTENCE_SPECS:
            rows.append(
                metrics_row(
                    split=split_name,
                    model=model_name,
                    target=target_col,
                    predictor=predictor_col,
                    n=len(split_df),
                    y_true=split_df[target_col].to_numpy(dtype=float),
                    y_pred=split_df[predictor_col].to_numpy(dtype=float),
                )
            )
    return rows


def fit_predict_tabular_model(
    splits: dict[str, pd.DataFrame],
    *,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
    estimator,
    scale_features: bool = True,
) -> list[dict]:
    train = splits["train"].dropna(subset=feature_cols + [target_col]).copy()
    val = splits["val"].dropna(subset=feature_cols + [target_col]).copy()
    test = splits["test"].dropna(subset=feature_cols + [target_col]).copy()

    if scale_features:
        model = make_pipeline(StandardScaler(), estimator)
    else:
        model = estimator

    model.fit(train[feature_cols].to_numpy(dtype=float), train[target_col].to_numpy(dtype=float))

    rows = []
    for split_name, split_df in {"train": train, "val": val, "test": test}.items():
        y_true = split_df[target_col].to_numpy(dtype=float)
        y_pred = model.predict(split_df[feature_cols].to_numpy(dtype=float))
        y_pred = np.maximum(y_pred, 1e-8)
        rows.append(
            metrics_row(
                split=split_name,
                model=model_name,
                target=target_col,
                predictor="model_forecast",
                n=len(split_df),
                y_true=y_true,
                y_pred=y_pred,
            )
        )
    return rows


def run_baselines() -> pd.DataFrame:
    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)

    rows: list[dict] = []
    rows.extend(evaluate_persistence(splits))

    for target_col in TARGET_COLUMNS:
        har_features = HAR_FEATURES_5D if target_col == "future_rv_5d" else HAR_FEATURES_20D

        rows.extend(
            fit_predict_tabular_model(
                splits,
                target_col=target_col,
                feature_cols=har_features,
                model_name="har_linear",
                estimator=LinearRegression(),
                scale_features=False,
            )
        )
        rows.extend(
            fit_predict_tabular_model(
                splits,
                target_col=target_col,
                feature_cols=har_features,
                model_name="har_ridge_alpha_1",
                estimator=Ridge(alpha=1.0),
                scale_features=True,
            )
        )

        # Full-feature linear baselines use the leakage-safe feature set defined for May 22.
        rows.extend(
            fit_predict_tabular_model(
                splits,
                target_col=target_col,
                feature_cols=FEATURE_COLUMNS,
                model_name="full_feature_ridge_alpha_1",
                estimator=Ridge(alpha=1.0),
                scale_features=True,
            )
        )
        rows.extend(
            fit_predict_tabular_model(
                splits,
                target_col=target_col,
                feature_cols=FEATURE_COLUMNS,
                model_name="full_feature_elasticnet_alpha_0.001_l1_0.1",
                estimator=ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000, random_state=42),
                scale_features=True,
            )
        )

    out = pd.DataFrame(rows)
    out = out[
        [
            "split",
            "model",
            "target",
            "predictor",
            "n",
            "rmse",
            "qlike",
            "mz_alpha",
            "mz_beta",
            "mz_r2",
        ]
    ].sort_values(["target", "split", "rmse", "model"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    table = run_baselines()
    print(f"Saved: {OUTPUT_PATH}")
    print(table.to_string(index=False))
