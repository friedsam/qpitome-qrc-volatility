#!/usr/bin/env python3
"""Fast baseline screen for the Phase 3 volatility-transition target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.targets import RV_INNOVATION_TARGET, add_rv_innovation_target
from qpitome_qrc.evaluation.transition import evaluate_transition_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds

HAR_RV_FEATURES = ["rv_5d", "rv_10d", "rv_20d", "rv_60d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    frame = add_rv_innovation_target(frame)

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=5,
        min_train=2500,
        val_size=504,
        purge=60,
    )

    models = [
        ("zero_change", [], None),
        (
            "HAR_RV_linear",
            HAR_RV_FEATURES,
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        ),
        (
            "full_linear",
            list(FEATURE_COLUMNS),
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        ),
        (
            "full_GBM",
            list(FEATURE_COLUMNS),
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
    ]

    rows: list[dict] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        train_start, train_end = fold["train"]
        test_start, test_end = fold["test"]
        train = frame.iloc[train_start:train_end]
        test = frame.iloc[test_start:test_end]

        for model_name, features, model in models:
            required = features + [RV_INNOVATION_TARGET]
            train_clean = train[required].replace([np.inf, -np.inf], np.nan).dropna()
            test_clean = test[required].replace([np.inf, -np.inf], np.nan).dropna()

            y_train = train_clean[RV_INNOVATION_TARGET].to_numpy(dtype=float)
            y_test = test_clean[RV_INNOVATION_TARGET].to_numpy(dtype=float)

            if model is None:
                prediction = np.zeros_like(y_test)
            else:
                model.fit(train_clean[features], y_train)
                prediction = model.predict(test_clean[features])

            rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "n_test": int(len(y_test)),
                    **evaluate_transition_forecast(y_test, prediction),
                }
            )

    result = pd.DataFrame(rows)
    print("\nPer-fold innovation R-squared:")
    print(
        result.pivot(
            index="fold",
            columns="model",
            values="innovation_r2",
        ).to_string()
    )
    print("\nMedian innovation R-squared:")
    print(
        result.groupby("model")["innovation_r2"]
        .median()
        .sort_values(ascending=False)
        .to_string()
    )
    print("\nMedian innovation RMSE:")
    print(
        result.groupby("model")["innovation_rmse"]
        .median()
        .sort_values()
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
