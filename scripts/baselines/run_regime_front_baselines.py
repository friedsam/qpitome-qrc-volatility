"""Run the first primitive next-regime forecasting controls.

This runner answers the immediate Phase 3 question:

    Which broad regimes and transitions are already covered by cheap classical
    information, and where do they fail?

It does not tune ESN/QRC and does not modify historical canonical runners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds
from qpitome_qrc.regimes.broad_state_map import (
    REGIME_ORDER,
    BroadRegimeConfig,
    add_future_regime_target,
    assign_broad_regimes,
)


DEFAULT_EXTENDED_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_FROZEN_DATA = Path("data/processed/phase2_spy_vix_volatility.csv")
DEFAULT_OUTPUT = Path("results/baselines/regime_front_v1")

HAR_STATE_FEATURES = ["log_rv_5d", "log_rv_20d", "log_rv_60d"]
STATE_FEATURES = [
    "log_rv_5d",
    "log_rv_20d",
    "log_rv_60d",
    "rv_ratio_5_20_map",
    "rv_ratio_20_60_map",
    "rv_5d_change_5d",
    "rv_20d_change_5d",
    "return_5d",
    "return_20d",
    "drawdown_120d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=20)
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


def prepare_frame(path: Path, horizon: int) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["date"])
    mapped = assign_broad_regimes(raw, BroadRegimeConfig())
    mapped = add_future_regime_target(mapped, horizon=horizon)

    for col in ["rv_5d", "rv_20d", "rv_60d"]:
        mapped[f"log_{col}"] = np.log(np.maximum(mapped[col], 1e-12))

    required = STATE_FEATURES + ["broad_regime", "future_broad_regime"]
    return (
        mapped.dropna(subset=required)
        .sort_values("date")
        .reset_index(drop=True)
    )


def make_logit(feature_cols: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), feature_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prep", prep),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def probability_frame(
    *,
    model_name: str,
    dates: pd.Series,
    fold_id: int,
    current_regime: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "fold": fold_id,
            "model": model_name,
            "current_regime": current_regime.astype("string").to_numpy(),
            "future_regime": y_true,
            "predicted_regime": y_pred,
            "correct": y_true == y_pred,
        }
    )
    for regime in REGIME_ORDER:
        out[f"p_{regime}"] = 0.0
    for column, class_name in enumerate(classes):
        out[f"p_{class_name}"] = probabilities[:, column]
    return out


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    # Expand probabilities to the global class order for comparable log loss.
    expanded = np.full((len(y_true), len(REGIME_ORDER)), 1e-15, dtype=float)
    for column, class_name in enumerate(classes):
        expanded[:, REGIME_ORDER.index(str(class_name))] = probabilities[:, column]
    expanded /= expanded.sum(axis=1, keepdims=True)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=REGIME_ORDER, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, expanded, labels=REGIME_ORDER)),
    }


def persistence_predictions(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = test["broad_regime"].astype("string").to_numpy()
    probabilities = np.zeros((len(test), len(REGIME_ORDER)), dtype=float)
    for i, label in enumerate(pred):
        probabilities[i, REGIME_ORDER.index(label)] = 1.0
    return pred, probabilities, np.asarray(REGIME_ORDER)


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(data_path, args.horizon)
    folds = make_purged_walkforward_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
        min_test_size=args.min_test_size,
    )

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    confusion_rows: list[dict[str, int | str]] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        train_start, train_end = fold["train"]
        val_start, val_end = fold["val"]
        test_start, test_end = fold["test"]

        # Primitive front: use all available pre-purge labeled history for fitting.
        fit = df.iloc[train_start:val_end]
        test = df.iloc[test_start:test_end]
        y_fit = fit["future_broad_regime"].astype("string").to_numpy()
        y_test = test["future_broad_regime"].astype("string").to_numpy()

        model_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        model_outputs["persistence"] = persistence_predictions(test)

        for name, features in [
            ("har_state_logit", HAR_STATE_FEATURES),
            ("state_logit", STATE_FEATURES),
        ]:
            model = make_logit(features)
            model.fit(fit[features], y_fit)
            pred = model.predict(test[features])
            probabilities = model.predict_proba(test[features])
            classes = model.named_steps["model"].classes_
            model_outputs[name] = (pred, probabilities, classes)

        for model_name, (pred, probabilities, classes) in model_outputs.items():
            scores = score(y_test, pred, probabilities, classes)
            metric_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "n_test": len(test),
                    **scores,
                }
            )
            prediction_rows.append(
                probability_frame(
                    model_name=model_name,
                    dates=test["date"],
                    fold_id=fold_id,
                    current_regime=test["broad_regime"],
                    y_true=y_test,
                    y_pred=pred,
                    probabilities=probabilities,
                    classes=classes,
                )
            )

            matrix = confusion_matrix(y_test, pred, labels=REGIME_ORDER)
            for i, truth in enumerate(REGIME_ORDER):
                for j, predicted in enumerate(REGIME_ORDER):
                    confusion_rows.append(
                        {
                            "fold": fold_id,
                            "model": model_name,
                            "true_regime": truth,
                            "predicted_regime": predicted,
                            "count": int(matrix[i, j]),
                        }
                    )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    confusions = pd.DataFrame(confusion_rows)

    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
        .sort_values("mean_log_loss")
    )

    regime_counts = (
        df.groupby("broad_regime", observed=False)
        .size()
        .rename("days")
        .reset_index()
    )
    transitions = pd.crosstab(
        df["broad_regime"],
        df["future_broad_regime"],
        dropna=False,
    ).reindex(index=REGIME_ORDER, columns=REGIME_ORDER, fill_value=0)

    predictions.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    confusions.to_csv(output / "confusion_counts.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    regime_counts.to_csv(output / "regime_counts.csv", index=False)
    transitions.to_csv(output / "transition_counts.csv")

    manifest = {
        "data_path": str(data_path),
        "n_rows": len(df),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "target": f"broad regime {args.horizon} trading rows ahead",
        "regime_order": list(REGIME_ORDER),
        "models": {
            "persistence": "future regime equals current regime",
            "har_state_logit": HAR_STATE_FEATURES,
            "state_logit": STATE_FEATURES,
        },
        "regime_config": BroadRegimeConfig().__dict__,
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
    print(f"Prepared rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Target horizon: {args.horizon} trading rows")
    print("\nRegime counts:")
    print(regime_counts.to_string(index=False))
    print("\nPrimitive baseline summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
