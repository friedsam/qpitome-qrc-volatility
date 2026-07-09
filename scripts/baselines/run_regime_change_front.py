"""Forecast broad-regime change and localize unresolved transitions.

This is the second diagnostic front after six-class next-regime forecasting.
It asks a narrower question:

    Will the broad regime 20 trading rows ahead differ from the current regime?

The runner also reports exact current->future transition difficulty so that the
next branch-resolution target is selected from observed failure structure rather
than imposed in advance.
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
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds
from qpitome_qrc.regimes.broad_state_map import (
    BroadRegimeConfig,
    add_future_regime_target,
    assign_broad_regimes,
)


DEFAULT_EXTENDED_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_FROZEN_DATA = Path("data/processed/phase2_spy_vix_volatility.csv")
DEFAULT_OUTPUT = Path("results/baselines/regime_change_front_v1")

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
    mapped = (
        mapped.dropna(subset=required)
        .sort_values("date")
        .reset_index(drop=True)
    )
    mapped["regime_changed"] = (
        mapped["broad_regime"].astype("string")
        != mapped["future_broad_regime"].astype("string")
    ).astype(int)
    mapped["transition"] = (
        mapped["broad_regime"].astype("string")
        + " -> "
        + mapped["future_broad_regime"].astype("string")
    )
    return mapped


def make_binary_logit(feature_cols: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
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


def binary_scores(y_true: np.ndarray, p_change: np.ndarray) -> dict[str, float]:
    pred = (p_change >= 0.5).astype(int)
    scores = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - p_change, p_change]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_change)),
    }
    scores["roc_auc"] = (
        float(roc_auc_score(y_true, p_change)) if len(np.unique(y_true)) == 2 else np.nan
    )
    return scores


def transition_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OOF change probabilities by exact current->future transition."""
    rows = []
    for (model, transition), group in predictions.groupby(["model", "transition"]):
        y = group["regime_changed"].to_numpy(dtype=int)
        p = group["p_change"].to_numpy(dtype=float)
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "model": model,
                "transition": transition,
                "n": len(group),
                "true_change_rate": float(y.mean()),
                "mean_p_change": float(p.mean()),
                "accuracy": float((pred == y).mean()),
                "mean_abs_error": float(np.mean(np.abs(y - p))),
                "mean_brier": float(np.mean((y - p) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model", "mean_brier", "n"], ascending=[True, False, False]
    )


def current_state_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, current_regime), group in predictions.groupby(["model", "current_regime"]):
        y = group["regime_changed"].to_numpy(dtype=int)
        p = group["p_change"].to_numpy(dtype=float)
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "model": model,
                "current_regime": current_regime,
                "n": len(group),
                "change_rate": float(y.mean()),
                "mean_p_change": float(p.mean()),
                "accuracy": float((pred == y).mean()),
                "mean_brier": float(np.mean((y - p) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "mean_brier"], ascending=[True, False])


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

    for fold in folds:
        fold_id = int(fold["fold"])
        train_start, _ = fold["train"]
        _, val_end = fold["val"]
        test_start, test_end = fold["test"]

        fit = df.iloc[train_start:val_end]
        test = df.iloc[test_start:test_end]
        y_fit = fit["regime_changed"].to_numpy(dtype=int)
        y_test = test["regime_changed"].to_numpy(dtype=int)

        # No-change persistence baseline. Use a tiny epsilon so probabilistic
        # metrics remain finite while preserving the deterministic decision.
        outputs: dict[str, np.ndarray] = {
            "no_change": np.full(len(test), 1e-6, dtype=float)
        }

        for name, features in [
            ("har_change_logit", HAR_STATE_FEATURES),
            ("state_change_logit", STATE_FEATURES),
        ]:
            model = make_binary_logit(features)
            model.fit(fit[features], y_fit)
            outputs[name] = model.predict_proba(test[features])[:, 1]

        for model_name, p_change in outputs.items():
            metric_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "n_test": len(test),
                    "test_change_rate": float(y_test.mean()),
                    **binary_scores(y_test, p_change),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "date": test["date"].to_numpy(),
                        "fold": fold_id,
                        "model": model_name,
                        "current_regime": test["broad_regime"].astype("string").to_numpy(),
                        "future_regime": test["future_broad_regime"].astype("string").to_numpy(),
                        "transition": test["transition"].to_numpy(),
                        "regime_changed": y_test,
                        "p_change": p_change,
                        "predicted_change": (p_change >= 0.5).astype(int),
                    }
                )
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_change_rate=("test_change_rate", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values("mean_brier")
    )

    transition_summary = transition_diagnostics(predictions)
    state_summary = current_state_diagnostics(predictions)

    transition_counts = (
        df.groupby(["broad_regime", "future_broad_regime"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    predictions.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    transition_summary.to_csv(output / "transition_difficulty.csv", index=False)
    state_summary.to_csv(output / "current_state_difficulty.csv", index=False)
    transition_counts.to_csv(output / "transition_counts.csv", index=False)

    manifest = {
        "data_path": str(data_path),
        "n_rows": len(df),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "target": f"broad regime changes within {args.horizon} trading rows",
        "models": {
            "no_change": "predict no regime change",
            "har_change_logit": HAR_STATE_FEATURES,
            "state_change_logit": STATE_FEATURES,
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
    print(f"Overall change rate: {df['regime_changed'].mean():.4f}")
    print("\nRegime-change baseline summary:")
    print(summary.to_string(index=False))
    print("\nHardest exact transitions by OOF Brier error (state model):")
    hardest = transition_summary[
        transition_summary["model"] == "state_change_logit"
    ].head(15)
    print(hardest.to_string(index=False))
    print("\nCurrent-state difficulty (state model):")
    current = state_summary[
        state_summary["model"] == "state_change_logit"
    ]
    print(current.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
