#!/usr/bin/env python3
"""Run the LSTM baseline on the canonical Phase 3 walk-forward geometry.

The runner uses the repository's shared fold construction and metrics. LSTM
mechanics live in ``qpitome_qrc.baselines.lstm``.

Use ``--feature-cols`` to pass the exact comma-separated feature set used by the
comparison model. Leaving it unset selects all numeric columns except the date,
target, and optional return column; that convenience mode is for diagnostics,
not headline comparisons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.lstm import (
    LSTMConfig,
    build_sequences,
    config_to_dict,
    fit_lstm,
    predict_lstm,
)
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)


DEFAULT_TARGET = "future_rv_20d"


def _load_table(path: Path, date_column: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("--data must be a .parquet or .csv file")

    if date_column not in frame:
        raise KeyError(f"Missing date column: {date_column}")
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values(date_column).reset_index(drop=True)


def _resolve_features(
    frame: pd.DataFrame,
    *,
    requested: str | None,
    date_column: str,
    target: str,
    return_column: str,
) -> tuple[list[str], str]:
    if requested:
        columns = [name.strip() for name in requested.split(",") if name.strip()]
        if not columns:
            raise ValueError("--feature-cols did not contain any column names")
        source = "explicit"
    else:
        excluded = {date_column, target, return_column}
        columns = [
            name
            for name in frame.select_dtypes(include=[np.number]).columns
            if name not in excluded
        ]
        source = "automatic_numeric_diagnostic"

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if not columns:
        raise ValueError("No feature columns selected")
    return columns, source


def _metric_row(
    fold_id: int,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = evaluate_volatility_forecast(y_true, y_pred)
    return {
        "fold": fold_id,
        "split": split,
        "n_predictions": int(len(y_true)),
        **asdict(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--return-column", default="log_return")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--min-test-size", type=int, default=100)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/baselines/lstm"),
    )
    args = parser.parse_args()

    frame = _load_table(args.data, args.date_column)
    if args.target not in frame:
        raise KeyError(f"Missing target column: {args.target}")

    feature_columns, feature_source = _resolve_features(
        frame,
        requested=args.feature_cols,
        date_column=args.date_column,
        target=args.target,
        return_column=args.return_column,
    )
    required_columns = feature_columns + [args.target]
    if frame[required_columns].isna().any().any():
        raise ValueError(
            "LSTM runner does not silently fill missing feature/target rows; "
            "prepare an explicit canonical benchmark table first"
        )

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
        min_test_size=args.min_test_size,
    )
    config = LSTMConfig(
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    training_rows: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        frames = slice_fold_frames(frame, fold)

        scaler = StandardScaler()
        train_features = scaler.fit_transform(
            frames["train"][feature_columns].to_numpy(dtype=float)
        )
        train_sequences, train_targets = build_sequences(
            train_features,
            frames["train"][args.target].to_numpy(dtype=float),
            args.lookback,
        )
        fit = fit_lstm(train_sequences, train_targets, config=config)
        training_rows.append(
            {
                "fold": fold_id,
                "n_sequences": fit.n_sequences,
                "final_train_mse": fit.final_train_mse,
            }
        )

        for split in ("val", "test"):
            split_features = scaler.transform(
                frames[split][feature_columns].to_numpy(dtype=float)
            )
            split_sequences, split_targets = build_sequences(
                split_features,
                frames[split][args.target].to_numpy(dtype=float),
                args.lookback,
            )
            predictions = predict_lstm(fit.model, split_sequences)
            metric_rows.append(
                _metric_row(fold_id, split, split_targets, predictions)
            )

            dates = frames[split][args.date_column].iloc[
                args.lookback - 1 :
            ].reset_index(drop=True)
            for date, target, prediction in zip(
                dates,
                split_targets,
                predictions,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "fold": fold_id,
                        "split": split,
                        "date": date,
                        "target": args.target,
                        "y_true": float(target),
                        "y_pred": float(prediction),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    training = pd.DataFrame(training_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_fold.csv", index=False)
    training.to_csv(args.outdir / "training_diagnostics.csv", index=False)

    test_rows = predictions.loc[predictions["split"].eq("test")]
    aggregate = _metric_row(
        0,
        "test_aggregate",
        test_rows["y_true"].to_numpy(dtype=float),
        test_rows["y_pred"].to_numpy(dtype=float),
    )
    aggregate.pop("fold", None)

    manifest = {
        "model": "lstm",
        "model_family": "LSTM",
        "source_prototype": "reset-branch monthly LSTM implementation",
        "adaptation": (
            "Sequence/model mechanics retained; monthly feature catalog and "
            "245-fold protocol removed. Uses canonical purged walk-forward folds."
        ),
        "data": str(args.data),
        "date_column": args.date_column,
        "target": args.target,
        "features": feature_columns,
        "feature_selection": feature_source,
        "lookback": args.lookback,
        "lstm_config": config_to_dict(config),
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "min_test_size": args.min_test_size,
        },
        "aggregate_test_metrics": aggregate,
        "known_limitations": [
            "Headline comparisons must pass an explicit feature set shared with the comparator.",
            "The fixed architecture is a baseline, not a tuned neural-network search.",
            "Seed stability must be tested before making comparative claims.",
        ],
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"Outputs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
