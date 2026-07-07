#!/usr/bin/env python3
"""Run the GARCH baseline on the canonical Phase 3 walk-forward geometry.

The runner is intentionally thin: GARCH mechanics live in
``qpitome_qrc.baselines.garch`` and fold construction / metrics come from the
shared evaluation package.

The input table must contain one row per chronological forecast origin with a
return column and the positive realized-volatility target attached to that row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.garch import (
    GARCHConfig,
    config_to_dict,
    fit_garch_variance_path,
    variance_path_to_realized_volatility,
)
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_walkforward_folds


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


def _metrics_row(fold_id: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 3:
        return {
            "fold": fold_id,
            "n_predictions": int(finite.sum()),
            "rmse": np.nan,
            "qlike": np.nan,
            "mz_alpha": np.nan,
            "mz_beta": np.nan,
            "mz_r2": np.nan,
        }

    metrics = evaluate_volatility_forecast(y_true[finite], y_pred[finite])
    return {
        "fold": fold_id,
        "n_predictions": int(finite.sum()),
        **asdict(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--return-column", default="log_return")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--min-test-size", type=int, default=100)
    parser.add_argument("--history", type=int, default=2500)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/baselines/garch"),
    )
    args = parser.parse_args()

    frame = _load_table(args.data, args.date_column)
    required = {args.return_column, args.target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if frame[[args.return_column, args.target]].isna().any().any():
        raise ValueError(
            "GARCH runner does not silently fill missing return/target rows; "
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
    config = GARCHConfig()

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        test_start, test_end = fold["test"]
        fold_predictions: list[float] = []
        fold_targets: list[float] = []

        for row_index in range(test_start, test_end):
            history_start = max(0, row_index - args.history + 1)
            returns = frame.iloc[history_start : row_index + 1][
                args.return_column
            ].to_numpy(dtype=float)

            forecast = fit_garch_variance_path(
                returns,
                horizon=args.horizon,
                config=config,
            )
            prediction = (
                variance_path_to_realized_volatility(
                    forecast.variance_path,
                    return_scale=config.return_scale,
                )
                if forecast.converged
                else float("nan")
            )
            target = float(frame.iloc[row_index][args.target])

            prediction_rows.append(
                {
                    "fold": fold_id,
                    "row_index": row_index,
                    "date": frame.iloc[row_index][args.date_column],
                    "target": args.target,
                    "y_true": target,
                    "y_pred": prediction,
                    "converged": forecast.converged,
                    "convergence_flag": forecast.convergence_flag,
                    "fit_note": forecast.note,
                    "history_rows": len(returns),
                }
            )
            fold_predictions.append(prediction)
            fold_targets.append(target)

        metric_rows.append(
            _metrics_row(
                fold_id,
                np.asarray(fold_targets),
                np.asarray(fold_predictions),
            )
        )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_fold.csv", index=False)

    finite = np.isfinite(predictions["y_pred"].to_numpy(dtype=float))
    aggregate = _metrics_row(
        0,
        predictions.loc[finite, "y_true"].to_numpy(dtype=float),
        predictions.loc[finite, "y_pred"].to_numpy(dtype=float),
    )
    aggregate.pop("fold", None)

    manifest = {
        "model": "garch",
        "model_family": "GARCH",
        "source_prototype": "reset-branch monthly GARCH implementation",
        "adaptation": (
            "Model mechanics retained; monthly dataset/protocol removed. "
            "Runs on the canonical daily future_rv_20d-style task."
        ),
        "data": str(args.data),
        "date_column": args.date_column,
        "return_column": args.return_column,
        "target": args.target,
        "horizon": args.horizon,
        "history": args.history,
        "garch_config": config_to_dict(config),
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "min_test_size": args.min_test_size,
        },
        "n_predictions": int(len(predictions)),
        "n_failed": int((~finite).sum()),
        "aggregate_metrics": aggregate,
        "known_limitations": [
            "The current runner refits GARCH at each test origin; this is a sequential rolling-origin baseline.",
            "The horizon-to-target mapping assumes target units sqrt(sum(future_return**2)).",
            "No leverage extension or hyperparameter search is performed.",
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
