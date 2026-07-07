#!/usr/bin/env python3
"""Canonical Phase 3 GARCH baseline under the shared purged walk-forward protocol.

GARCH is a sequential econometric comparator rather than a fixed-window
reservoir. For each requested test origin, the runner fits GARCH(1,1)-t using
return history available through that origin and forecasts the next 20 daily
conditional variances. Their annualized aggregate is converted to the same
realized-volatility units as ``future_rv_20d``.

Model mechanics live in ``qpitome_qrc.baselines.garch``. This runner owns only
canonical data access, fold geometry, artifact writing, and reporting.
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

TARGET = "future_rv_20d"
RETURN_COLUMN = "spy_log_return"
ANNUALIZATION_PERIOD = 252.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("scratch/garch_walkforward"))
    parser.add_argument("--tag", default="garch_phase3")
    parser.add_argument("--return-column", default=RETURN_COLUMN)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--history",
        type=int,
        default=2500,
        help="Maximum return-history rows used at each forecast origin.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    return parser.parse_args()


def finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 3:
        return {
            "n_predictions": int(mask.sum()),
            "rmse": np.nan,
            "qlike": np.nan,
            "mz_alpha": np.nan,
            "mz_beta": np.nan,
            "mz_r2": np.nan,
        }

    metrics = evaluate_volatility_forecast(y_true[mask], y_pred[mask])
    return {"n_predictions": int(mask.sum()), **asdict(metrics)}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = ["date", args.return_column, args.target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model_frame = frame[required].replace([np.inf, -np.inf], np.nan)
    if model_frame.isna().any().any():
        counts = model_frame.isna().sum()
        bad = {column: int(count) for column, count in counts.items() if count}
        raise ValueError(
            "GARCH runner does not silently alter canonical rows; "
            f"non-finite required values: {bad}"
        )

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        selected = set(args.only_folds)
        folds = [fold for fold in folds if fold["fold"] in selected]

    config = GARCHConfig()
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    fold_manifests: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        test_start, test_end = fold["test"]
        fold_predictions: list[float] = []
        fold_targets: list[float] = []
        n_failed = 0

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
            y_pred = (
                variance_path_to_realized_volatility(
                    forecast.variance_path,
                    return_scale=config.return_scale,
                    annualization_period=ANNUALIZATION_PERIOD,
                )
                if forecast.converged
                else float("nan")
            )
            if not np.isfinite(y_pred):
                n_failed += 1

            y_true = float(frame.iloc[row_index][args.target])
            prediction_rows.append(
                {
                    "fold": fold_id,
                    "split": "test",
                    "row_index": row_index,
                    "date": frame.iloc[row_index]["date"],
                    "model": "garch_1_1_t",
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "converged": bool(forecast.converged),
                    "convergence_flag": forecast.convergence_flag,
                    "fit_note": forecast.note,
                    "history_start": history_start,
                    "history_rows": int(len(returns)),
                }
            )
            fold_predictions.append(y_pred)
            fold_targets.append(y_true)

        fold_metric = finite_metrics(
            np.asarray(fold_targets, dtype=float),
            np.asarray(fold_predictions, dtype=float),
        )
        metric_rows.append(
            {
                "fold": fold_id,
                "split": "test",
                "model": "garch_1_1_t",
                "n_failed": n_failed,
                **fold_metric,
            }
        )
        fold_manifests.append(
            {
                "fold": fold_id,
                "row_ranges": {
                    split: list(fold[split])
                    for split in ("train", "val", "purge", "test")
                },
                "test_predictions": int(len(fold_targets)),
                "failed_predictions": n_failed,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)

    metrics_path = args.out_dir / f"garch_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"garch_predictions_{args.tag}.csv"
    manifest_path = args.out_dir / f"garch_manifest_{args.tag}.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    manifest = {
        "tag": args.tag,
        "model": "garch_1_1_t",
        "model_family": "GARCH",
        "source_prototype": "reset-branch monthly GARCH implementation",
        "adaptation": (
            "Retains GARCH(1,1)-t mechanics, convergence checks, and analytic "
            "multi-step variance forecasts; removes monthly target, NYSE-month "
            "aggregation, and 245-fold reset protocol."
        ),
        "data": str(args.data),
        "return_column": args.return_column,
        "target": args.target,
        "target_definition": "sqrt(252 / 20 * sum(next 20 daily spy_log_return squared))",
        "horizon": args.horizon,
        "annualization_period": ANNUALIZATION_PERIOD,
        "history": args.history,
        "garch_config": config_to_dict(config),
        "forecast_policy": (
            "Sequential rolling-origin refit at every test date using returns "
            "available through that date."
        ),
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "selected_folds": [int(fold["fold"]) for fold in folds],
        },
        "folds": fold_manifests,
        "known_limitations": [
            "Sequential refitting is not the same training policy as fixed-readout reservoir models.",
            "No GJR/EGARCH leverage extension or hyperparameter search is performed.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
