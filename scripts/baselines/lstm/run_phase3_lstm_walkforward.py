#!/usr/bin/env python3
"""Canonical Phase 3 LSTM baseline under the shared purged walk-forward protocol.

This runner deliberately matches the established ESN preprocessing path:

1. canonical Phase 2 feature frame;
2. split-local non-finite-row removal;
3. StandardScaler fit on training rows only;
4. PCA fit on scaled training rows only (six components by default);
5. split-local 40-row sequences;
6. LSTM fit to log future realized volatility;
7. positive-volatility forecasts recovered with ``exp`` for Track A metrics.

LSTM mechanics live in ``qpitome_qrc.baselines.lstm``. This file owns only
experiment geometry, preprocessing, artifact writing, and reporting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qpitome_qrc.baselines.lstm import (
    LSTMConfig,
    config_to_dict,
    fit_lstm,
    predict_lstm,
)
from qpitome_qrc.data.features import (
    FEATURE_COLUMNS,
    drop_nonfinite_model_rows,
    make_sequence_arrays,
    scale_splits_train_only,
)
from qpitome_qrc.data.targets import (
    RV_INNOVATION_TARGET,
    RV_LEVEL_TARGET,
    RV_REFERENCE_COLUMN,
    add_rv_innovation_target,
    reconstruct_future_rv,
)
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.transition import evaluate_transition_forecast
from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("scratch/lstm_walkforward"))
    parser.add_argument("--tag", default="lstm_phase3")
    parser.add_argument("--task", choices=("level", "innovation"), default="level")
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--pca-components", type=int, default=6)
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
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    if args.task == "innovation":
        frame = add_rv_innovation_target(frame)
        target = RV_INNOVATION_TARGET
    else:
        target = RV_LEVEL_TARGET

    required = ["date", target] + list(FEATURE_COLUMNS)
    if args.task == "innovation":
        required += [RV_LEVEL_TARGET, RV_REFERENCE_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        selected = set(args.only_folds)
        available = {int(fold["fold"]) for fold in folds}
        folds = [fold for fold in folds if int(fold["fold"]) in selected]
        if not folds:
            raise ValueError(
                f"Requested folds {sorted(selected)} do not match available fold IDs "
                f"{sorted(available)}"
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

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    fold_manifests: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        split_frames = slice_fold_frames(frame, fold)
        clean = {
            split: drop_nonfinite_model_rows(
                split_frame,
                feature_columns=list(FEATURE_COLUMNS),
                target_columns=(
                    [target]
                    if args.task == "level"
                    else [target, RV_LEVEL_TARGET, RV_REFERENCE_COLUMN]
                ),
            )
            for split, split_frame in split_frames.items()
        }
        scaled, _ = scale_splits_train_only(
            clean,
            feature_columns=list(FEATURE_COLUMNS),
            scaler_name="standard",
        )

        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        pca.fit(scaled["train"][FEATURE_COLUMNS].to_numpy(dtype=float))
        pca_columns = [f"pca{i + 1}" for i in range(args.pca_components)]

        sequences: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        dates: dict[str, pd.Series] = {}
        references: dict[str, np.ndarray] = {}
        future_levels: dict[str, np.ndarray] = {}
        for split in SPLITS:
            components = pca.transform(
                scaled[split][FEATURE_COLUMNS].to_numpy(dtype=float)
            )
            sequence_frame = pd.DataFrame(components, columns=pca_columns)
            sequence_frame["date"] = scaled[split]["date"].to_numpy()
            sequence_frame[target] = scaled[split][target].to_numpy(dtype=float)
            X, y, d = make_sequence_arrays(
                sequence_frame,
                feature_columns=pca_columns,
                target_column=target,
                lookback=args.lookback,
            )
            sequences[split] = X.astype(np.float32)
            targets[split] = y.astype(float)
            dates[split] = d
            if args.task == "innovation":
                aligned = scaled[split].iloc[args.lookback - 1 :]
                references[split] = aligned[RV_REFERENCE_COLUMN].to_numpy(dtype=float)
                future_levels[split] = aligned[RV_LEVEL_TARGET].to_numpy(dtype=float)

        train_target = targets["train"]
        if args.task == "level":
            train_target = np.log(np.clip(train_target, 1e-12, None))
        fit = fit_lstm(sequences["train"], train_target, config=config)
        diagnostic_rows.append(
            {
                "fold": fold_id,
                "train_sequences": fit.n_sequences,
                "final_train_mse": fit.final_train_mse,
                "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )

        fold_manifest = {
            "fold": fold_id,
            "row_ranges": {
                split: list(fold[split])
                for split in ("train", "val", "purge", "test")
            },
            "split_sequence_counts": {
                split: int(len(sequences[split])) for split in SPLITS
            },
            "pca_explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
        }
        fold_manifests.append(fold_manifest)

        for split in ("val", "test"):
            scores = predict_lstm(fit.model, sequences[split])

            if args.task == "level":
                forecasts = np.exp(scores)
                metrics = asdict(
                    evaluate_volatility_forecast(targets[split], forecasts)
                )
            else:
                forecasts = reconstruct_future_rv(references[split], scores)
                metrics = {
                    **evaluate_transition_forecast(targets[split], scores),
                    **{
                        f"reconstructed_{key}": value
                        for key, value in asdict(
                            evaluate_volatility_forecast(
                                future_levels[split], forecasts
                            )
                        ).items()
                    },
                }

            metric_rows.append(
                {
                    "fold": fold_id,
                    "split": split,
                    "model": "lstm",
                    "task": args.task,
                    "n_predictions": int(len(targets[split])),
                    **metrics,
                }
            )

            for date, y_true, score, forecast in zip(
                dates[split],
                targets[split],
                scores,
                forecasts,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "fold": fold_id,
                        "split": split,
                        "date": date,
                        "model": "lstm",
                        "task": args.task,
                        "y_true": float(y_true),
                        "score": float(score),
                        "y_pred": float(forecast),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    metrics_path = args.out_dir / f"lstm_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"lstm_predictions_{args.tag}.csv"
    diagnostics_path = args.out_dir / f"lstm_training_diagnostics_{args.tag}.csv"
    manifest_path = args.out_dir / f"lstm_manifest_{args.tag}.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    manifest = {
        "tag": args.tag,
        "model": "lstm",
        "model_family": "LSTM",
        "source_prototype": "reset-branch monthly LSTM implementation",
        "adaptation": (
            "Retains compact LSTM mechanics only; replaces monthly data, monthly "
            "feature catalog, and 245-fold protocol with canonical Phase 3 inputs."
        ),
        "data": str(args.data),
        "task": args.task,
        "target": target,
        "target_transform": "log" if args.task == "level" else "none",
        "forecast_transform": (
            "exp"
            if args.task == "level"
            else "rv_20d * exp(predicted_innovation)"
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "preprocessing": {
            "nonfinite_policy": "drop explicitly within each split",
            "scaler": "StandardScaler fit on train only",
            "pca_components": args.pca_components,
            "pca_fit": "train only",
            "sequence_construction": "split local",
            "lookback": args.lookback,
        },
        "lstm_config": config_to_dict(config),
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "selected_folds": [int(fold["fold"]) for fold in folds],
        },
        "folds": fold_manifests,
        "known_limitations": [
            "Architecture and optimizer settings are fixed rather than tuned.",
            "Single-seed default is insufficient for a stability claim.",
            "PCA6 is a matched preprocessing choice, not evidence that six components are optimal for LSTM.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {diagnostics_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
