#!/usr/bin/env python3
"""PyTorch confirmation study for the Phase 3 LSTM repair.

Runs two isolated variants on the canonical innovation target and fold protocol:

A. validation-based early stopping only;
B. early stopping plus train-only target standardization.

The canonical LSTM module and canonical runner are not modified. This script is
only for verifying whether the NumPy-replica diagnosis transfers to the actual
PyTorch implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qpitome_qrc.baselines.lstm import LSTMConfig, _make_model, _require_torch
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
VARIANTS = {
    "early_stop": False,
    "early_stop_standardized": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/lstm_repair_ablation_v1"),
    )
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--pca-components", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    return parser.parse_args()


def fit_with_early_stopping(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    *,
    config: LSTMConfig,
    standardize_target: bool,
    patience: int,
):
    torch = _require_torch()
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    target_mean = float(np.mean(train_y)) if standardize_target else 0.0
    target_std = float(np.std(train_y)) + 1e-8 if standardize_target else 1.0
    fit_y = ((train_y - target_mean) / target_std).astype(np.float32)

    train_X = np.asarray(train_X, dtype=np.float32)
    val_X = np.asarray(val_X, dtype=np.float32)
    val_y = np.asarray(val_y, dtype=float)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_X),
        torch.from_numpy(fit_y),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = _make_model(train_X.shape[2], config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    val_tensor = torch.from_numpy(val_X)

    best_val_mse = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0
    final_train_mse = float("nan")

    for epoch in range(config.epochs):
        model.train()
        weighted_loss = 0.0
        n_seen = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(len(X_batch))
            weighted_loss += float(loss.item()) * batch_size
            n_seen += batch_size
        final_train_mse = weighted_loss / n_seen

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).detach().cpu().numpy().astype(float)
        val_pred = val_pred * target_std + target_mean
        val_mse = float(np.mean((val_pred - val_y) ** 2))

        if val_mse < best_val_mse - 1e-9:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Early stopping failed to record a validation checkpoint")
    model.load_state_dict(best_state)

    return {
        "model": model,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch + 1),
        "best_val_mse": best_val_mse,
        "final_train_mse": final_train_mse,
    }


def predict_original_scale(model, X, *, target_mean: float, target_std: float):
    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    return raw.detach().cpu().numpy().astype(float) * target_std + target_mean


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    frame = add_rv_innovation_target(frame)
    target = RV_INNOVATION_TARGET

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        selected = set(args.only_folds)
        folds = [fold for fold in folds if int(fold["fold"]) in selected]
        if not folds:
            raise ValueError("No requested folds matched the canonical fold IDs")

    config = LSTMConfig(
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    metric_rows = []
    diagnostic_rows = []

    for fold in folds:
        fold_id = int(fold["fold"])
        split_frames = slice_fold_frames(frame, fold)
        clean = {
            split: drop_nonfinite_model_rows(
                split_frame,
                feature_columns=list(FEATURE_COLUMNS),
                target_columns=[target, RV_LEVEL_TARGET, RV_REFERENCE_COLUMN],
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

        sequences = {}
        targets = {}
        references = {}
        future_levels = {}
        for split in SPLITS:
            components = pca.transform(
                scaled[split][FEATURE_COLUMNS].to_numpy(dtype=float)
            )
            sequence_frame = pd.DataFrame(components, columns=pca_columns)
            sequence_frame["date"] = scaled[split]["date"].to_numpy()
            sequence_frame[target] = scaled[split][target].to_numpy(dtype=float)
            X, y, _ = make_sequence_arrays(
                sequence_frame,
                feature_columns=pca_columns,
                target_column=target,
                lookback=args.lookback,
            )
            sequences[split] = X.astype(np.float32)
            targets[split] = y.astype(float)
            aligned = clean[split].iloc[args.lookback - 1 :]
            references[split] = aligned[RV_REFERENCE_COLUMN].to_numpy(dtype=float)
            future_levels[split] = aligned[RV_LEVEL_TARGET].to_numpy(dtype=float)

        for variant, standardize_target in VARIANTS.items():
            fit = fit_with_early_stopping(
                sequences["train"],
                targets["train"],
                sequences["val"],
                targets["val"],
                config=config,
                standardize_target=standardize_target,
                patience=args.patience,
            )

            diagnostic_rows.append(
                {
                    "fold": fold_id,
                    "variant": variant,
                    "standardize_target": standardize_target,
                    "best_epoch": fit["best_epoch"],
                    "epochs_ran": fit["epochs_ran"],
                    "best_val_mse": fit["best_val_mse"],
                    "final_train_mse": fit["final_train_mse"],
                    "target_mean": fit["target_mean"],
                    "target_std": fit["target_std"],
                    "pca_explained_variance_sum": float(
                        pca.explained_variance_ratio_.sum()
                    ),
                }
            )

            for split in ("val", "test"):
                scores = predict_original_scale(
                    fit["model"],
                    sequences[split],
                    target_mean=fit["target_mean"],
                    target_std=fit["target_std"],
                )
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
                        "variant": variant,
                        "n_predictions": int(len(scores)),
                        **metrics,
                    }
                )

            test_row = metric_rows[-1]
            print(
                f"fold {fold_id} {variant:24s} "
                f"best_epoch={fit['best_epoch']:2d} "
                f"test_R2={test_row['innovation_r2']:+.4f} "
                f"test_RMSE={test_row['innovation_rmse']:.4f}"
            )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    diagnostics.to_csv(args.out_dir / "diagnostics.csv", index=False)

    test = metrics[metrics["split"] == "test"]
    summary = (
        test.groupby("variant")[["innovation_rmse", "innovation_r2", "reconstructed_rmse"]]
        .agg(["median", "mean", "std"])
    )
    summary.to_csv(args.out_dir / "summary.csv")

    manifest = {
        "purpose": "PyTorch confirmation of LSTM early-stopping repair",
        "target": target,
        "variants": VARIANTS,
        "canonical_lstm_config": config.__dict__,
        "patience": args.patience,
        "lookback": args.lookback,
        "pca_components": args.pca_components,
        "walkforward": {
            "n_folds": args.n_folds,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "purge": args.purge,
            "selected_folds": [int(fold["fold"]) for fold in folds],
        },
        "note": "No gradient clipping; canonical module and runner left unchanged.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print("\nMedian test metrics:")
    print(
        test.groupby("variant")[["innovation_rmse", "innovation_r2", "reconstructed_rmse"]]
        .median()
        .sort_values("innovation_r2", ascending=False)
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
