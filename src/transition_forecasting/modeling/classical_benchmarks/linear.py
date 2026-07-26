from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.classical_benchmarks.common import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    RematchedDataset,
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    load_rematched_dataset,
)

DEFAULT_SELECTION_FOLDS = (4, 5, 6)
DEFAULT_CONFIRMATION_FOLDS = (7, 8)


def _fit_scaled_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    model = Ridge(alpha=float(alpha))
    model.fit(scaler.fit_transform(train_x), train_y)
    return model.predict(scaler.transform(predict_x))


def _fold_rows(
    dataset: RematchedDataset,
    fold: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mask = dataset.manifest["fold"].eq(fold).to_numpy()
    frame = dataset.manifest.loc[mask].reset_index(drop=True)
    sequences = dataset.sequences[mask]
    keep = frame["fold_split"].isin(["train", "val"]).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    sequences = sequences[keep]
    train_mask = frame["fold_split"].eq("train").to_numpy()
    val_mask = frame["fold_split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has an empty train or validation partition")
    return frame, sequences, train_mask, val_mask


def predict_fold(
    dataset: RematchedDataset,
    *,
    fold: int,
    sequence_alpha: float,
    har_alpha: float,
) -> pd.DataFrame:
    frame, sequences, train_mask, val_mask = _fold_rows(dataset, fold)
    target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_val = target[val_mask]
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    predictions = {
        "persistence": np.repeat(
            sequences[val_mask, -1, 0][:, None],
            len(TARGET_COLUMNS),
            axis=1,
        ),
        "har": _fit_scaled_ridge(
            frame.loc[train_mask, list(HAR_FEATURES)].to_numpy(dtype=float),
            target[train_mask],
            frame.loc[val_mask, list(HAR_FEATURES)].to_numpy(dtype=float),
            har_alpha,
        ),
        "sequence_ridge": _fit_scaled_ridge(
            sequences[train_mask, :, 0],
            target[train_mask],
            sequences[val_mask, :, 0],
            sequence_alpha,
        ),
    }
    metadata = [
        "sample_id", "episode_id", "index", "market_group", "origin_date",
        "event_onset", "label", "lead", "control_stratum", "fold", "fold_split",
    ]
    rows = []
    for model, prediction in predictions.items():
        output = val_frame[metadata].copy()
        output.insert(0, "model", model)
        output["sequence_alpha"] = sequence_alpha if model == "sequence_ridge" else np.nan
        output["har_alpha"] = har_alpha if model == "har" else np.nan
        for horizon in range(1, 11):
            output[f"actual_h{horizon}"] = y_val[:, horizon - 1]
            output[f"predicted_h{horizon}"] = prediction[:, horizon - 1]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def run_linear_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    run_id: str,
    sequence_alpha: float = 3000.0,
    har_alpha: float = 100.0,
    selection_folds: tuple[int, ...] = DEFAULT_SELECTION_FOLDS,
    confirmation_folds: tuple[int, ...] = DEFAULT_CONFIRMATION_FOLDS,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_rematched_dataset(dataset_root)
    all_folds = tuple(selection_folds) + tuple(confirmation_folds)
    if set(selection_folds).intersection(confirmation_folds):
        raise ValueError("selection and confirmation folds overlap")
    params = {
        "benchmark": "linear",
        "dataset_root": dataset_root,
        "selection_folds": selection_folds,
        "confirmation_folds": confirmation_folds,
        "sequence_alpha": sequence_alpha,
        "har_alpha": har_alpha,
        "test_evaluated": False,
    }
    run_dir = begin_run(results_root, params, run_id=run_id)
    predictions = pd.concat(
        [
            predict_fold(
                dataset,
                fold=fold,
                sequence_alpha=sequence_alpha,
                har_alpha=har_alpha,
            )
            for fold in all_folds
        ],
        ignore_index=True,
    )
    if set(predictions["fold_split"]) != {"val"}:
        raise AssertionError("benchmark predictions must contain validation rows only")
    stages = {
        "selection": predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": predictions.reset_index(drop=True),
    }
    metrics = pd.concat(
        [grouped_path_metrics(frame, stage=stage) for stage, frame in stages.items()],
        ignore_index=True,
    )
    horizon_metrics = pd.concat(
        [grouped_horizon_metrics(frame, stage=stage) for stage, frame in stages.items()],
        ignore_index=True,
    )
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    grouped_fold_metrics(predictions).to_csv(run_dir / "metrics_by_fold.csv", index=False)
    horizon_metrics.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    config = {
        **params,
        "models": ["persistence", "har", "sequence_ridge"],
        "reporting_groups": ["Transition", "L1", "L5", "L10", "Controls", "Pooled"],
        "metrics": ["RMSE", "QLIKE", "MZ alpha", "MZ beta", "MZ R2"],
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "manifest_path": str(dataset.manifest_path),
                "tensors_path": str(dataset.tensors_path),
                "manifest_sha256": dataset.manifest_sha256,
                "tensors_sha256": dataset.tensors_sha256,
                "source_rows": int(len(dataset.manifest)),
                "prediction_rows": int(len(predictions)),
                "test_rows_used": 0,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (run_dir / "runtime.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - started}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "complete",
        "benchmark": "linear",
        "run_id": run_id,
        "sequence_alpha": sequence_alpha,
        "har_alpha": har_alpha,
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "test_evaluated": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, summary
