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
    RematchedDataset,
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    load_rematched_dataset,
)
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

DEFAULT_SELECTION_FOLDS = (4, 5, 6)
DEFAULT_CONFIRMATION_FOLDS = (7, 8)
DEFAULT_SEQUENCE_ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
HAR_ALPHA = 100.0


def _fit_scaled_ridge(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    predict_scaled = scaler.transform(predict_x)
    model = Ridge(alpha=float(alpha))
    model.fit(train_scaled, train_y)
    return model.predict(predict_scaled)


def _fold_rows(dataset: RematchedDataset, fold: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mask = dataset.manifest["fold"].eq(fold).to_numpy()
    frame = dataset.manifest.loc[mask].reset_index(drop=True)
    sequences = dataset.sequences[mask]
    train_mask = frame["fold_split"].eq("train").to_numpy()
    val_mask = frame["fold_split"].eq("val").to_numpy()
    if frame["fold_split"].eq("test").any():
        keep = train_mask | val_mask
        frame = frame.loc[keep].reset_index(drop=True)
        sequences = sequences[keep]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has an empty train or validation partition")
    return frame, sequences, train_mask, val_mask


def _sequence_alpha_table(dataset: RematchedDataset, folds: tuple[int, ...], alphas: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in folds:
        frame, sequences, train_mask, val_mask = _fold_rows(dataset, fold)
        target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        flat = sequences[:, :, 0]
        for alpha in alphas:
            prediction = _fit_scaled_ridge(flat[train_mask], target[train_mask], flat[val_mask], alpha)
            y_val = target[val_mask]
            rows.append({
                "fold": int(fold),
                "alpha": float(alpha),
                "qlike": float(qlike_loss(y_val, prediction).mean()),
                "rmse": float(np.sqrt(np.mean((y_val - prediction) ** 2))),
                "n_samples": int(val_mask.sum()),
            })
    return pd.DataFrame(rows)


def select_sequence_alpha(dataset: RematchedDataset, *, folds: tuple[int, ...], alphas: tuple[float, ...]) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    by_fold = _sequence_alpha_table(dataset, folds, alphas)
    aggregate = (
        by_fold.groupby("alpha", as_index=False)
        .agg(mean_fold_qlike=("qlike", "mean"), std_fold_qlike=("qlike", "std"), mean_fold_rmse=("rmse", "mean"))
        .sort_values(["mean_fold_qlike", "mean_fold_rmse", "alpha"], ignore_index=True)
    )
    return float(aggregate.iloc[0]["alpha"]), by_fold, aggregate


def predict_fold(dataset: RematchedDataset, *, fold: int, sequence_alpha: float) -> pd.DataFrame:
    frame, sequences, train_mask, val_mask = _fold_rows(dataset, fold)
    target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_val = target[val_mask]
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    predictions: dict[str, np.ndarray] = {
        "persistence": np.repeat(sequences[val_mask, -1, 0][:, None], len(TARGET_COLUMNS), axis=1),
        "har": _fit_scaled_ridge(
            frame.loc[train_mask, list(HAR_FEATURES)].to_numpy(dtype=float),
            target[train_mask],
            frame.loc[val_mask, list(HAR_FEATURES)].to_numpy(dtype=float),
            HAR_ALPHA,
        ),
        "sequence_ridge": _fit_scaled_ridge(
            sequences[train_mask, :, 0], target[train_mask], sequences[val_mask, :, 0], sequence_alpha
        ),
    }
    rows: list[pd.DataFrame] = []
    metadata_columns = [
        "sample_id", "episode_id", "index", "market_group", "origin_date", "event_onset",
        "label", "lead", "control_stratum", "fold", "fold_split",
    ]
    for model, prediction in predictions.items():
        output = val_frame[metadata_columns].copy()
        output.insert(0, "model", model)
        output["sequence_alpha"] = sequence_alpha if model == "sequence_ridge" else np.nan
        output["har_alpha"] = HAR_ALPHA if model == "har" else np.nan
        for horizon in range(1, 11):
            output[f"actual_h{horizon}"] = y_val[:, horizon - 1]
            output[f"predicted_h{horizon}"] = prediction[:, horizon - 1]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def run_linear_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    run_id: str | None = None,
    selection_folds: tuple[int, ...] = DEFAULT_SELECTION_FOLDS,
    confirmation_folds: tuple[int, ...] = DEFAULT_CONFIRMATION_FOLDS,
    sequence_alphas: tuple[float, ...] = DEFAULT_SEQUENCE_ALPHAS,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_rematched_dataset(dataset_root)
    all_folds = tuple(selection_folds) + tuple(confirmation_folds)
    overlap = set(selection_folds).intersection(confirmation_folds)
    if overlap:
        raise ValueError(f"selection and confirmation folds overlap: {sorted(overlap)}")
    if any(fold not in set(dataset.manifest["fold"].astype(int)) for fold in all_folds):
        raise ValueError("requested fold is missing from dataset")
    params = {
        "benchmark": "linear",
        "dataset_root": dataset_root,
        "selection_folds": selection_folds,
        "confirmation_folds": confirmation_folds,
        "sequence_alphas": sequence_alphas,
        "har_alpha": HAR_ALPHA,
        "test_evaluated": False,
    }
    run_dir = begin_run(results_root, params, run_id=run_id)
    selected_alpha, tuning_by_fold, tuning_summary = select_sequence_alpha(
        dataset, folds=selection_folds, alphas=sequence_alphas
    )
    predictions = pd.concat(
        [predict_fold(dataset, fold=fold, sequence_alpha=selected_alpha) for fold in all_folds],
        ignore_index=True,
    )
    if set(predictions["fold_split"]) != {"val"}:
        raise AssertionError("benchmark predictions must contain validation rows only")
    stage_frames = {
        "selection": predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": predictions.reset_index(drop=True),
    }
    submission_metrics = pd.concat(
        [grouped_path_metrics(frame, stage=stage) for stage, frame in stage_frames.items()],
        ignore_index=True,
    )
    metrics_by_horizon = pd.concat(
        [grouped_horizon_metrics(frame, stage=stage) for stage, frame in stage_frames.items()],
        ignore_index=True,
    )
    metrics_by_fold = grouped_fold_metrics(predictions)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    submission_metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    metrics_by_fold.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    metrics_by_horizon.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    tuning_by_fold.to_csv(run_dir / "sequence_ridge_tuning_by_fold.csv", index=False)
    tuning_summary.to_csv(run_dir / "sequence_ridge_tuning_summary.csv", index=False)
    config = {
        **params,
        "selected_sequence_alpha": selected_alpha,
        "models": ["persistence", "har", "sequence_ridge"],
        "reporting_groups": ["Transition", "L1", "L5", "L10", "Controls", "Pooled"],
        "metrics": ["RMSE", "QLIKE", "MZ alpha", "MZ beta", "MZ R2"],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    dataset_manifest = {
        "manifest_path": str(dataset.manifest_path),
        "tensors_path": str(dataset.tensors_path),
        "manifest_sha256": dataset.manifest_sha256,
        "tensors_sha256": dataset.tensors_sha256,
        "source_rows": int(len(dataset.manifest)),
        "prediction_rows": int(len(predictions)),
        "test_rows_used": 0,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    runtime = {"wall_seconds": time.perf_counter() - started}
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    confirmation = submission_metrics[
        submission_metrics["stage"].eq("confirmation") & submission_metrics["group"].eq("Pooled")
    ].sort_values(["qlike", "rmse"])
    summary = {
        "status": "complete",
        "benchmark": "linear",
        "run_id": run_dir.name,
        "selected_sequence_alpha": selected_alpha,
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "test_evaluated": False,
        "confirmation_pooled_ranking": confirmation[["model", "qlike", "rmse", "mz_alpha", "mz_beta", "mz_r2"]].to_dict("records"),
        "artifacts": sorted(path.name for path in run_dir.iterdir()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary
