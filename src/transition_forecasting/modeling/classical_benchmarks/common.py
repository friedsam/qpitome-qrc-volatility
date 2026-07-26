from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import mincer_zarnowitz
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS, qlike_loss

REQUIRED_GROUPS = ("Transition", "L1", "L5", "L10", "Controls", "Pooled")


@dataclass(frozen=True)
class RematchedDataset:
    manifest: pd.DataFrame
    sequences: np.ndarray
    manifest_path: Path
    tensors_path: Path
    manifest_sha256: str
    tensors_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rematched_dataset(dataset_root: Path) -> RematchedDataset:
    fold_root = Path(dataset_root) / "purged_walk_forward_folds"
    manifest_path = fold_root / "rematched_rolling_manifest.csv"
    tensors_path = fold_root / "rematched_rolling_tensors.npz"
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    with np.load(tensors_path, allow_pickle=False) as tensors:
        sequences = np.asarray(tensors["X"], dtype=float)
        tensor_ids = tensors["sample_id"].astype(str)
        tensor_folds = tensors["fold"].astype(int)
        tensor_splits = tensors["fold_split"].astype(str)
    required = {
        "sample_id", "episode_id", "label", "lead", "fold", "fold_split",
        "control_stratum", "market_group", *TARGET_COLUMNS,
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"rematched manifest missing columns: {sorted(missing)}")
    if sequences.shape != (len(manifest), 40, 1):
        raise ValueError(f"expected tensors shaped (n, 40, 1), got {sequences.shape}")
    if not np.array_equal(tensor_ids, manifest["sample_id"].astype(str).to_numpy()):
        raise ValueError("manifest and tensor sample IDs are not aligned")
    if not np.array_equal(tensor_folds, manifest["fold"].astype(int).to_numpy()):
        raise ValueError("manifest and tensor folds are not aligned")
    if not np.array_equal(tensor_splits, manifest["fold_split"].astype(str).to_numpy()):
        raise ValueError("manifest and tensor splits are not aligned")
    if set(manifest["label"].unique()) != {0, 1}:
        raise ValueError("expected binary label values 0 and 1")
    if set(manifest.loc[manifest["label"].eq(1), "lead"].astype(int).unique()) != {1, 5, 10}:
        raise ValueError("transition rows must contain leads 1, 5, and 10")
    if manifest.duplicated(["fold", "sample_id"]).any():
        raise ValueError("duplicate (fold, sample_id) rows in rematched manifest")
    return RematchedDataset(
        manifest=manifest,
        sequences=sequences,
        manifest_path=manifest_path,
        tensors_path=tensors_path,
        manifest_sha256=sha256_file(manifest_path),
        tensors_sha256=sha256_file(tensors_path),
    )


def group_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    label = frame["label"].to_numpy(dtype=int)
    lead = frame["lead"].to_numpy(dtype=int)
    masks = {
        "Transition": label == 1,
        "L1": (label == 1) & (lead == 1),
        "L5": (label == 1) & (lead == 5),
        "L10": (label == 1) & (lead == 10),
        "Controls": label == 0,
        "Pooled": np.ones(len(frame), dtype=bool),
    }
    return masks


def metric_row(
    *,
    model: str,
    group: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage: str | None = None,
    fold: int | None = None,
    horizon: int | str = "path",
) -> dict[str, object]:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    if observed.shape != forecast.shape:
        raise ValueError("observed and forecast shapes differ")
    try:
        mz = mincer_zarnowitz(observed.reshape(-1), forecast.reshape(-1))
    except ValueError:
        mz = {"alpha": np.nan, "beta": np.nan, "r2": np.nan}
    row: dict[str, object] = {
        "model": model,
        "group": group,
        "horizon": horizon,
        "n_samples": int(observed.shape[0]),
        "n_forecasts": int(observed.size),
        "rmse": float(np.sqrt(np.mean((observed - forecast) ** 2))),
        "qlike": float(qlike_loss(observed, forecast).mean()),
        "mz_alpha": mz["alpha"],
        "mz_beta": mz["beta"],
        "mz_r2": mz["r2"],
    }
    if stage is not None:
        row["stage"] = stage
    if fold is not None:
        row["fold"] = int(fold)
    return row


def grouped_path_metrics(predictions: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    actual_columns = [f"actual_h{h}" for h in range(1, 11)]
    predicted_columns = [f"predicted_h{h}" for h in range(1, 11)]
    rows: list[dict[str, object]] = []
    for model, model_frame in predictions.groupby("model", sort=False):
        masks = group_masks(model_frame)
        for group in REQUIRED_GROUPS:
            mask = masks[group]
            if not mask.any():
                continue
            rows.append(metric_row(
                model=str(model), group=group,
                y_true=model_frame.loc[mask, actual_columns].to_numpy(dtype=float),
                y_pred=model_frame.loc[mask, predicted_columns].to_numpy(dtype=float),
                stage=stage,
            ))
    return pd.DataFrame(rows)


def grouped_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_columns = [f"actual_h{h}" for h in range(1, 11)]
    predicted_columns = [f"predicted_h{h}" for h in range(1, 11)]
    rows: list[dict[str, object]] = []
    for (fold, model), model_frame in predictions.groupby(["fold", "model"], sort=True):
        masks = group_masks(model_frame)
        for group in REQUIRED_GROUPS:
            mask = masks[group]
            if not mask.any():
                continue
            rows.append(metric_row(
                model=str(model), group=group, fold=int(fold),
                y_true=model_frame.loc[mask, actual_columns].to_numpy(dtype=float),
                y_pred=model_frame.loc[mask, predicted_columns].to_numpy(dtype=float),
            ))
    return pd.DataFrame(rows)


def grouped_horizon_metrics(predictions: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_frame in predictions.groupby("model", sort=False):
        masks = group_masks(model_frame)
        for group in REQUIRED_GROUPS:
            mask = masks[group]
            if not mask.any():
                continue
            for horizon in range(1, 11):
                rows.append(metric_row(
                    model=str(model), group=group, stage=stage, horizon=horizon,
                    y_true=model_frame.loc[mask, [f"actual_h{horizon}"]].to_numpy(dtype=float),
                    y_pred=model_frame.loc[mask, [f"predicted_h{horizon}"]].to_numpy(dtype=float),
                ))
    return pd.DataFrame(rows)
