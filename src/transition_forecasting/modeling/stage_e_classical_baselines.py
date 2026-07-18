from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HAR_FEATURES = ("level", "mean5", "mean20")
EXTENDED_HAR_FEATURES = (
    "level",
    "mean5",
    "mean20",
    "slope5",
    "slope20",
    "std20",
    "max20",
)
TARGET_COLUMNS = tuple(f"target_x_h{h}" for h in range(1, 11))
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class StageEData:
    manifest: pd.DataFrame
    sequences: np.ndarray


def load_stage_d_run(run_dir: Path) -> StageEData:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv")
    tensors = np.load(run_dir / "sequence_tensors.npz")
    sequences = np.asarray(tensors["X"], dtype=float)
    tensor_ids = tensors["sample_id"].astype(str)
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if sequences.ndim != 3 or sequences.shape[1:] != (40, 1):
        raise ValueError(f"expected sequence tensor shape (n, 40, 1), got {sequences.shape}")
    if len(manifest) != len(sequences) or not np.array_equal(manifest_ids, tensor_ids):
        raise ValueError("sample manifest and tensor sample IDs are not aligned")
    required = {"sample_id", "label", "lead", "episode_id", "split", *TARGET_COLUMNS}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Stage D manifest is missing columns: {sorted(missing)}")
    return StageEData(manifest=manifest, sequences=sequences)


def validate_split_integrity(manifest: pd.DataFrame) -> dict[str, object]:
    split_counts = manifest.groupby("episode_id")["split"].nunique()
    leaking = sorted(split_counts[split_counts > 1].index.astype(str))
    observed = set(manifest["split"].astype(str))
    missing_splits = sorted({"train", "val", "test"}.difference(observed))
    if leaking:
        raise ValueError(f"episodes span multiple splits: {leaking[:10]}")
    if missing_splits:
        raise ValueError(f"missing required splits: {missing_splits}")
    return {
        "episode_split_overlap": 0,
        "episodes_by_split": {
            split: int(group["episode_id"].nunique())
            for split, group in manifest.groupby("split")
        },
        "samples_by_split": {
            split: int(len(group)) for split, group in manifest.groupby("split")
        },
    }


def qlike_loss(y_true_logvol: np.ndarray, y_pred_logvol: np.ndarray) -> np.ndarray:
    true_logvar = np.clip(2.0 * np.asarray(y_true_logvol, dtype=float), -40.0, 20.0)
    pred_logvar = np.clip(2.0 * np.asarray(y_pred_logvol, dtype=float), -40.0, 20.0)
    ratio = np.exp(np.clip(true_logvar - pred_logvar, -40.0, 40.0))
    return ratio - np.log(ratio) - 1.0


def _independently_shuffle_rows(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.asarray(values, dtype=float).copy()
    for row in result:
        rng.shuffle(row)
    return result


def _feature_matrix(model_name: str, manifest: pd.DataFrame, sequences: np.ndarray, seed: int) -> np.ndarray:
    if model_name == "level_ridge":
        return manifest[["level"]].to_numpy(dtype=float)
    if model_name == "har_ridge":
        return manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    if model_name == "extended_har_ridge":
        return manifest[list(EXTENDED_HAR_FEATURES)].to_numpy(dtype=float)
    flattened = sequences[:, :, 0]
    if model_name == "sequence_ridge":
        return flattened
    if model_name == "shuffled_sequence_ridge":
        return _independently_shuffle_rows(flattened, seed)
    raise ValueError(f"unknown model: {model_name}")


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best_prediction: np.ndarray | None = None
    best_alpha: float | None = None
    best_qlike = np.inf
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_x, train_y)
        prediction = model.predict(val_x)
        mean_qlike = float(qlike_loss(val_y, prediction).mean())
        rmse = float(np.sqrt(np.mean((val_y - prediction) ** 2)))
        rows.append({"alpha": float(alpha), "mean_path_qlike": mean_qlike, "path_rmse": rmse})
        if mean_qlike < best_qlike:
            best_qlike = mean_qlike
            best_alpha = float(alpha)
            best_prediction = prediction
    if best_prediction is None or best_alpha is None:
        raise ValueError("alpha grid is empty")
    return best_prediction, best_alpha, pd.DataFrame(rows)


def _metric_rows(
    model_name: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    group_specs: list[tuple[str, str, np.ndarray]] = [("pooled", "all", np.ones(len(frame), dtype=bool))]
    for label, name in ((1, "positive"), (0, "control")):
        group_specs.append(("label", name, frame["label"].to_numpy() == label))
    for lead in sorted(frame["lead"].unique()):
        group_specs.append(("lead", str(int(lead)), frame["lead"].to_numpy() == lead))
        for label, name in ((1, "positive"), (0, "control")):
            mask = (frame["lead"].to_numpy() == lead) & (frame["label"].to_numpy() == label)
            group_specs.append(("lead_label", f"L{int(lead)}_{name}", mask))

    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in group_specs:
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        for horizon in range(yt.shape[1]):
            rows.append({
                "model": model_name,
                "group_type": group_type,
                "group_value": group_value,
                "horizon": horizon + 1,
                "n_samples": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((yt[:, horizon] - yp[:, horizon]) ** 2))),
                "qlike": float(qlike_loss(yt[:, horizon], yp[:, horizon]).mean()),
                "peak_abs_error": np.nan,
            })
        rows.append({
            "model": model_name,
            "group_type": group_type,
            "group_value": group_value,
            "horizon": "path",
            "n_samples": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "qlike": float(qlike_loss(yt, yp).mean()),
            "peak_abs_error": float(np.mean(np.abs(yt.max(axis=1) - yp.max(axis=1)))),
        })
    return rows


def run_classical_sanity_ladder(
    data: StageEData,
    *,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = data.manifest.reset_index(drop=True)
    sequences = np.asarray(data.sequences, dtype=float)
    validate_split_integrity(manifest)
    train_mask = manifest["split"].eq("train").to_numpy()
    val_mask = manifest["split"].eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_train, y_val = y[train_mask], y[val_mask]
    val_frame = manifest.loc[val_mask].reset_index(drop=True)

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[pd.DataFrame] = []
    selected_alphas: dict[str, float] = {}

    persistence = np.repeat(sequences[val_mask, -1, 0][:, None], len(TARGET_COLUMNS), axis=1)
    predictions_by_model: dict[str, np.ndarray] = {"persistence": persistence}

    for model_name in (
        "level_ridge",
        "har_ridge",
        "extended_har_ridge",
        "sequence_ridge",
        "shuffled_sequence_ridge",
    ):
        x = _feature_matrix(model_name, manifest, sequences, seed)
        prediction, alpha, tuning = _fit_ridge(x[train_mask], y_train, x[val_mask], y_val, alphas)
        selected_alphas[model_name] = alpha
        tuning.insert(0, "model", model_name)
        tuning_rows.append(tuning)
        predictions_by_model[model_name] = prediction

    for model_name, prediction in predictions_by_model.items():
        output = val_frame[["sample_id", "episode_id", "label", "lead", "split"]].copy()
        output.insert(0, "model", model_name)
        for h, column in enumerate(TARGET_COLUMNS):
            output[f"actual_h{h + 1}"] = y_val[:, h]
            output[f"predicted_h{h + 1}"] = prediction[:, h]
        prediction_rows.append(output)
        metric_rows.extend(_metric_rows(model_name, val_frame, y_val, prediction))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    pooled_path = metrics[(metrics["group_type"] == "pooled") & (metrics["horizon"] == "path")]
    ranked = pooled_path.sort_values(["qlike", "rmse"])
    summary = {
        "selection_split": "val",
        "test_evaluated": False,
        "primary_selection_metric": "pooled mean path QLIKE",
        "selected_alphas": selected_alphas,
        "validation_ranking": ranked[["model", "qlike", "rmse", "peak_abs_error"]].to_dict("records"),
        "questions": {
            "har_beats_persistence": bool(
                pooled_path.set_index("model").loc["har_ridge", "qlike"]
                < pooled_path.set_index("model").loc["persistence", "qlike"]
            ),
            "ordered_sequence_beats_har": bool(
                pooled_path.set_index("model").loc["sequence_ridge", "qlike"]
                < pooled_path.set_index("model").loc["har_ridge", "qlike"]
            ),
            "ordered_sequence_beats_shuffled": bool(
                pooled_path.set_index("model").loc["sequence_ridge", "qlike"]
                < pooled_path.set_index("model").loc["shuffled_sequence_ridge", "qlike"]
            ),
        },
    }
    return metrics, predictions, tuning, summary
