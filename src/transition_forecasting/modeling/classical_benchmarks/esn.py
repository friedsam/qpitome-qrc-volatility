from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import transform_input
from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling.classical_benchmarks.common import TARGET_COLUMNS

REPRESENTATION = "level_diff_time"
POOLING = "final_mean_std"
WASHOUT = 10


def _fold_rows(dataset, fold: int):
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


def _scale_channels(inputs: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = inputs[train_mask].reshape(-1, inputs.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (inputs - mean[None, None, :]) / scale[None, None, :]


def _shuffle_windows(inputs: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 20000)
    shuffled = np.asarray(inputs, dtype=float).copy()
    for sample in range(len(shuffled)):
        shuffled[sample] = shuffled[
            sample,
            rng.permutation(shuffled.shape[1]),
            :,
        ]
    return shuffled


def pooled_features_vectorized(
    inputs: np.ndarray,
    *,
    w_in: np.ndarray,
    w: np.ndarray,
    leak: float,
    washout: int = WASHOUT,
) -> np.ndarray:
    """Pool final, mean, and standard-deviation reservoir states."""
    values = np.asarray(inputs, dtype=float)
    recurrent = csr_matrix(w)
    state = np.zeros((len(values), w.shape[0]), dtype=float)
    state_sum = np.zeros_like(state)
    state_sq_sum = np.zeros_like(state)
    kept = 0
    for step in range(values.shape[1]):
        recurrent_term = recurrent.dot(state.T).T
        candidate = np.tanh(values[:, step, :] @ w_in.T + recurrent_term)
        state = (1.0 - leak) * state + leak * candidate
        if step >= washout:
            state_sum += state
            state_sq_sum += state * state
            kept += 1
    mean = state_sum / kept
    variance = np.maximum(state_sq_sum / kept - mean * mean, 0.0)
    return np.concatenate([state, mean, np.sqrt(variance)], axis=1)


def _seed_features(
    inputs: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    w_in, w = make_esn_weights(
        n_inputs=inputs.shape[-1],
        n_reservoir=int(config["n"]),
        spectral_radius=float(config["sr"]),
        input_scale=float(config["inp"]),
        seed=int(seed),
        connectivity=float(config["connectivity"]),
    )
    ordered = pooled_features_vectorized(
        inputs,
        w_in=w_in,
        w=w,
        leak=float(config["leak"]),
    )
    shuffled = pooled_features_vectorized(
        _shuffle_windows(inputs, seed),
        w_in=w_in,
        w=w,
        leak=float(config["leak"]),
    )
    return ordered, shuffled


def _ridge_path_predictions(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Exact standardized Ridge predictions for a full alpha path from one SVD."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_mask])
    val_features = scaler.transform(features[val_mask])
    train_target = np.asarray(target[train_mask], dtype=float)
    target_mean = train_target.mean(axis=0, keepdims=True)
    centered_target = train_target - target_mean
    u, singular, vt = np.linalg.svd(train_features, full_matrices=False)
    val_projection = val_features @ vt.T
    target_projection = u.T @ centered_target
    return {
        float(alpha): target_mean
        + (
            val_projection
            * (singular / (singular * singular + float(alpha)))
        )
        @ target_projection
        for alpha in alphas
    }


def _readout_prediction(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return _ridge_path_predictions(
        features,
        target,
        train_mask,
        val_mask,
        (float(alpha),),
    )[float(alpha)]


def predict_fold(
    dataset,
    *,
    fold: int,
    config: dict[str, object],
    alpha: float,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    frame, sequences, train_mask, val_mask = _fold_rows(dataset, fold)
    target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_val = target[val_mask]
    inputs = _scale_channels(
        transform_input(sequences, REPRESENTATION),
        train_mask,
    )
    ordered_predictions = []
    shuffled_predictions = []
    for seed in seeds:
        ordered, shuffled = _seed_features(inputs, config, seed)
        ordered_predictions.append(
            _readout_prediction(ordered, target, train_mask, val_mask, alpha)
        )
        shuffled_predictions.append(
            _readout_prediction(shuffled, target, train_mask, val_mask, alpha)
        )
    predictions = {
        "esn_direct_tuned": np.mean(ordered_predictions, axis=0),
        "esn_shuffled_tuned": np.mean(shuffled_predictions, axis=0),
    }
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    metadata = [
        "sample_id", "episode_id", "index", "market_group", "origin_date",
        "event_onset", "label", "lead", "control_stratum", "fold", "fold_split",
    ]
    rows = []
    for model, prediction in predictions.items():
        output = val_frame[metadata].copy()
        output.insert(0, "model", model)
        output["config_id"] = str(config["config_id"])
        output["alpha"] = float(alpha)
        output["seeds"] = ",".join(str(seed) for seed in seeds)
        for horizon in range(1, 11):
            output[f"actual_h{horizon}"] = y_val[:, horizon - 1]
            output[f"predicted_h{horizon}"] = prediction[:, horizon - 1]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)
