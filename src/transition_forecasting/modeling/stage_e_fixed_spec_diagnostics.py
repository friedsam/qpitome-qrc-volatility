from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_sequence_models import (
    DEFAULT_ESN_CONFIG,
    har_predictions,
    metrics,
    scale_sequences,
)


def mincer_zarnowitz(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return Mincer-Zarnowitz calibration diagnostics for y = alpha + beta*yhat.

    Multi-horizon paths are flattened so every sample-horizon forecast contributes one
    observation. The ideal calibrated forecast has alpha=0 and beta=1.
    """

    observed = np.asarray(y, dtype=float).reshape(-1)
    forecast = np.asarray(prediction, dtype=float).reshape(-1)
    finite = np.isfinite(observed) & np.isfinite(forecast)
    observed = observed[finite]
    forecast = forecast[finite]
    n = len(observed)
    if n < 4:
        return {
            "mz_n": float(n),
            "mz_intercept": np.nan,
            "mz_slope": np.nan,
            "mz_r2": np.nan,
            "mz_joint_f": np.nan,
            "mz_joint_p": np.nan,
        }

    design = np.column_stack([np.ones(n), forecast])
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    fitted = design @ coefficients
    residual = observed - fitted
    sse = float(residual @ residual)
    centered = observed - observed.mean()
    sst = float(centered @ centered)
    r2 = np.nan if sst <= 0.0 else 1.0 - sse / sst

    restricted_residual = observed - forecast
    restricted_sse = float(restricted_residual @ restricted_residual)
    restrictions = 2
    denominator_df = n - design.shape[1]
    if denominator_df <= 0 or sse <= 0.0:
        joint_f = np.nan
        joint_p = np.nan
    else:
        numerator = max(0.0, (restricted_sse - sse) / restrictions)
        denominator = sse / denominator_df
        joint_f = numerator / denominator
        joint_p = float(f_distribution.sf(joint_f, restrictions, denominator_df))

    return {
        "mz_n": float(n),
        "mz_intercept": float(coefficients[0]),
        "mz_slope": float(coefficients[1]),
        "mz_r2": float(r2),
        "mz_joint_f": float(joint_f),
        "mz_joint_p": float(joint_p),
    }


def horizon_mz_summary(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diagnostics = [
        mincer_zarnowitz(y[:, horizon], prediction[:, horizon])
        for horizon in range(y.shape[1])
    ]
    slopes = np.asarray([row["mz_slope"] for row in diagnostics], dtype=float)
    intercepts = np.asarray([row["mz_intercept"] for row in diagnostics], dtype=float)
    r2 = np.asarray([row["mz_r2"] for row in diagnostics], dtype=float)
    return {
        "mz_horizon_mean_intercept": float(np.nanmean(intercepts)),
        "mz_horizon_mean_slope": float(np.nanmean(slopes)),
        "mz_horizon_mean_r2": float(np.nanmean(r2)),
        "mz_horizon_mean_abs_slope_error": float(np.nanmean(np.abs(slopes - 1.0))),
    }


def evaluate_fold_fixed_with_mz(
    manifest: pd.DataFrame,
    sequences: np.ndarray,
    *,
    fold: int,
    seeds: tuple[int, ...] = (1, 2, 3),
    sequence_alpha: float = 100.0,
    esn_alpha: float = 1000.0,
    pca_components: int = 10,
    config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Evaluate fixed compact models with RMSE, QLIKE, and MZ diagnostics."""

    config = dict(DEFAULT_ESN_CONFIG if config is None else config)
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has an empty train or validation partition")
    if manifest["fold_split"].eq("test").any():
        raise ValueError("fixed-spec development evaluation must not receive test rows")

    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = har_predictions(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []

    def append(model: str, seed: int, alpha: float, prediction: np.ndarray) -> None:
        qlike, rmse = metrics(y, prediction, val_mask)
        val_y = y[val_mask]
        val_prediction = prediction[val_mask]
        rows.append(
            {
                "fold": int(fold),
                "model": model,
                "seed": int(seed),
                "alpha": float(alpha),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "val_qlike": qlike,
                "val_rmse": rmse,
                **mincer_zarnowitz(val_y, val_prediction),
                **horizon_mz_summary(val_y, val_prediction),
            }
        )

    append("har", 0, 100.0, har)

    flat = sequences.reshape(len(sequences), -1)
    flat_scaler = StandardScaler()
    flat_train = flat_scaler.fit_transform(flat[train_mask])
    flat_all = flat_scaler.transform(flat)
    sequence_model = Ridge(alpha=float(sequence_alpha))
    sequence_model.fit(flat_train, y[train_mask])
    append("sequence_ridge", 0, sequence_alpha, sequence_model.predict(flat_all))

    scaled_sequences = scale_sequences(sequences, train_mask)
    for seed in seeds:
        w_in, w = make_esn_weights(
            n_inputs=scaled_sequences.shape[-1],
            n_reservoir=int(config["n"]),
            spectral_radius=float(config["sr"]),
            input_scale=float(config["inp"]),
            seed=int(seed),
        )
        ordered = esn_states(scaled_sequences, w_in, w, float(config["leak"]))
        rng = np.random.default_rng(int(seed) + 20000)
        shuffled_sequences = scaled_sequences.copy()
        for sample in range(len(shuffled_sequences)):
            permutation = rng.permutation(shuffled_sequences.shape[1])
            shuffled_sequences[sample] = shuffled_sequences[sample, permutation, :]
        shuffled = esn_states(shuffled_sequences, w_in, w, float(config["leak"]))

        ordered_scaler = StandardScaler()
        ordered_train = ordered_scaler.fit_transform(ordered[train_mask])
        ordered_all = ordered_scaler.transform(ordered)
        count = min(int(pca_components), ordered_train.shape[0], ordered_train.shape[1])
        pca = PCA(n_components=count, svd_solver="full")
        pca_train = pca.fit_transform(ordered_train)
        pca_all = pca.transform(ordered_all)

        shuffled_scaler = StandardScaler()
        shuffled_train = shuffled_scaler.fit_transform(shuffled[train_mask])
        shuffled_all = shuffled_scaler.transform(shuffled)

        for model_name, x_train, x_all in (
            ("pca10_esn", pca_train, pca_all),
            ("shuffled_esn", shuffled_train, shuffled_all),
        ):
            readout = Ridge(alpha=float(esn_alpha))
            readout.fit(x_train, residual[train_mask])
            append(model_name, seed, esn_alpha, har + readout.predict(x_all))

    return pd.DataFrame(rows)
