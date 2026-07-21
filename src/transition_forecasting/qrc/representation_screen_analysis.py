from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
)
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import metrics
from transition_forecasting.qrc.representation_candidates import SECOND_CHANNEL_NAMES
from transition_forecasting.qrc.temporal_rydberg_chain import effective_rank
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import signal_diagnostics

ReadoutMode = str


def _fit_har(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _prequential_har_residuals(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal training residuals and the rows on which they are valid."""
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["origin_date"], errors="raise", utc=True)
    train_dates = np.asarray(sorted(dates[train_mask].unique()))
    chunks = [
        chunk
        for chunk in np.array_split(train_dates, min(blocks, len(train_dates)))
        if len(chunk)
    ]
    predictions = np.full_like(y, np.nan, dtype=float)

    for block_index in range(1, len(chunks)):
        score_dates = chunks[block_index]
        first_score_date = score_dates[0]
        fit_mask = train_mask & dates.lt(first_score_date).to_numpy()
        score_mask = train_mask & dates.isin(score_dates).to_numpy()
        if fit_mask.sum() < max(10, len(HAR_FEATURES) + 2) or not score_mask.any():
            continue
        scaler = StandardScaler()
        model = Ridge(alpha=100.0)
        model.fit(scaler.fit_transform(x[fit_mask]), y[fit_mask])
        predictions[score_mask] = model.predict(scaler.transform(x[score_mask]))

    valid = train_mask & np.isfinite(predictions).all(axis=1)
    residuals = y - predictions
    return residuals, valid


def _metric_payload(
    y: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    qlike, rmse = metrics(y, prediction, mask)
    observed = y[mask]
    forecast = prediction[mask]
    return {
        "qlike": float(qlike),
        "rmse": float(rmse),
        **mincer_zarnowitz(observed, forecast),
        **horizon_mz_summary(observed, forecast),
        **signal_diagnostics(observed, forecast),
    }


def _component_values(
    requested: Iterable[int],
    train_rows: int,
    width: int,
) -> tuple[int, ...]:
    maximum = max(1, min(train_rows, width))
    values = []
    for value in requested:
        resolved = maximum if int(value) == 0 else min(int(value), maximum)
        if resolved >= 1 and resolved not in values:
            values.append(resolved)
    return tuple(values)


def _evaluate_feature_matrix(
    *,
    feature_matrix: np.ndarray,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    fold: int,
    representation: str,
    model_family: str,
    seed: int,
    alphas: tuple[float, ...],
    components: tuple[int, ...],
    readout_modes: tuple[ReadoutMode, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features = np.asarray(feature_matrix, dtype=float)
    if features.ndim != 2 or len(features) != len(frame):
        raise ValueError("feature matrix is not aligned with the selected frame")
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(features[train_mask])
    scaled_all = scaler.transform(features)
    component_grid = _component_values(
        components,
        len(scaled_train),
        features.shape[1],
    )
    metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for component_count in component_grid:
        if component_count < scaled_train.shape[1]:
            pca = PCA(n_components=component_count, svd_solver="full")
            transformed_train = pca.fit_transform(scaled_train)
            transformed_all = pca.transform(scaled_all)
            explained = float(pca.explained_variance_ratio_.sum())
        else:
            transformed_train = scaled_train
            transformed_all = scaled_all
            explained = 1.0

        for readout_mode in readout_modes:
            if readout_mode == "direct":
                fit_mask = train_mask
                fit_x = transformed_all[fit_mask]
                fit_y = y[fit_mask]
                baseline = np.zeros_like(y)
            else:
                fit_mask = residual_train_mask
                if not fit_mask.any():
                    continue
                fit_x = transformed_all[fit_mask]
                fit_y = residuals[fit_mask]
                baseline = har

            for alpha in alphas:
                readout = Ridge(alpha=float(alpha))
                readout.fit(fit_x, fit_y)
                prediction = baseline + readout.predict(transformed_all)
                row = {
                    "fold": int(fold),
                    "representation": representation,
                    "second_channel": SECOND_CHANNEL_NAMES[representation],
                    "model_family": model_family,
                    "seed": int(seed),
                    "readout_mode": readout_mode,
                    "components": int(component_count),
                    "feature_width": int(features.shape[1]),
                    "explained_variance": explained,
                    "alpha": float(alpha),
                    "train_rows_used": int(fit_mask.sum()),
                    "val_rows": int(val_mask.sum()),
                    **{
                        f"val_{key}": value
                        for key, value in _metric_payload(
                            y,
                            prediction,
                            val_mask,
                        ).items()
                    },
                }
                metric_rows.append(row)

                validation = frame.loc[val_mask].reset_index(drop=True)
                y_val = y[val_mask]
                pred_val = prediction[val_mask]
                for lead in sorted(validation["lead"].unique()):
                    for label in sorted(validation["label"].unique()):
                        local = (
                            validation["lead"].eq(lead).to_numpy()
                            & validation["label"].eq(label).to_numpy()
                        )
                        if not local.any():
                            continue
                        local_mask = np.ones(int(local.sum()), dtype=bool)
                        payload = _metric_payload(
                            y_val[local],
                            pred_val[local],
                            local_mask,
                        )
                        group_rows.append(
                            {
                                "fold": int(fold),
                                "representation": representation,
                                "model_family": model_family,
                                "seed": int(seed),
                                "readout_mode": readout_mode,
                                "components": int(component_count),
                                "alpha": float(alpha),
                                "lead": int(lead),
                                "label": int(label),
                                "samples": int(local.sum()),
                                **payload,
                            }
                        )
    return metric_rows, group_rows


def _feature_diagnostic(
    *,
    features: np.ndarray,
    train_mask: np.ndarray,
    fold: int,
    representation: str,
    model_family: str,
    seed: int,
) -> dict[str, object]:
    train = np.asarray(features[train_mask], dtype=float)
    centered = train - train.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    fraction = (
        variance / variance.sum()
        if variance.sum() > 0
        else np.zeros_like(variance)
    )
    cumulative = np.cumsum(fraction)

    def dimensions(threshold: float) -> int:
        if not cumulative.size or cumulative[-1] <= 0:
            return 0
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "fold": int(fold),
        "representation": representation,
        "model_family": model_family,
        "seed": int(seed),
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "effective_rank_train": effective_rank(train),
        "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
        "dimensions_90pct_variance": dimensions(0.90),
        "dimensions_95pct_variance": dimensions(0.95),
        "dimensions_99pct_variance": dimensions(0.99),
    }


def _write_feature_archive(
    path: Path,
    blocks: list[dict[str, object]],
    *,
    feature_names: tuple[str, ...],
) -> None:
    if not blocks:
        return
    features = np.concatenate(
        [np.asarray(block["features"], dtype=float) for block in blocks]
    )
    targets = np.concatenate(
        [np.asarray(block["targets"], dtype=float) for block in blocks]
    )
    har = np.concatenate(
        [np.asarray(block["har"], dtype=float) for block in blocks]
    )
    input_sequences = np.concatenate(
        [np.asarray(block["input_sequences"], dtype=float) for block in blocks]
    )
    prequential_residuals = np.concatenate(
        [
            np.asarray(block["prequential_residuals"], dtype=float)
            for block in blocks
        ]
    )
    prequential_valid = np.concatenate(
        [np.asarray(block["prequential_valid"], dtype=bool) for block in blocks]
    )
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match feature-matrix width")
    metadata: dict[str, np.ndarray] = {}
    for key in (
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
    ):
        values = np.concatenate([np.asarray(block[key]) for block in blocks])
        metadata[key] = (
            values.astype(int)
            if key in {"fold", "lead", "label"}
            else values.astype(str)
        )
    metadata["representation"] = np.concatenate(
        [
            np.repeat(str(block["representation"]), len(block["sample_id"]))
            for block in blocks
        ]
    ).astype(str)
    metadata["seed"] = np.concatenate(
        [
            np.repeat(int(block["seed"]), len(block["sample_id"]))
            for block in blocks
        ]
    ).astype(int)
    np.savez_compressed(
        path,
        feature_matrix=features,
        feature_names=np.asarray(feature_names, dtype=str),
        input_sequences=input_sequences,
        target_path=targets,
        har_prediction_path=har,
        prequential_har_residual_path=prequential_residuals,
        prequential_residual_valid=prequential_valid,
        **metadata,
    )


def _best_mean_configuration(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics_frame.groupby(
            [
                "representation",
                "model_family",
                "readout_mode",
                "components",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_mz_r2=("val_mz_r2", "mean"),
            folds=("fold", "nunique"),
            seeds=("seed", "nunique"),
        )
        .sort_values(
            [
                "representation",
                "model_family",
                "mean_val_qlike",
                "mean_val_rmse",
            ]
        )
    )
    return grouped.groupby(
        ["representation", "model_family"],
        as_index=False,
    ).head(1)


def _write_summary_plot(best: pd.DataFrame, path: Path) -> None:
    pivot = best.pivot(
        index="representation",
        columns="model_family",
        values="mean_val_qlike",
    )
    ax = pivot.plot(kind="bar", figsize=(11, 5.8))
    ax.set_ylabel("Mean validation QLIKE")
    ax.set_xlabel("Input representation")
    ax.set_title(
        "Representation screen: best development configuration by model family"
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Model family", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)
