from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    chronological_inner_split,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
)

ReadoutKind = Literal["linear", "polynomial_degree2"]
CalibrationKind = Literal["global", "segmented"]

MODEL_VARIANTS: tuple[tuple[str, ReadoutKind, CalibrationKind], ...] = (
    ("ladder_linear_global", "linear", "global"),
    ("ladder_linear_segmented", "linear", "segmented"),
    ("ladder_poly2_global", "polynomial_degree2", "global"),
    ("ladder_poly2_segmented", "polynomial_degree2", "segmented"),
)


@dataclass(frozen=True)
class LadderReadoutUpgradeConfig:
    """Post-confirmation development protocol for two targeted readout upgrades."""

    folds: tuple[int, ...] = tuple(range(1, 9))
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    global_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
    )
    linear_alpha: float = 100.0
    polynomial_alphas: tuple[float, ...] = (100.0, 1000.0, 10000.0)
    polynomial_degree: int = 2
    ladder_interaction_scale: float = 1.25
    seed: int = 20260722
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or any(int(fold) < 1 for fold in self.folds):
            raise ValueError("folds must be positive and nonempty")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if not self.leads:
            raise ValueError("leads cannot be empty")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not 1 <= self.split_horizon < 10:
            raise ValueError("split_horizon must lie between 1 and 9")
        for name, grid in (
            ("global_lambdas", self.global_lambdas),
            ("early_lambdas", self.early_lambdas),
            ("transition_lambdas", self.transition_lambdas),
        ):
            if not grid or any(float(value) < 0 for value in grid):
                raise ValueError(f"{name} must be nonempty and nonnegative")
        if self.linear_alpha <= 0:
            raise ValueError("linear_alpha must be positive")
        if not self.polynomial_alphas or any(
            float(value) <= 0 for value in self.polynomial_alphas
        ):
            raise ValueError("polynomial_alphas must be positive")
        if self.polynomial_degree != 2:
            raise ValueError("this controlled assay supports degree two only")
        if self.ladder_interaction_scale <= 0:
            raise ValueError("ladder_interaction_scale must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_arrays(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(matrix, dtype=float)
    targets = np.asarray(residuals, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("matrix must be finite and two-dimensional")
    if targets.ndim != 2 or len(targets) != len(features):
        raise ValueError("residuals must align with the feature matrix")
    if fit.shape != (len(features),) or not fit.any():
        raise ValueError("fit_mask must select aligned rows")
    if not np.isfinite(targets[fit]).all():
        raise ValueError("fitted residual targets must be finite")
    return features, targets, fit


def fit_residual_correction(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    readout_kind: ReadoutKind,
    ridge_alpha: float,
    polynomial_degree: int = 2,
) -> tuple[np.ndarray, int]:
    """Fit one multi-output residual head and predict all aligned rows.

    Polynomial terms are constructed from train-standardized physical modes and
    standardized again after expansion. This prevents high-variance squares from
    receiving an arbitrary scale advantage while keeping all preprocessing causal.
    """

    features, targets, fit = _validated_arrays(matrix, residuals, fit_mask)
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    raw_scaler = StandardScaler().fit(features[fit])
    standardized = raw_scaler.transform(features)

    if readout_kind == "linear":
        design = standardized
    elif readout_kind == "polynomial_degree2":
        if polynomial_degree != 2:
            raise ValueError("polynomial readout is frozen to degree two")
        expansion = PolynomialFeatures(
            degree=2,
            include_bias=False,
            interaction_only=False,
        )
        expanded = expansion.fit_transform(standardized)
        expanded_scaler = StandardScaler().fit(expanded[fit])
        design = expanded_scaler.transform(expanded)
    else:
        raise ValueError(f"unsupported readout kind: {readout_kind}")

    model = Ridge(alpha=float(ridge_alpha)).fit(design[fit], targets[fit])
    prediction = np.asarray(model.predict(design), dtype=float)
    if prediction.shape != targets.shape or not np.isfinite(prediction).all():
        raise RuntimeError("residual head returned an invalid correction path")
    return prediction, int(design.shape[1])


def apply_calibration(
    har: np.ndarray,
    correction: np.ndarray,
    *,
    early_lambda: float,
    transition_lambda: float,
    split_horizon: int,
) -> np.ndarray:
    baseline = np.asarray(har, dtype=float)
    residual = np.asarray(correction, dtype=float)
    if baseline.shape != residual.shape or baseline.ndim != 2:
        raise ValueError("HAR and correction paths must be aligned matrices")
    if not 1 <= split_horizon < baseline.shape[1]:
        raise ValueError("split_horizon is outside the forecast path")
    if early_lambda < 0 or transition_lambda < 0:
        raise ValueError("calibration weights must be nonnegative")
    scale = np.full(baseline.shape[1], float(transition_lambda), dtype=float)
    scale[: int(split_horizon)] = float(early_lambda)
    return baseline + residual * scale[None, :]


def _candidate_key(row: dict[str, float]) -> tuple[float, ...]:
    return (
        float(row["inner_qlike"]),
        float(row["inner_rmse"]),
        float(row["ridge_alpha"]),
        float(row["transition_lambda"] + row["early_lambda"]),
        float(row["transition_lambda"]),
        float(row["early_lambda"]),
    )


def calibration_candidates(
    correction: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    tune_mask: np.ndarray,
    calibration_kind: CalibrationKind,
    ridge_alpha: float,
    config: LadderReadoutUpgradeConfig,
) -> list[dict[str, float]]:
    tune = np.asarray(tune_mask, dtype=bool)
    if tune.shape != (len(y),) or not tune.any():
        raise ValueError("tune_mask must select aligned rows")
    if calibration_kind == "global":
        pairs = [(value, value) for value in config.global_lambdas]
    elif calibration_kind == "segmented":
        pairs = [
            (early, transition)
            for early in config.early_lambdas
            for transition in config.transition_lambdas
            if float(transition) >= float(early)
        ]
    else:
        raise ValueError(f"unsupported calibration kind: {calibration_kind}")

    rows: list[dict[str, float]] = []
    for early, transition in pairs:
        prediction = apply_calibration(
            har,
            correction,
            early_lambda=float(early),
            transition_lambda=float(transition),
            split_horizon=config.split_horizon,
        )
        payload = _metric_payload(y, prediction, tune)
        rows.append(
            {
                "ridge_alpha": float(ridge_alpha),
                "early_lambda": float(early),
                "transition_lambda": float(transition),
                **{f"inner_{key}": float(value) for key, value in payload.items()},
            }
        )
    return rows


def select_inner_configuration(
    matrix: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    origin_date: np.ndarray,
    readout_kind: ReadoutKind,
    calibration_kind: CalibrationKind,
    config: LadderReadoutUpgradeConfig,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, int]:
    """Select alpha and calibration only on a chronological inner holdout."""

    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        residual_train_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )
    alphas = (
        (config.linear_alpha,)
        if readout_kind == "linear"
        else config.polynomial_alphas
    )
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        correction, _ = fit_residual_correction(
            matrix,
            residuals,
            inner_fit,
            readout_kind=readout_kind,
            ridge_alpha=float(alpha),
            polynomial_degree=config.polynomial_degree,
        )
        rows.extend(
            calibration_candidates(
                correction,
                y=y,
                har=har,
                tune_mask=inner_tune,
                calibration_kind=calibration_kind,
                ridge_alpha=float(alpha),
                config=config,
            )
        )
    candidate_frame = pd.DataFrame(rows)
    selected = min(rows, key=_candidate_key)
    alpha = float(selected["ridge_alpha"])
    full_correction, design_width = fit_residual_correction(
        matrix,
        residuals,
        residual_train_mask,
        readout_kind=readout_kind,
        ridge_alpha=alpha,
        polynomial_degree=config.polynomial_degree,
    )
    diagnostics = {
        **selected,
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
        "raw_feature_width": float(np.asarray(matrix).shape[1]),
        "design_feature_width": float(design_width),
    }
    return diagnostics, candidate_frame, full_correction, int(design_width)


def prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    *,
    mask: np.ndarray,
    model_name: str,
    readout_kind: str,
    calibration_kind: str,
    ridge_alpha: float,
    early_lambda: float,
    transition_lambda: float,
) -> pd.DataFrame:
    selected = frame.loc[mask].reset_index(drop=True)
    horizons = int(y.shape[1])
    count = int(mask.sum()) * horizons
    return pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(
                selected["sample_id"].astype(str).to_numpy(), horizons
            ),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "episode_id": np.repeat(
                selected["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                selected["origin_date"].astype(str).to_numpy(), horizons
            ),
            "evaluation_split": np.repeat("val", count),
            "model_name": np.repeat(model_name, count),
            "readout_kind": np.repeat(readout_kind, count),
            "calibration_kind": np.repeat(calibration_kind, count),
            "ridge_alpha": np.repeat(float(ridge_alpha), count),
            "early_lambda": np.repeat(float(early_lambda), count),
            "transition_lambda": np.repeat(float(transition_lambda), count),
            "horizon": np.tile(np.arange(1, horizons + 1), int(mask.sum())),
            "y_true": np.asarray(y)[mask].reshape(-1),
            "y_pred": np.asarray(prediction)[mask].reshape(-1),
            "har_pred": np.asarray(har)[mask].reshape(-1),
        }
    )


def pooled_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, local in predictions.groupby("model_name"):
        payload = _metric_payload(
            local["y_true"].to_numpy(dtype=float)[:, None],
            local["y_pred"].to_numpy(dtype=float)[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": str(model_name),
                "folds": int(local["fold"].nunique()),
                "rows": int(len(local)),
                **{key: float(value) for key, value in payload.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["qlike", "rmse"])


def period_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = {
        "folds_1_3": (1, 2, 3),
        "folds_4_8": (4, 5, 6, 7, 8),
        "all_folds": tuple(sorted(predictions["fold"].unique())),
    }
    for period, folds in periods.items():
        selected = predictions.loc[predictions["fold"].isin(folds)]
        if selected.empty:
            continue
        for model_name, local in selected.groupby("model_name"):
            payload = _metric_payload(
                local["y_true"].to_numpy(dtype=float)[:, None],
                local["y_pred"].to_numpy(dtype=float)[:, None],
                np.ones(len(local), dtype=bool),
            )
            rows.append(
                {
                    "period": period,
                    "model_name": str(model_name),
                    "folds": int(local["fold"].nunique()),
                    "rows": int(len(local)),
                    **{key: float(value) for key, value in payload.items()},
                }
            )
    return pd.DataFrame(rows).sort_values(["period", "qlike", "rmse"])


def group_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, local in predictions.groupby(
        ["model_name", "lead", "label"],
        dropna=False,
    ):
        payload = _metric_payload(
            local["y_true"].to_numpy(dtype=float)[:, None],
            local["y_pred"].to_numpy(dtype=float)[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": str(keys[0]),
                "lead": int(keys[1]),
                "label": int(keys[2]),
                "folds": int(local["fold"].nunique()),
                "rows": int(len(local)),
                **{key: float(value) for key, value in payload.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["lead", "label", "qlike", "rmse"])


def horizon_mean_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, local in predictions.groupby(
        ["model_name", "lead", "label", "horizon"],
        dropna=False,
    ):
        actual = local["y_true"].to_numpy(dtype=float)
        forecast = local["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": str(keys[0]),
                "lead": int(keys[1]),
                "label": int(keys[2]),
                "horizon": int(keys[3]),
                "samples": int(len(local)),
                "actual_mean": float(actual.mean()),
                "forecast_mean": float(forecast.mean()),
                "forecast_bias": float((forecast - actual).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lead", "label", "model_name", "horizon"]
    )
