from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
)

MODEL_NAMES = ("mse_har", "mse_har_qlike_offset", "qlike_har")


@dataclass(frozen=True)
class L5ResidualFoundationConfig:
    """Leakage-safe residual construction before any QRC row selection."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    selection_folds: tuple[int, ...] = (4, 5, 6)
    confirmation_folds: tuple[int, ...] = (7, 8)
    lead: int = 5
    target_horizon: int = 10
    target_availability_calendar_days: int = 21
    min_fit_rows: int = 40
    min_calibration_rows: int = 20
    ridge_alpha: float = 100.0
    qlike_max_iter: int = 100
    qlike_gradient_tolerance: float = 1e-8
    models: tuple[str, ...] = MODEL_NAMES

    def validate(self) -> None:
        selection = set(self.selection_folds)
        confirmation = set(self.confirmation_folds)
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if selection & confirmation or selection | confirmation != set(self.folds):
            raise ValueError("selection and confirmation folds must partition folds")
        if any(int(fold) <= 3 for fold in self.folds):
            raise ValueError("residual foundation is restricted to development folds 4+")
        if self.lead != 5:
            raise ValueError("residual foundation is intentionally restricted to L5")
        if not 1 <= self.target_horizon <= len(TARGET_COLUMNS):
            raise ValueError("target_horizon lies outside the target path")
        if self.target_availability_calendar_days < self.target_horizon:
            raise ValueError("target availability embargo is shorter than target horizon")
        if self.min_fit_rows < 10 * (len(HAR_FEATURES) + 1):
            raise ValueError("min_fit_rows must provide ten rows per HAR parameter")
        if self.min_calibration_rows < 10:
            raise ValueError("min_calibration_rows must be at least ten")
        if self.ridge_alpha <= 0 or self.qlike_max_iter < 1:
            raise ValueError("invalid model controls")
        if self.qlike_gradient_tolerance <= 0:
            raise ValueError("qlike_gradient_tolerance must be positive")
        unknown = set(self.models) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        if "mse_har_qlike_offset" in self.models and "mse_har" not in self.models:
            raise ValueError("mse_har_qlike_offset requires mse_har")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FittedPathModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    model_name: str
    converged: tuple[bool, ...]
    iterations: tuple[int, ...]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        features = np.asarray(matrix, dtype=float)
        standardized = (
            features - self.feature_mean[None, :]
        ) / self.feature_scale[None, :]
        design = np.column_stack([np.ones(len(features)), standardized])
        return design @ self.coefficients.T


def parse_utc(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values)
    try:
        return pd.to_datetime(series, errors="raise", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="raise", utc=True)


def qlike_logvol_cells(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    true_logvar = np.clip(2.0 * np.asarray(y_true, dtype=float), -40.0, 20.0)
    pred_logvar = np.clip(2.0 * np.asarray(prediction, dtype=float), -40.0, 20.0)
    log_ratio = np.clip(true_logvar - pred_logvar, -40.0, 40.0)
    ratio = np.exp(log_ratio)
    return ratio - log_ratio - 1.0


def qlike_score_residual(log_residual: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(2.0 * np.asarray(log_residual), -40.0, 40.0)) - 1.0


def exact_qlike_offset(log_residual: np.ndarray) -> np.ndarray:
    """Constant log-volatility correction minimizing mean QLIKE exactly."""

    residual = np.asarray(log_residual, dtype=float)
    if residual.ndim == 1:
        residual = residual[:, None]
    if residual.ndim != 2 or not len(residual) or not np.isfinite(residual).all():
        raise ValueError("log residuals must be a finite nonempty matrix")
    doubled = 2.0 * residual
    maximum = doubled.max(axis=0)
    log_mean_exp = maximum + np.log(
        np.mean(np.exp(doubled - maximum[None, :]), axis=0)
    )
    return 0.5 * log_mean_exp


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(matrix, dtype=float)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("feature matrix must be finite and nonempty")
    mean = features.mean(axis=0)
    scale = np.where(features.std(axis=0) > 1e-12, features.std(axis=0), 1.0)
    return (features - mean[None, :]) / scale[None, :], mean, scale


def fit_mse_har(matrix: np.ndarray, target: np.ndarray, *, alpha: float) -> FittedPathModel:
    features = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    standardized, mean, scale = _standardize(features)
    model = Ridge(alpha=float(alpha), fit_intercept=True).fit(standardized, y)
    coefficients = np.column_stack(
        [np.atleast_1d(model.intercept_), np.atleast_2d(model.coef_)]
    )
    if coefficients.shape != (y.shape[1], features.shape[1] + 1):
        raise RuntimeError("unexpected MSE-HAR coefficient shape")
    return FittedPathModel(
        mean,
        scale,
        coefficients,
        "mse_har",
        tuple(True for _ in range(y.shape[1])),
        tuple(1 for _ in range(y.shape[1])),
    )


def _qlike_objective_gradient_hessian(
    beta: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    residual = np.asarray(target, dtype=float) - design @ beta
    doubled = np.clip(2.0 * residual, -40.0, 40.0)
    ratio = np.exp(doubled)
    objective = float(
        np.sum(ratio - doubled - 1.0) + float(alpha) * np.dot(beta[1:], beta[1:])
    )
    gradient = 2.0 * design.T @ (1.0 - ratio)
    gradient[1:] += 2.0 * float(alpha) * beta[1:]
    hessian = 4.0 * design.T @ (ratio[:, None] * design)
    hessian[1:, 1:] += 2.0 * float(alpha) * np.eye(design.shape[1] - 1)
    return objective, gradient, hessian


def _fit_qlike_horizon(
    design: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, bool, int]:
    beta = np.asarray(initial, dtype=float).copy()
    for iteration in range(1, int(max_iter) + 1):
        objective, gradient, hessian = _qlike_objective_gradient_hessian(
            beta, design, target, alpha
        )
        if np.max(np.abs(gradient)) <= tolerance:
            return beta, True, iteration - 1
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        directional = float(gradient @ step)
        if not np.isfinite(directional) or directional <= 0.0:
            step = gradient
            directional = float(gradient @ gradient)
        step_scale = 1.0
        for _ in range(40):
            candidate = beta - step_scale * step
            candidate_objective, _, _ = _qlike_objective_gradient_hessian(
                candidate, design, target, alpha
            )
            if candidate_objective <= objective - 1e-4 * step_scale * directional:
                beta = candidate
                break
            step_scale *= 0.5
        else:
            return beta, False, iteration
        if np.max(np.abs(step_scale * step)) <= 1e-10:
            _, final_gradient, _ = _qlike_objective_gradient_hessian(
                beta, design, target, alpha
            )
            return beta, bool(np.max(np.abs(final_gradient)) <= 10.0 * tolerance), iteration
    return beta, False, int(max_iter)


def fit_qlike_har(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    gradient_tolerance: float,
    initial_model: FittedPathModel | None = None,
) -> FittedPathModel:
    features = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    standardized, mean, scale = _standardize(features)
    design = np.column_stack([np.ones(len(features)), standardized])
    initial_model = initial_model or fit_mse_har(features, y, alpha=alpha)
    if not np.allclose(initial_model.feature_mean, mean) or not np.allclose(
        initial_model.feature_scale, scale
    ):
        raise ValueError("initial model uses different feature scaling")
    fitted = [
        _fit_qlike_horizon(
            design,
            y[:, horizon],
            initial_model.coefficients[horizon],
            alpha=alpha,
            max_iter=max_iter,
            tolerance=gradient_tolerance,
        )
        for horizon in range(y.shape[1])
    ]
    return FittedPathModel(
        mean,
        scale,
        np.vstack([item[0] for item in fitted]),
        "qlike_har",
        tuple(item[1] for item in fitted),
        tuple(item[2] for item in fitted),
    )


def target_availability_dates(
    frame: pd.DataFrame,
    config: L5ResidualFoundationConfig,
) -> tuple[pd.Series, str]:
    exact_columns = (
        "target_end_date",
        f"target_x_h{config.target_horizon}_date",
        f"target_date_h{config.target_horizon}",
        f"h{config.target_horizon}_date",
    )
    for column in exact_columns:
        if column in frame.columns:
            parsed = parse_utc(frame[column])
            if parsed.notna().all():
                return parsed, f"manifest:{column}"
    origin = parse_utc(frame["origin_date"])
    days = int(config.target_availability_calendar_days)
    return origin + pd.to_timedelta(days, unit="D"), f"calendar_embargo:{days}d"


def _validate_manifest(manifest: pd.DataFrame, config: L5ResidualFoundationConfig) -> None:
    required = {
        "sample_id",
        "fold",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
        *HAR_FEATURES,
        *TARGET_COLUMNS[: config.target_horizon],
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    unknown = set(manifest["fold_split"].dropna().astype(str)) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unexpected fold_split values: {sorted(unknown)}")


def build_fold_residual_registry(
    manifest: pd.DataFrame,
    *,
    fold: int,
    config: L5ResidualFoundationConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config.validate()
    _validate_manifest(manifest, config)
    frame = manifest.loc[
        manifest["fold"].eq(int(fold))
        & manifest["lead"].eq(config.lead)
        & manifest["fold_split"].isin(("train", "val"))
    ].copy()
    if frame.empty:
        raise RuntimeError(f"fold {fold}: no L5 development rows")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"fold {fold}: duplicate sample_id")
    frame["_origin"] = parse_utc(frame["origin_date"]).to_numpy()
    availability, source = target_availability_dates(frame, config)
    frame["_available"] = availability.to_numpy()
    frame = frame.sort_values(
        ["_origin", "fold_split", "label", "episode_id", "sample_id"]
    ).reset_index(drop=True)

    features = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    targets = frame[list(TARGET_COLUMNS[: config.target_horizon])].to_numpy(dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"fold {fold}: nonfinite features or targets")
    train = frame["fold_split"].eq("train").to_numpy()
    origins = pd.DatetimeIndex(frame["_origin"])
    available = pd.DatetimeIndex(frame["_available"])
    rows, horizons = targets.shape
    predictions = {
        name: np.full((rows, horizons), np.nan, dtype=float) for name in config.models
    }
    fit_counts = np.zeros(rows, dtype=int)
    calibration_counts = np.zeros(rows, dtype=int)
    convergence: list[dict[str, object]] = []

    for score_date in sorted(origins.unique()):
        score = origins == score_date
        fit = train & (available < score_date)
        fit_count = int(fit.sum())
        fit_counts[score] = fit_count
        if fit_count < config.min_fit_rows:
            continue
        mse = fit_mse_har(features[fit], targets[fit], alpha=config.ridge_alpha)
        mse_score_prediction = mse.predict(features[score])
        if "mse_har" in predictions:
            predictions["mse_har"][score] = mse_score_prediction
        if "qlike_har" in predictions:
            qlike = fit_qlike_har(
                features[fit],
                targets[fit],
                alpha=config.ridge_alpha,
                max_iter=config.qlike_max_iter,
                gradient_tolerance=config.qlike_gradient_tolerance,
                initial_model=mse,
            )
            predictions["qlike_har"][score] = qlike.predict(features[score])
            convergence.append(
                {
                    "score_date": pd.Timestamp(score_date).isoformat(),
                    "fit_rows": fit_count,
                    "all_horizons_converged": bool(all(qlike.converged)),
                    "max_iterations": int(max(qlike.iterations, default=0)),
                }
            )
        if "mse_har_qlike_offset" in predictions:
            prior = predictions["mse_har"]
            calibration = train & (available < score_date) & np.isfinite(prior).all(axis=1)
            calibration_count = int(calibration.sum())
            calibration_counts[score] = calibration_count
            if calibration_count >= config.min_calibration_rows:
                offset = exact_qlike_offset(targets[calibration] - prior[calibration])
                predictions["mse_har_qlike_offset"][score] = (
                    mse_score_prediction + offset[None, :]
                )

    metadata = ["sample_id", "fold_split", "label", "episode_id", "origin_date"]
    metadata += [
        column
        for column in ("market_group", "index", "event_onset")
        if column in frame.columns
    ]
    outputs: list[pd.DataFrame] = []
    for model_name, prediction in predictions.items():
        valid = np.isfinite(prediction).all(axis=1)
        if not valid.any():
            continue
        selected = frame.loc[valid, metadata].reset_index(drop=True)
        truth = targets[valid]
        forecast = prediction[valid]
        residual = truth - forecast
        ratio = np.exp(np.clip(2.0 * residual, -40.0, 40.0))
        cells = pd.DataFrame(
            {
                "fold": np.repeat(int(fold), int(valid.sum()) * horizons),
                "model": np.repeat(model_name, int(valid.sum()) * horizons),
                "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
                "fold_split": np.repeat(selected["fold_split"].astype(str), horizons),
                "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
                "episode_id": np.repeat(selected["episode_id"].astype(str), horizons),
                "origin_date": np.repeat(selected["origin_date"].astype(str), horizons),
                "horizon": np.tile(np.arange(1, horizons + 1), int(valid.sum())),
                "fit_rows": np.repeat(fit_counts[valid], horizons),
                "calibration_rows": np.repeat(calibration_counts[valid], horizons),
                "y_true_logvol": truth.reshape(-1),
                "baseline_prediction_logvol": forecast.reshape(-1),
                "log_residual": residual.reshape(-1),
                "variance_ratio": ratio.reshape(-1),
                "qlike_score_residual": (ratio - 1.0).reshape(-1),
                "qlike_cell": qlike_logvol_cells(truth, forecast).reshape(-1),
                "squared_log_error": np.square(residual).reshape(-1),
            }
        )
        for column in ("market_group", "index", "event_onset"):
            if column in selected:
                cells[column] = np.repeat(selected[column].astype(str), horizons)
        outputs.append(cells)
    if not outputs:
        raise RuntimeError(f"fold {fold}: no model produced valid predictions")
    positive_fit = fit_counts[fit_counts > 0]
    diagnostics = {
        "fold": int(fold),
        "development_rows": int(len(frame)),
        "train_rows": int(train.sum()),
        "validation_rows": int((~train).sum()),
        "target_availability_source": source,
        "minimum_positive_fit_rows": int(positive_fit.min()) if len(positive_fit) else 0,
        "maximum_fit_rows": int(fit_counts.max()),
        "qlike_optimizer": convergence,
        "test_rows_used": 0,
    }
    return pd.concat(outputs, ignore_index=True), diagnostics


def summarize_residual_registry(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = registry.groupby(["fold", "model", "fold_split"], sort=True)
    for (fold, model, split), group in groups:
        populations = (
            ("all", group),
            ("control", group.loc[group["label"].eq(0)]),
            ("transition", group.loc[group["label"].eq(1)]),
        )
        for population, local in populations:
            if local.empty:
                continue
            rows.append(
                {
                    "fold": int(fold),
                    "model": str(model),
                    "fold_split": str(split),
                    "population": population,
                    "samples": int(local["sample_id"].nunique()),
                    "cells": int(len(local)),
                    "qlike": float(local["qlike_cell"].mean()),
                    "rmse_logvol": float(np.sqrt(local["squared_log_error"].mean())),
                    "mean_log_residual": float(local["log_residual"].mean()),
                    "mean_qlike_score_residual": float(
                        local["qlike_score_residual"].mean()
                    ),
                    "underprediction_rate": float(local["log_residual"].gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def selection_summary(
    metrics: pd.DataFrame,
    config: L5ResidualFoundationConfig,
) -> dict[str, object]:
    validation = metrics.loc[
        metrics["fold_split"].eq("val") & metrics["population"].eq("all")
    ]
    if validation.empty:
        raise ValueError("no validation metrics")

    def aggregate(folds: tuple[int, ...]) -> pd.DataFrame:
        return (
            validation.loc[validation["fold"].isin(folds)]
            .groupby("model", as_index=False)
            .agg(
                mean_qlike=("qlike", "mean"),
                mean_rmse_logvol=("rmse_logvol", "mean"),
            )
            .sort_values(["mean_qlike", "mean_rmse_logvol", "model"])
            .reset_index(drop=True)
        )

    selection = aggregate(config.selection_folds)
    confirmation = aggregate(config.confirmation_folds)
    selected_model = str(selection.iloc[0]["model"])
    baseline = confirmation.loc[confirmation["model"].eq("mse_har")]
    selected = confirmation.loc[confirmation["model"].eq(selected_model)]
    delta = None
    if not baseline.empty and not selected.empty:
        delta = {
            "qlike": float(selected.iloc[0].mean_qlike - baseline.iloc[0].mean_qlike),
            "rmse_logvol": float(
                selected.iloc[0].mean_rmse_logvol - baseline.iloc[0].mean_rmse_logvol
            ),
        }
    return {
        "selected_on_folds": list(config.selection_folds),
        "confirmed_on_folds": list(config.confirmation_folds),
        "selected_model": selected_model,
        "selection_ranking": selection.to_dict("records"),
        "confirmation_metrics": confirmation.to_dict("records"),
        "selected_minus_mse_har_confirmation": delta,
    }


def selected_residual_paths(
    registry: pd.DataFrame,
    *,
    model_name: str,
    target_horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    local = registry.loc[registry["model"].eq(model_name)].sort_values(
        ["fold", "sample_id", "horizon"]
    )
    if local.empty:
        raise ValueError(f"registry contains no model {model_name}")
    counts = local.groupby(["fold", "sample_id"])["horizon"].nunique()
    if not counts.eq(int(target_horizon)).all():
        raise ValueError("registry has incomplete paths")
    keys = local.drop_duplicates(["fold", "sample_id"])[
        ["fold", "sample_id", "fold_split", "label", "episode_id", "origin_date"]
    ].reset_index(drop=True)
    truth = local["y_true_logvol"].to_numpy().reshape(-1, target_horizon)
    prediction = local["baseline_prediction_logvol"].to_numpy().reshape(
        -1, target_horizon
    )
    residual = local["log_residual"].to_numpy().reshape(-1, target_horizon)
    if not np.allclose(truth - prediction, residual):
        raise RuntimeError("residual path identity failed")
    return keys, truth, prediction, residual


def run_l5_residual_foundation(
    *,
    fold_dir: Path,
    results_root: Path,
    config: L5ResidualFoundationConfig = L5ResidualFoundationConfig(),
    run_id: str | None = None,
) -> Path:
    from experiments.runs import begin_run
    from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
        load_rolling_fold_dataset,
    )

    config.validate()
    dataset = load_rolling_fold_dataset(Path(fold_dir))
    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "qrc_row_selection_before_baseline": False,
            "test_rows_allowed": False,
            "new_quantum_simulation": False,
        },
        run_id=run_id,
    )
    registries: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for fold in config.folds:
        registry, diagnostic = build_fold_residual_registry(
            dataset.manifest, fold=int(fold), config=config
        )
        registries.append(registry)
        diagnostics.append(diagnostic)
    registry = pd.concat(registries, ignore_index=True)
    metrics = summarize_residual_registry(registry)
    decision = selection_summary(metrics, config)
    registry.to_csv(run_dir / "residual_registry_long.csv", index=False)
    metrics.to_csv(run_dir / "residual_metrics.csv", index=False)
    (run_dir / "fold_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "l5_residual_foundation_complete",
        "financial_test_rows_used": 0,
        "baseline_fit_population": "all eligible L5 development rows before QRC selection",
        "residual_coordinate": (
            "log_residual = true_log_volatility - predicted_log_volatility; "
            "variance_ratio = exp(2 * log_residual)"
        ),
        "selection": decision,
        "downstream_rule": (
            "Join by (fold, sample_id) after baseline freeze. The second-stage intercept "
            "is prohibited because level and QLIKE calibration belong to the baseline."
        ),
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
