from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_sequence_models import metrics
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.ladder_finite_shot_sampling import shot_modes_from_probabilities
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    LadderReadoutUpgradeConfig,
    apply_calibration,
    fit_residual_correction,
    select_inner_configuration,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import load_rolling_fold_dataset

NINE_MODE_INDICES = tuple(range(9))
DENSITY_CURVATURE_INDICES = (0, 2, 3, 5, 6, 8)
PRIMARY_MODELS = ("nine_mode_ridge", "density_curvature_ridge")
QLIKE_MODELS = ("density_curvature_qlike_constrained", "density_curvature_qlike_unconstrained")


@dataclass(frozen=True)
class FinalSparseQlikeAssayConfig:
    folds: tuple[int, ...] = tuple(range(1, 9))
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    seed: int = 20260722
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    linear_alpha: float = 100.0
    qlike_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    transition_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    shot_count: int = 1000
    shot_seeds: tuple[int, ...] = (20260731, 20260732, 20260733, 20260734, 20260735)
    constrained_rmse_tolerance: float = 0.005
    constrained_control_qlike_tolerance: float = 0.005
    constrained_control_rmse_tolerance: float = 0.010

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.later_folds or set(self.later_folds).difference(self.folds):
            raise ValueError("later_folds must be a nonempty subset of folds")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive")
        if self.max_per_class < 1 or self.shot_count < 1:
            raise ValueError("sample cap and shot count must be positive")
        if not self.shot_seeds or len(set(self.shot_seeds)) != len(self.shot_seeds):
            raise ValueError("shot seeds must be nonempty and unique")
        if self.linear_alpha <= 0 or not self.qlike_alphas or any(value <= 0 for value in self.qlike_alphas):
            raise ValueError("ridge and QLIKE alphas must be positive")
        if min(
            self.constrained_rmse_tolerance,
            self.constrained_control_qlike_tolerance,
            self.constrained_control_rmse_tolerance,
        ) < 0:
            raise ValueError("QLIKE constraints must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def readout_config(self) -> LadderReadoutUpgradeConfig:
        return LadderReadoutUpgradeConfig(
            folds=self.folds,
            leads=self.leads,
            max_per_class=self.max_per_class,
            prequential_blocks=self.prequential_blocks,
            inner_holdout_fraction=self.inner_holdout_fraction,
            split_horizon=self.split_horizon,
            early_lambdas=self.early_lambdas,
            transition_lambdas=self.transition_lambdas,
            linear_alpha=self.linear_alpha,
            seed=self.seed,
        )


def qlike_loss(y_true: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    difference = 2.0 * (observed - forecast)
    return float(np.mean(np.exp(np.clip(difference, -50.0, 50.0)) - difference - 1.0))


def _metric(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(y),) or not selected.any():
        return float("nan"), float("nan")
    qlike, rmse = metrics(y, prediction, selected)
    return float(qlike), float(rmse)


def _calibration_pairs(config: FinalSparseQlikeAssayConfig) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(early), float(late))
        for early in config.early_lambdas
        for late in config.transition_lambdas
        if float(late) >= float(early)
    )


def _standardized_design(matrix: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    features = np.asarray(matrix, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    scaler = StandardScaler().fit(features[fit])
    standardized = scaler.transform(features)
    return np.column_stack([np.ones(len(standardized)), standardized]), scaler


def fit_qlike_correction(
    matrix: np.ndarray,
    *,
    y: np.ndarray,
    prequential_har: np.ndarray,
    fit_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a multi-horizon linear correction under QLIKE with coefficient L2 penalty."""
    fit = np.asarray(fit_mask, dtype=bool)
    design, _ = _standardized_design(matrix, fit)
    correction = np.empty_like(y, dtype=float)
    penalty = float(alpha)
    for horizon in range(y.shape[1]):
        local_design = design[fit]
        observed = np.asarray(y[fit, horizon], dtype=float)
        baseline = np.asarray(prequential_har[fit, horizon], dtype=float)
        initial = Ridge(alpha=penalty, fit_intercept=False).fit(
            local_design,
            observed - baseline,
        ).coef_

        def objective(coefficients: np.ndarray) -> float:
            forecast = baseline + local_design @ coefficients
            difference = 2.0 * (observed - forecast)
            loss = np.mean(np.exp(np.clip(difference, -50.0, 50.0)) - difference - 1.0)
            regularizer = penalty * float(np.dot(coefficients[1:], coefficients[1:])) / (2.0 * len(observed))
            return float(loss + regularizer)

        def gradient(coefficients: np.ndarray) -> np.ndarray:
            forecast = baseline + local_design @ coefficients
            difference = 2.0 * (observed - forecast)
            exponential = np.exp(np.clip(difference, -50.0, 50.0))
            values = local_design.T @ (2.0 * (1.0 - exponential)) / len(observed)
            values[1:] += penalty * coefficients[1:] / len(observed)
            return np.asarray(values, dtype=float)

        result = minimize(
            objective,
            np.asarray(initial, dtype=float),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"QLIKE optimization failed at horizon {horizon + 1}: {result.message}")
        correction[:, horizon] = design @ result.x
    return correction


def _ridge_inner_reference(
    matrix: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train: np.ndarray,
    origin_date: np.ndarray,
    labels: np.ndarray,
    config: FinalSparseQlikeAssayConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float]]:
    readout_config = config.readout_config()
    diagnostics, _, full_correction, _ = select_inner_configuration(
        matrix,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train,
        origin_date=origin_date,
        readout_kind="linear",
        calibration_kind="segmented",
        config=readout_config,
    )
    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        residual_train,
        holdout_fraction=config.inner_holdout_fraction,
    )
    inner_correction, _ = fit_residual_correction(
        matrix,
        residuals,
        inner_fit,
        readout_kind="linear",
        ridge_alpha=float(diagnostics["ridge_alpha"]),
    )
    inner_prediction = apply_calibration(
        har,
        inner_correction,
        early_lambda=float(diagnostics["early_lambda"]),
        transition_lambda=float(diagnostics["transition_lambda"]),
        split_horizon=config.split_horizon,
    )
    control_tune = inner_tune & (np.asarray(labels, dtype=int) == 0)
    qlike, rmse = _metric(y, inner_prediction, inner_tune)
    control_qlike, control_rmse = _metric(y, inner_prediction, control_tune)
    reference = {
        "inner_qlike": qlike,
        "inner_rmse": rmse,
        "inner_control_qlike": control_qlike,
        "inner_control_rmse": control_rmse,
    }
    return diagnostics, full_correction, inner_tune, reference


def select_qlike_configuration(
    matrix: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train: np.ndarray,
    origin_date: np.ndarray,
    labels: np.ndarray,
    config: FinalSparseQlikeAssayConfig,
    constrained: bool,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        residual_train,
        holdout_fraction=config.inner_holdout_fraction,
    )
    ridge_diagnostics, ridge_full_correction, _, reference = _ridge_inner_reference(
        matrix,
        y=y,
        har=har,
        residuals=residuals,
        residual_train=residual_train,
        origin_date=origin_date,
        labels=labels,
        config=config,
    )
    prequential_har = y - residuals
    control_tune = inner_tune & (np.asarray(labels, dtype=int) == 0)
    rows: list[dict[str, object]] = []
    for alpha in config.qlike_alphas:
        correction = fit_qlike_correction(
            matrix,
            y=y,
            prequential_har=prequential_har,
            fit_mask=inner_fit,
            alpha=float(alpha),
        )
        for early, late in _calibration_pairs(config):
            prediction = apply_calibration(
                har,
                correction,
                early_lambda=early,
                transition_lambda=late,
                split_horizon=config.split_horizon,
            )
            qlike, rmse = _metric(y, prediction, inner_tune)
            control_qlike, control_rmse = _metric(y, prediction, control_tune)
            eligible = bool(
                not constrained
                or (
                    rmse <= reference["inner_rmse"] + config.constrained_rmse_tolerance
                    and control_qlike
                    <= reference["inner_control_qlike"] + config.constrained_control_qlike_tolerance
                    and control_rmse
                    <= reference["inner_control_rmse"] + config.constrained_control_rmse_tolerance
                )
            )
            rows.append(
                {
                    "alpha": float(alpha),
                    "early_lambda": early,
                    "transition_lambda": late,
                    "inner_qlike": qlike,
                    "inner_rmse": rmse,
                    "inner_control_qlike": control_qlike,
                    "inner_control_rmse": control_rmse,
                    "eligible": eligible,
                }
            )
    eligible_rows = [row for row in rows if bool(row["eligible"])]
    if constrained and not eligible_rows:
        diagnostics: dict[str, object] = {
            "selected_kind": "ridge_fallback",
            "alpha": float(ridge_diagnostics["ridge_alpha"]),
            "early_lambda": float(ridge_diagnostics["early_lambda"]),
            "transition_lambda": float(ridge_diagnostics["transition_lambda"]),
            **reference,
        }
        return diagnostics, pd.DataFrame(rows), ridge_full_correction

    selected = min(
        eligible_rows,
        key=lambda row: (
            float(row["inner_qlike"]),
            float(row["inner_rmse"]),
            float(row["alpha"]),
            float(row["early_lambda"] + row["transition_lambda"]),
        ),
    )
    alpha = float(selected["alpha"])
    full_correction = fit_qlike_correction(
        matrix,
        y=y,
        prequential_har=prequential_har,
        fit_mask=residual_train,
        alpha=alpha,
    )
    diagnostics = {"selected_kind": "qlike", **selected, **reference}
    return diagnostics, pd.DataFrame(rows), full_correction


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    validation: np.ndarray,
    model_name: str,
    protocol: str,
    shots: int,
    shot_seed: int,
    diagnostics: dict[str, object],
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = y.shape[1]
    count = int(validation.sum()) * horizons
    return pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(selected["sample_id"].astype(str).to_numpy(), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "episode_id": np.repeat(selected["episode_id"].astype(str).to_numpy(), horizons),
            "origin_date": np.repeat(selected["origin_date"].astype(str).to_numpy(), horizons),
            "model_name": np.repeat(model_name, count),
            "protocol": np.repeat(protocol, count),
            "shots": np.repeat(int(shots), count),
            "shot_seed": np.repeat(int(shot_seed), count),
            "selected_kind": np.repeat(str(diagnostics.get("selected_kind", "ridge")), count),
            "alpha": np.repeat(float(diagnostics.get("alpha", diagnostics.get("ridge_alpha", np.nan))), count),
            "early_lambda": np.repeat(float(diagnostics["early_lambda"]), count),
            "transition_lambda": np.repeat(float(diagnostics["transition_lambda"]), count),
            "horizon": np.tile(np.arange(1, horizons + 1), int(validation.sum())),
            "y_true": y[validation].reshape(-1),
            "y_pred": prediction[validation].reshape(-1),
            "har_pred": har[validation].reshape(-1),
        }
    )


def _metric_tables(predictions: pd.DataFrame, later_folds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    keys = ["model_name", "protocol", "shots", "shot_seed"]
    for values, group in predictions.groupby(keys, sort=True):
        scopes = {
            "overall": np.ones(len(group), dtype=bool),
            "later_overall": group["fold"].isin(later_folds).to_numpy(),
            "later_l1_transition": (
                group["fold"].isin(later_folds) & group["lead"].eq(1) & group["label"].eq(1)
            ).to_numpy(),
            "later_l5_transition": (
                group["fold"].isin(later_folds) & group["lead"].eq(5) & group["label"].eq(1)
            ).to_numpy(),
            "later_l10_transition": (
                group["fold"].isin(later_folds) & group["lead"].eq(10) & group["label"].eq(1)
            ).to_numpy(),
            "later_controls": (
                group["fold"].isin(later_folds) & group["label"].eq(0)
            ).to_numpy(),
        }
        for scope, mask in scopes.items():
            local = group.loc[mask]
            if local.empty:
                continue
            observed = local["y_true"].to_numpy(dtype=float)[:, None]
            forecast = local["y_pred"].to_numpy(dtype=float)[:, None]
            qlike, rmse = metrics(observed, forecast, np.ones(len(local), dtype=bool))
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "scope": scope,
                    "prediction_rows": int(len(local)),
                    "qlike": float(qlike),
                    "rmse": float(rmse),
                    "mean_error": float(np.mean(forecast - observed)),
                }
            )
    metrics_frame = pd.DataFrame(rows)
    shot = metrics_frame.loc[metrics_frame["protocol"].eq("shot_consistent")]
    summary_rows: list[dict[str, object]] = []
    for values, group in shot.groupby(["model_name", "shots", "scope"], sort=True):
        row = dict(zip(["model_name", "shots", "scope"], values, strict=True))
        row["seeds"] = int(group["shot_seed"].nunique())
        for metric_name in ("qlike", "rmse", "mean_error"):
            series = group[metric_name].astype(float)
            row[f"{metric_name}_median"] = float(series.median())
            row[f"{metric_name}_minimum"] = float(series.min())
            row[f"{metric_name}_maximum"] = float(series.max())
        summary_rows.append(row)
    return metrics_frame, pd.DataFrame(summary_rows)


def run_final_sparse_qlike_assay(
    *,
    fold_dir: Path,
    source_spacing_run: Path,
    results_root: Path,
    config: FinalSparseQlikeAssayConfig = FinalSparseQlikeAssayConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    readout_config = config.readout_config()
    readout_config.validate()
    source_spacing_run = Path(source_spacing_run)
    cache_root = source_spacing_run / "probability_cache" / "row_9p0um"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"missing frozen 9.0 um probability cache: {cache_root}")
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "source_spacing_run": str(source_spacing_run),
            "assay": config.to_dict(),
            "readout_families": {
                "nine_modes": list(NINE_MODE_INDICES),
                "density_curvature": list(DENSITY_CURVATURE_INDICES),
            },
            "qlike_specialist_role": "secondary diagnostic; cannot replace balanced ridge primary",
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )
    dataset = load_rolling_fold_dataset(fold_dir)
    predictions: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("final sparse assay must not receive test rows")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )
        cache_path = cache_root / f"fold_{int(fold)}.npz"
        with np.load(cache_path, allow_pickle=True) as bundle:
            probabilities = np.asarray(bundle["probabilities"], dtype=float)
            exact_modes = np.asarray(bundle["exact_modes"], dtype=float)
            cache_ids = np.asarray(bundle["sample_id"]).astype(str)
            probe_steps = tuple(int(value) for value in np.asarray(bundle["probe_steps"]))
        selected_ids = frame["sample_id"].astype(str).to_numpy()
        if not np.array_equal(cache_ids, selected_ids):
            raise RuntimeError(f"fold {fold}: spacing cache and selected panel differ")
        matrices: list[tuple[str, int, int, np.ndarray]] = [("exact", 0, -1, exact_modes)]
        for shot_seed in config.shot_seeds:
            sampled = shot_modes_from_probabilities(
                probabilities,
                shots=config.shot_count,
                base_seed=int(shot_seed),
                fold=int(fold),
                sample_ids=selected_ids,
                probe_steps=probe_steps,
            )
            matrices.append(("shot_consistent", config.shot_count, int(shot_seed), sampled))

        for protocol, shots, shot_seed, full_matrix in matrices:
            for model_name, indices in (
                ("nine_mode_ridge", NINE_MODE_INDICES),
                ("density_curvature_ridge", DENSITY_CURVATURE_INDICES),
            ):
                matrix = full_matrix[:, indices]
                diagnostics, candidate_frame, correction, _ = select_inner_configuration(
                    matrix,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train,
                    origin_date=frame["origin_date"].astype(str).to_numpy(),
                    readout_kind="linear",
                    calibration_kind="segmented",
                    config=readout_config,
                )
                prediction = apply_calibration(
                    har,
                    correction,
                    early_lambda=float(diagnostics["early_lambda"]),
                    transition_lambda=float(diagnostics["transition_lambda"]),
                    split_horizon=config.split_horizon,
                )
                diagnostics = {"selected_kind": "ridge", **diagnostics}
                predictions.append(
                    _prediction_rows(
                        frame,
                        y=y,
                        prediction=prediction,
                        har=har,
                        validation=validation,
                        model_name=model_name,
                        protocol=protocol,
                        shots=shots,
                        shot_seed=shot_seed,
                        diagnostics=diagnostics,
                    )
                )
                candidate_frame = candidate_frame.copy()
                candidate_frame.insert(0, "fold", int(fold))
                candidate_frame.insert(1, "model_name", model_name)
                candidate_frame.insert(2, "protocol", protocol)
                candidate_frame.insert(3, "shots", shots)
                candidate_frame.insert(4, "shot_seed", shot_seed)
                candidates.append(candidate_frame)

            sparse_matrix = full_matrix[:, DENSITY_CURVATURE_INDICES]
            for model_name, constrained in (
                ("density_curvature_qlike_constrained", True),
                ("density_curvature_qlike_unconstrained", False),
            ):
                diagnostics, candidate_frame, correction = select_qlike_configuration(
                    sparse_matrix,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train=residual_train,
                    origin_date=frame["origin_date"].astype(str).to_numpy(),
                    labels=frame["label"].to_numpy(dtype=int),
                    config=config,
                    constrained=constrained,
                )
                prediction = apply_calibration(
                    har,
                    correction,
                    early_lambda=float(diagnostics["early_lambda"]),
                    transition_lambda=float(diagnostics["transition_lambda"]),
                    split_horizon=config.split_horizon,
                )
                predictions.append(
                    _prediction_rows(
                        frame,
                        y=y,
                        prediction=prediction,
                        har=har,
                        validation=validation,
                        model_name=model_name,
                        protocol=protocol,
                        shots=shots,
                        shot_seed=shot_seed,
                        diagnostics=diagnostics,
                    )
                )
                candidate_frame = candidate_frame.copy()
                candidate_frame.insert(0, "fold", int(fold))
                candidate_frame.insert(1, "model_name", model_name)
                candidate_frame.insert(2, "protocol", protocol)
                candidate_frame.insert(3, "shots", shots)
                candidate_frame.insert(4, "shot_seed", shot_seed)
                candidates.append(candidate_frame)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    metric_frame, shot_summary = _metric_tables(prediction_frame, config.later_folds)
    exact = metric_frame.loc[metric_frame["protocol"].eq("exact")]
    recommendation = {
        "balanced_primary_candidates": list(PRIMARY_MODELS),
        "balanced_primary_candidate_under_test": "density_curvature_ridge",
        "qlike_specialist": "density_curvature_qlike_constrained",
        "unconstrained_qlike_status": "diagnostic_only_due_to_asymmetric upward-bias risk",
        "selection_note": (
            "Confirm the sparse ridge only if exact and 1000-shot QLIKE gains persist within "
            "the predeclared RMSE and control tolerances. The constrained QLIKE head is a "
            "secondary metric specialist and cannot replace the balanced primary."
        ),
        "exact_metrics": exact.to_dict(orient="records"),
    }
    prediction_frame.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    candidate_frame.to_csv(run_dir / "inner_candidates.csv.gz", index=False, compression="gzip")
    metric_frame.to_csv(run_dir / "metrics_by_scope_and_seed.csv", index=False)
    shot_summary.to_csv(run_dir / "shot_summary.csv", index=False)
    (run_dir / "recommendation.json").write_text(json.dumps(recommendation, indent=2) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "final_sparse_and_qlike_assay_complete",
                "folds": list(config.folds),
                "shot_count": config.shot_count,
                "shot_seeds": list(config.shot_seeds),
                "test_evaluated": False,
                "models": [*PRIMARY_MODELS, *QLIKE_MODELS],
            },
            indent=2,
        )
        + "\n"
    )
    return run_dir
