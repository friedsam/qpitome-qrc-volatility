from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.control_strata import CALM, HARD_NEGATIVE, TRANSITION
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_sequence_models import metrics
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.ladder_finite_shot_sampling import load_probability_cache
from transition_forecasting.qrc.representation_screen_analysis import _parse_mixed_utc
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import load_rolling_fold_dataset

ReadoutKind = Literal["linear", "polynomial_degree2"]
HarScope = Literal["transition_calm"]

MODEL_SPECS: tuple[tuple[str, ReadoutKind], ...] = (
    ("ladder_linear_transition_emphasis", "linear"),
    ("ladder_poly2_transition_emphasis", "polynomial_degree2"),
)


@dataclass(frozen=True)
class TransitionSignalAssayConfig:
    """Exact-feature readout assay that prioritizes transition L5 signal.

    Hard negatives are retained as untouched validation diagnostics but are excluded
    from HAR-residual construction, ridge fitting, and calibration selection. Calm
    matched controls anchor the correction close to HAR. Hyperparameters are selected
    on a chronological inner holdout only.
    """

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    split_horizon: int = 4
    transition_weights: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    fit_intercepts: tuple[bool, ...] = (False, True)
    linear_alphas: tuple[float, ...] = (100.0,)
    polynomial_alphas: tuple[float, ...] = (100.0, 1000.0, 10000.0)
    early_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    late_lambdas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    )
    calm_qlike_tolerance: float = 0.02
    calm_rmse_tolerance: float = 0.02
    transition_l1_qlike_tolerance: float = 0.05
    seed: int = 20260722
    har_scope: HarScope = "transition_calm"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive and nonempty")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not 1 <= self.split_horizon < len(TARGET_COLUMNS):
            raise ValueError("split_horizon must lie inside the target path")
        if not self.transition_weights or any(value <= 0 for value in self.transition_weights):
            raise ValueError("transition_weights must be positive")
        if not self.fit_intercepts:
            raise ValueError("fit_intercepts cannot be empty")
        if not self.linear_alphas or any(value <= 0 for value in self.linear_alphas):
            raise ValueError("linear_alphas must be positive")
        if not self.polynomial_alphas or any(value <= 0 for value in self.polynomial_alphas):
            raise ValueError("polynomial_alphas must be positive")
        if not self.early_lambdas or any(value < 0 for value in self.early_lambdas):
            raise ValueError("early_lambdas must be nonnegative")
        if not self.late_lambdas or any(value < 0 for value in self.late_lambdas):
            raise ValueError("late_lambdas must be nonnegative")
        if min(
            self.calm_qlike_tolerance,
            self.calm_rmse_tolerance,
            self.transition_l1_qlike_tolerance,
        ) < 0:
            raise ValueError("selection tolerances must be nonnegative")
        if self.har_scope != "transition_calm":
            raise ValueError("the frozen assay supports transition_calm HAR scope only")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _strata(frame: pd.DataFrame) -> np.ndarray:
    if "evaluation_stratum" not in frame.columns:
        raise ValueError("deterministic manifest requires evaluation_stratum")
    values = frame["evaluation_stratum"].astype(str).to_numpy()
    unknown = set(values).difference({TRANSITION, CALM, HARD_NEGATIVE})
    if unknown:
        raise ValueError(f"unexpected evaluation strata: {sorted(unknown)}")
    return values


def transition_calm_mask(frame: pd.DataFrame) -> np.ndarray:
    values = _strata(frame)
    return np.isin(values, [TRANSITION, CALM])


def _fit_har_scope(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    fit = np.asarray(train_mask, dtype=bool) & np.asarray(eligible_mask, dtype=bool)
    if fit.sum() < max(10, len(HAR_FEATURES) + 2):
        raise ValueError(f"HAR fit is too small after scope restriction: {int(fit.sum())}")
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[fit]), y[fit])
    return np.asarray(model.predict(scaler.transform(x)), dtype=float)


def _prequential_har_residuals_scope(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    dates = _parse_mixed_utc(frame["origin_date"])
    train = np.asarray(train_mask, dtype=bool)
    eligible = np.asarray(eligible_mask, dtype=bool)
    scoped_train = train & eligible
    train_dates = np.asarray(sorted(dates[scoped_train].unique()))
    chunks = [
        chunk
        for chunk in np.array_split(train_dates, min(int(blocks), len(train_dates)))
        if len(chunk)
    ]
    predictions = np.full_like(y, np.nan, dtype=float)
    minimum_fit = max(10, len(HAR_FEATURES) + 2)
    for block_index in range(1, len(chunks)):
        score_dates = chunks[block_index]
        first_score_date = score_dates[0]
        fit = scoped_train & dates.lt(first_score_date).to_numpy()
        score = scoped_train & dates.isin(score_dates).to_numpy()
        if fit.sum() < minimum_fit or not score.any():
            continue
        scaler = StandardScaler()
        model = Ridge(alpha=100.0)
        model.fit(scaler.fit_transform(x[fit]), y[fit])
        predictions[score] = model.predict(scaler.transform(x[score]))
    valid = scoped_train & np.isfinite(predictions).all(axis=1)
    return y - predictions, valid


def _design_matrix(
    matrix: np.ndarray,
    fit_mask: np.ndarray,
    *,
    readout_kind: ReadoutKind,
) -> tuple[np.ndarray, int]:
    features = np.asarray(matrix, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2 or fit.shape != (len(features),) or not fit.any():
        raise ValueError("feature matrix and fit mask are not aligned")
    raw_scaler = StandardScaler().fit(features[fit])
    standardized = raw_scaler.transform(features)
    if readout_kind == "linear":
        return standardized, int(standardized.shape[1])
    if readout_kind != "polynomial_degree2":
        raise ValueError(f"unsupported readout_kind: {readout_kind}")
    expansion = PolynomialFeatures(degree=2, include_bias=False)
    expanded = expansion.fit_transform(standardized)
    expanded_scaler = StandardScaler().fit(expanded[fit])
    return expanded_scaler.transform(expanded), int(expanded.shape[1])


def _fit_weighted_correction(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    strata: np.ndarray,
    *,
    readout_kind: ReadoutKind,
    ridge_alpha: float,
    transition_weight: float,
    fit_intercept: bool,
) -> tuple[np.ndarray, int]:
    fit = np.asarray(fit_mask, dtype=bool)
    design, width = _design_matrix(matrix, fit, readout_kind=readout_kind)
    weights = np.ones(int(fit.sum()), dtype=float)
    weights[np.asarray(strata)[fit] == TRANSITION] = float(transition_weight)
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=bool(fit_intercept))
    model.fit(design[fit], np.asarray(residuals, dtype=float)[fit], sample_weight=weights)
    correction = np.asarray(model.predict(design), dtype=float)
    if correction.shape != np.asarray(residuals).shape or not np.isfinite(correction).all():
        raise RuntimeError("weighted residual head returned invalid corrections")
    return correction, width


def apply_segmented_calibration(
    har: np.ndarray,
    correction: np.ndarray,
    *,
    early_lambda: float,
    late_lambda: float,
    split_horizon: int,
) -> np.ndarray:
    scale = np.full(np.asarray(har).shape[1], float(late_lambda), dtype=float)
    scale[: int(split_horizon)] = float(early_lambda)
    return np.asarray(har, dtype=float) + np.asarray(correction, dtype=float) * scale[None, :]


def _metric(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(y),) or not selected.any():
        return float("nan"), float("nan")
    qlike, rmse = metrics(np.asarray(y), np.asarray(prediction), selected)
    return float(qlike), float(rmse)


def _candidate_metrics(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    tune_mask: np.ndarray,
) -> dict[str, float]:
    strata = _strata(frame)
    lead = frame["lead"].to_numpy(dtype=int)
    tune = np.asarray(tune_mask, dtype=bool)
    masks = {
        "transition_all": tune & (strata == TRANSITION),
        "transition_l1": tune & (strata == TRANSITION) & (lead == 1),
        "transition_l5": tune & (strata == TRANSITION) & (lead == 5),
        "calm": tune & (strata == CALM),
        "hard_negative": tune & (strata == HARD_NEGATIVE),
    }
    payload: dict[str, float] = {}
    final_correction = np.asarray(prediction) - np.asarray(har)
    for name, mask in masks.items():
        pred_qlike, pred_rmse = _metric(y, prediction, mask)
        har_qlike, har_rmse = _metric(y, har, mask)
        payload[f"{name}_qlike"] = pred_qlike
        payload[f"{name}_rmse"] = pred_rmse
        payload[f"{name}_har_qlike"] = har_qlike
        payload[f"{name}_har_rmse"] = har_rmse
        payload[f"{name}_qlike_gain"] = har_qlike - pred_qlike
        payload[f"{name}_qlike_delta"] = pred_qlike - har_qlike
        payload[f"{name}_rmse_delta"] = pred_rmse - har_rmse
        payload[f"{name}_correction_mae"] = (
            float(np.mean(np.abs(final_correction[mask]))) if mask.any() else float("nan")
        )
        payload[f"{name}_raw_correction_mae"] = (
            float(np.mean(np.abs(np.asarray(correction)[mask]))) if mask.any() else float("nan")
        )
    payload["transition_calm_margin"] = (
        payload["transition_l5_correction_mae"] - payload["calm_correction_mae"]
    )
    return payload


def candidate_is_eligible(row: dict[str, object], config: TransitionSignalAssayConfig) -> bool:
    required = (
        "transition_l5_qlike_gain",
        "calm_qlike_delta",
        "calm_rmse_delta",
        "transition_l1_qlike_delta",
    )
    if any(not np.isfinite(float(row[name])) for name in required):
        return False
    return bool(
        float(row["calm_qlike_delta"]) <= config.calm_qlike_tolerance
        and float(row["calm_rmse_delta"]) <= config.calm_rmse_tolerance
        and float(row["transition_l1_qlike_delta"])
        <= config.transition_l1_qlike_tolerance
    )


def candidate_key(row: dict[str, object]) -> tuple[float, ...]:
    """Lexicographic key: admissibility, L5 signal, then calm restraint."""
    return (
        0.0 if bool(row["eligible"]) else 1.0,
        -float(row["transition_l5_qlike_gain"]),
        -float(row["transition_all_qlike_gain"]),
        -float(row["transition_calm_margin"]),
        float(row["calm_qlike_delta"]),
        float(row["calm_correction_mae"]),
        0.0 if not bool(row["fit_intercept"]) else 1.0,
        float(row["ridge_alpha"]),
        float(row["transition_weight"]),
        float(row["early_lambda"] + row["late_lambda"]),
    )


def select_transition_signal_configuration(
    matrix: np.ndarray,
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    readout_kind: ReadoutKind,
    config: TransitionSignalAssayConfig,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, int]:
    strata = _strata(frame)
    eligible = np.asarray(residual_train_mask, dtype=bool) & (strata != HARD_NEGATIVE)
    inner_fit, inner_tune = chronological_inner_split(
        frame["origin_date"].astype(str).to_numpy(),
        eligible,
        holdout_fraction=config.inner_holdout_fraction,
    )
    alphas = config.linear_alphas if readout_kind == "linear" else config.polynomial_alphas
    rows: list[dict[str, object]] = []
    for transition_weight in config.transition_weights:
        for alpha in alphas:
            for fit_intercept in config.fit_intercepts:
                correction, width = _fit_weighted_correction(
                    matrix,
                    residuals,
                    inner_fit,
                    strata,
                    readout_kind=readout_kind,
                    ridge_alpha=float(alpha),
                    transition_weight=float(transition_weight),
                    fit_intercept=bool(fit_intercept),
                )
                for early_lambda in config.early_lambdas:
                    for late_lambda in config.late_lambdas:
                        if float(late_lambda) < float(early_lambda):
                            continue
                        prediction = apply_segmented_calibration(
                            har,
                            correction,
                            early_lambda=float(early_lambda),
                            late_lambda=float(late_lambda),
                            split_horizon=config.split_horizon,
                        )
                        row: dict[str, object] = {
                            "transition_weight": float(transition_weight),
                            "ridge_alpha": float(alpha),
                            "fit_intercept": bool(fit_intercept),
                            "early_lambda": float(early_lambda),
                            "late_lambda": float(late_lambda),
                            "inner_fit_rows": int(inner_fit.sum()),
                            "inner_tune_rows": int(inner_tune.sum()),
                            "design_feature_width": int(width),
                            **_candidate_metrics(
                                frame=frame,
                                y=y,
                                har=har,
                                prediction=prediction,
                                correction=correction,
                                tune_mask=inner_tune,
                            ),
                        }
                        row["eligible"] = candidate_is_eligible(row, config)
                        rows.append(row)
    if not rows:
        raise RuntimeError("transition-signal candidate grid is empty")
    selected = min(rows, key=candidate_key)
    full_correction, width = _fit_weighted_correction(
        matrix,
        residuals,
        eligible,
        strata,
        readout_kind=readout_kind,
        ridge_alpha=float(selected["ridge_alpha"]),
        transition_weight=float(selected["transition_weight"]),
        fit_intercept=bool(selected["fit_intercept"]),
    )
    diagnostics = {
        **selected,
        "full_fit_rows": int(eligible.sum()),
        "hard_negative_fit_rows": int((eligible & (strata == HARD_NEGATIVE)).sum()),
        "design_feature_width": int(width),
        "selection_key": list(candidate_key(selected)),
    }
    return diagnostics, pd.DataFrame(rows), full_correction, int(width)


def evaluate_exact_matrix(
    matrix: np.ndarray,
    *,
    frame: pd.DataFrame,
    config: TransitionSignalAssayConfig,
    model_name: str,
    readout_kind: ReadoutKind,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if not train.any() or not validation.any():
        raise RuntimeError("selected frame requires nonempty train and validation rows")
    scope = transition_calm_mask(frame)
    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har_scope(frame, y, train, scope)
    residuals, residual_train = _prequential_har_residuals_scope(
        frame,
        y,
        train,
        scope,
        blocks=config.prequential_blocks,
    )
    diagnostics, candidates, correction, width = select_transition_signal_configuration(
        matrix,
        frame=frame,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train,
        readout_kind=readout_kind,
        config=config,
    )
    prediction = apply_segmented_calibration(
        har,
        correction,
        early_lambda=float(diagnostics["early_lambda"]),
        late_lambda=float(diagnostics["late_lambda"]),
        split_horizon=config.split_horizon,
    )
    validation_metrics = _candidate_metrics(
        frame=frame,
        y=y,
        har=har,
        prediction=prediction,
        correction=correction,
        tune_mask=validation,
    )
    metric_row: dict[str, object] = {
        "fold": int(frame["fold"].iloc[0]),
        "model_name": model_name,
        "readout_kind": readout_kind,
        "validation_rows": int(validation.sum()),
        "hard_negative_validation_rows": int(
            (validation & (_strata(frame) == HARD_NEGATIVE)).sum()
        ),
        "ridge_alpha": float(diagnostics["ridge_alpha"]),
        "transition_weight": float(diagnostics["transition_weight"]),
        "fit_intercept": bool(diagnostics["fit_intercept"]),
        "early_lambda": float(diagnostics["early_lambda"]),
        "late_lambda": float(diagnostics["late_lambda"]),
        "design_feature_width": int(width),
        "selected_inner_eligible": bool(diagnostics["eligible"]),
        **validation_metrics,
    }
    candidates = candidates.copy()
    candidates.insert(0, "fold", int(frame["fold"].iloc[0]))
    candidates.insert(1, "model_name", model_name)
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = len(TARGET_COLUMNS)
    predictions = pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(selected["sample_id"].astype(str).to_numpy(), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "evaluation_stratum": np.repeat(
                selected["evaluation_stratum"].astype(str).to_numpy(), horizons
            ),
            "episode_id": np.repeat(selected["episode_id"].astype(str).to_numpy(), horizons),
            "origin_date": np.repeat(selected["origin_date"].astype(str).to_numpy(), horizons),
            "model_name": np.repeat(model_name, int(validation.sum()) * horizons),
            "horizon": np.tile(np.arange(1, horizons + 1), int(validation.sum())),
            "y_true": y[validation].reshape(-1),
            "y_pred": prediction[validation].reshape(-1),
            "har_pred": har[validation].reshape(-1),
            "qrc_correction": (prediction - har)[validation].reshape(-1),
            "raw_qrc_correction": correction[validation].reshape(-1),
        }
    )
    return metric_row, candidates, predictions


def _pooled_predictions_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model_name", "evaluation_stratum", "lead"]
    for values, group in predictions.groupby(group_columns, sort=True):
        observed = group["y_true"].to_numpy(dtype=float)[:, None]
        forecast = group["y_pred"].to_numpy(dtype=float)[:, None]
        baseline = group["har_pred"].to_numpy(dtype=float)[:, None]
        mask = np.ones(len(group), dtype=bool)
        qlike, rmse = metrics(observed, forecast, mask)
        har_qlike, har_rmse = metrics(observed, baseline, mask)
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "prediction_rows": int(len(group)),
                "qlike": float(qlike),
                "rmse": float(rmse),
                "har_qlike": float(har_qlike),
                "har_rmse": float(har_rmse),
                "qlike_gain": float(har_qlike - qlike),
                "rmse_delta": float(rmse - har_rmse),
                "correction_mae": float(group["qrc_correction"].abs().mean()),
                "raw_correction_mae": float(group["raw_qrc_correction"].abs().mean()),
                "mean_correction": float(group["qrc_correction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _freeze_recommendation(metrics_frame: pd.DataFrame, config: TransitionSignalAssayConfig) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for model_name, group in metrics_frame.groupby("model_name", sort=True):
        def one(stratum: str, lead: int) -> pd.Series:
            rows = group.loc[
                group["evaluation_stratum"].eq(stratum) & group["lead"].eq(int(lead))
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    f"missing pooled metric for {model_name} {stratum} L{lead}: {len(rows)}"
                )
            return rows.iloc[0]

        t5 = one(TRANSITION, 5)
        t1 = one(TRANSITION, 1)
        calm = group.loc[group["evaluation_stratum"].eq(CALM)]
        hard = group.loc[group["evaluation_stratum"].eq(HARD_NEGATIVE)]
        calm_qlike_delta = float(
            np.average(calm["qlike"] - calm["har_qlike"], weights=calm["prediction_rows"])
        )
        calm_correction = float(
            np.average(calm["correction_mae"], weights=calm["prediction_rows"])
        )
        hard_qlike_delta = float(
            np.average(hard["qlike"] - hard["har_qlike"], weights=hard["prediction_rows"])
        )
        transition_correction = float(t5["correction_mae"])
        eligible = bool(
            calm_qlike_delta <= config.calm_qlike_tolerance
            and float(t1["qlike"] - t1["har_qlike"])
            <= config.transition_l1_qlike_tolerance
        )
        candidates.append(
            {
                "model_name": str(model_name),
                "eligible": eligible,
                "transition_l5_qlike_gain": float(t5["qlike_gain"]),
                "transition_l1_qlike_gain": float(t1["qlike_gain"]),
                "calm_qlike_delta": calm_qlike_delta,
                "hard_negative_qlike_delta": hard_qlike_delta,
                "transition_l5_correction_mae": transition_correction,
                "calm_correction_mae": calm_correction,
                "transition_calm_correction_margin": transition_correction - calm_correction,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            0 if bool(row["eligible"]) else 1,
            -float(row["transition_l5_qlike_gain"]),
            -float(row["transition_calm_correction_margin"]),
            float(row["calm_qlike_delta"]),
        ),
    )
    return {
        "selection_rule": (
            "maximize transition L5 QLIKE gain subject to calm and L1 tolerances; "
            "hard negatives are diagnostics only"
        ),
        "selected_model": selected["model_name"],
        "selected_eligible": bool(selected["eligible"]),
        "selected": selected,
        "candidates": candidates,
        "freeze_allowed": bool(selected["eligible"])
        and float(selected["transition_l5_qlike_gain"]) > 0.0,
    }


def _load_source_parameters(source_run: Path) -> dict[str, object]:
    payload = json.loads((Path(source_run) / "params.json").read_text())
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("source params.json lacks parameters object")
    return parameters


def run_transition_signal_readout_assay(
    *,
    source_run: Path,
    results_root: Path,
    config: TransitionSignalAssayConfig = TransitionSignalAssayConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    source_run = Path(source_run)
    source_parameters = _load_source_parameters(source_run)
    fold_dir = Path(str(source_parameters["fold_dir"]))
    dataset = load_rolling_fold_dataset(fold_dir)
    cache_manifest = pd.read_csv(source_run / "cache_manifest.csv")
    run_dir = begin_run(
        results_root,
        {
            "source_run": str(source_run),
            "source_cache_manifest": str(source_run / "cache_manifest.csv"),
            "fold_dir": str(fold_dir),
            "assay": config.to_dict(),
            "hard_negative_role": "validation_diagnostic_only",
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    fold_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("transition signal assay must not receive test rows")
        cache_row = cache_manifest.loc[cache_manifest["fold"].eq(int(fold))]
        if len(cache_row) != 1:
            raise ValueError(f"source cache manifest requires one row for fold {fold}")
        cache = load_probability_cache(Path(str(cache_row.iloc[0]["cache_path"])))
        cached_ids = np.asarray(cache["sample_id"]).astype(str)
        selected_ids = frame["sample_id"].astype(str).to_numpy()
        if set(cached_ids) != set(selected_ids):
            raise RuntimeError(f"fold {fold}: source cache and selected panel differ")
        by_id = frame.set_index(frame["sample_id"].astype(str), drop=False)
        frame = by_id.loc[cached_ids].reset_index(drop=True)
        if not np.array_equal(frame["sample_id"].astype(str).to_numpy(), cached_ids):
            raise RuntimeError(f"fold {fold}: cache row alignment failed")
        matrix = np.asarray(cache["exact_modes"], dtype=float)
        for model_name, readout_kind in MODEL_SPECS:
            metric_row, candidates, predictions = evaluate_exact_matrix(
                matrix,
                frame=frame,
                config=config,
                model_name=model_name,
                readout_kind=readout_kind,
            )
            fold_rows.append(metric_row)
            candidate_frames.append(candidates)
            prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(fold_rows)
    inner_candidates = pd.concat(candidate_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = _pooled_predictions_metrics(predictions)
    recommendation = _freeze_recommendation(pooled, config)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    inner_candidates.to_csv(run_dir / "inner_candidates.csv.gz", index=False)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False)
    pooled.to_csv(run_dir / "pooled_metrics_by_stratum_lead.csv", index=False)
    (run_dir / "freeze_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n"
    )
    summary = {
        "source_run": str(source_run),
        "folds": list(config.folds),
        "models": [name for name, _ in MODEL_SPECS],
        "prediction_rows": int(len(predictions)),
        "hard_negatives_used_for_fit_or_calibration": False,
        "test_evaluated": False,
        "recommendation": recommendation,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
