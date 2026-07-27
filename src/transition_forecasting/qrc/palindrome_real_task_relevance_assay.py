from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    qlike_loss,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import _evolve_segment_batch
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    CrossoverSchedule,
    _branch_drive,
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.ladder_finite_shot_sampling import validate_probe_probabilities
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    Representation,
    build_candidate_sequences,
    elementwise_quadratic_matrix,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _fresh_states,
    _resolve_probe_steps,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)

TemporalCondition = Literal[
    "ordered",
    "shuffled_keep_endpoint",
    "recent_tail",
    "static_endpoint",
]

SUPPORTED_TEMPORAL_CONDITIONS: tuple[TemporalCondition, ...] = (
    "ordered",
    "shuffled_keep_endpoint",
    "recent_tail",
    "static_endpoint",
)


@dataclass(frozen=True)
class PalindromeRealTaskConfig:
    """Bounded real-data assay for the repaired palindromic reservoir."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    representations: tuple[Representation, ...] = (
        "level_only",
        "level_instability",
    )
    temporal_conditions: tuple[TemporalCondition, ...] = SUPPORTED_TEMPORAL_CONDITIONS
    recent_tail_steps: int = 5
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    correction_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    inner_holdout_fraction: float = 0.25
    prequential_blocks: int = 5
    selection_seed: int = 20260725
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    schedule_name: str = "crossover_Ahalf_B_Ahalf"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.representations:
            raise ValueError("representations cannot be empty")
        if set(self.representations).difference(
            {"level_only", "level_instability", "level_positive_shock_energy"}
        ):
            raise ValueError("unsupported real-task representation")
        if not self.temporal_conditions:
            raise ValueError("temporal_conditions cannot be empty")
        if set(self.temporal_conditions).difference(SUPPORTED_TEMPORAL_CONDITIONS):
            raise ValueError("unsupported temporal condition")
        if "ordered" not in self.temporal_conditions:
            raise ValueError("ordered must be included as the reference condition")
        if not 1 <= self.recent_tail_steps <= self.sequence_length:
            raise ValueError("recent_tail_steps lies outside the sequence")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if not self.correction_lambdas or any(
            value < 0 for value in self.correction_lambdas
        ):
            raise ValueError("correction_lambdas must be nonnegative")
        if 0.0 not in self.correction_lambdas:
            raise ValueError("correction_lambdas must contain zero")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        _resolve_schedule(self.schedule_name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_schedule(name: str) -> CrossoverSchedule:
    matches = [schedule for schedule in CROSSOVER_SCHEDULES if schedule.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous crossover schedule: {name!r}")
    schedule = matches[0]
    schedule.validate()
    return schedule


def validate_har_contract(frame: pd.DataFrame) -> None:
    """Fail loudly if the baseline no longer uses the intended causal HAR columns."""

    expected = ("level", "mean5", "mean20")
    if tuple(HAR_FEATURES) != expected:
        raise RuntimeError(
            f"HAR feature contract changed: expected {expected}, observed {HAR_FEATURES}"
        )
    required = {*HAR_FEATURES, *TARGET_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"selected frame is missing columns: {sorted(missing)}")
    matrix = frame[list(HAR_FEATURES)].to_numpy(dtype=float)
    target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("HAR features and targets must be finite")
    if set(HAR_FEATURES).intersection(TARGET_COLUMNS):
        raise RuntimeError("HAR features overlap target columns")


def apply_temporal_condition(
    sequences: np.ndarray,
    condition: TemporalCondition,
    *,
    recent_tail_steps: int,
    seed: int,
) -> np.ndarray:
    """Create causal diagnostic controls without changing channel scaling."""

    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("sequences must be finite with shape (samples, time, 2)")
    if condition not in SUPPORTED_TEMPORAL_CONDITIONS:
        raise ValueError(f"unsupported temporal condition: {condition}")
    steps = values.shape[1]
    if not 1 <= recent_tail_steps <= steps:
        raise ValueError("recent_tail_steps lies outside the sequence")

    if condition == "ordered":
        return values.copy()
    if condition == "static_endpoint":
        return np.repeat(values[:, -1:, :], steps, axis=1)
    if condition == "recent_tail":
        result = values.copy()
        boundary = steps - recent_tail_steps
        if boundary > 0:
            result[:, :boundary, :] = values[:, boundary : boundary + 1, :]
        return result

    rng = np.random.default_rng(int(seed))
    result = values.copy()
    if steps <= 2:
        return result
    for row in range(len(result)):
        order = rng.permutation(steps - 1)
        result[row, :-1, :] = values[row, order, :]
    return result


def evolve_palindrome_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    schedule: CrossoverSchedule,
    *,
    interaction_scale: float,
    interactions: bool,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate the same palindrome with interactions either enabled or removed."""

    schedule.validate()
    reservoir.validate()
    geometry.validate()
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("real-task palindrome assay requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")

    physical_scale = float(interaction_scale) if interactions else 0.0
    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=physical_scale,
    )
    samples, steps, _ = values.shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []

    for step in range(steps):
        for branch, fraction in schedule.segments:
            omega, delta, phase = _branch_drive(
                values,
                step,
                branch,
                reservoir,
                drive_phase_rad,
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                reservoir.step_duration_us * float(fraction),
                reservoir,
                precomputed,
                interactions=bool(interactions),
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1), n_atoms=precomputed.n_atoms
    )
    return stacked, {
        "encoding": "symmetric_crossover",
        "schedule": schedule.name,
        "segments": [[branch, float(fraction)] for branch, fraction in schedule.segments],
        "interaction_scale_requested": float(interaction_scale),
        "interaction_scale_applied": physical_scale,
        "interactions_enabled": bool(interactions),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def _rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(prediction)) ** 2)))


def _mean_qlike(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(qlike_loss(y_true, prediction)))


def _fit_no_intercept_residual_model(
    matrix: np.ndarray,
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    config: PalindromeRealTaskConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.DataFrame]:
    """Nested chronological selection with an explicitly intercept-free QRC head."""

    features = np.asarray(matrix, dtype=float)
    if features.ndim != 2 or len(features) != len(frame):
        raise ValueError("feature matrix is not aligned with the selected frame")
    if not np.isfinite(features).all():
        raise ValueError("feature matrix contains non-finite values")

    inner_fit, inner_tune = chronological_inner_split(
        frame["origin_date"].astype(str).to_numpy(),
        residual_train_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )
    candidates: list[dict[str, float]] = []
    best: tuple[float, float, float, float] | None = None

    for alpha in config.ridge_alphas:
        scaler = StandardScaler().fit(features[inner_fit])
        design = scaler.transform(features)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(design[inner_fit], residuals[inner_fit])
        raw = np.asarray(model.predict(design), dtype=float)
        for correction_lambda in config.correction_lambdas:
            prediction = har + float(correction_lambda) * raw
            qlike = _mean_qlike(y[inner_tune], prediction[inner_tune])
            rmse = _rmse(y[inner_tune], prediction[inner_tune])
            candidates.append(
                {
                    "alpha": float(alpha),
                    "correction_lambda": float(correction_lambda),
                    "inner_qlike": qlike,
                    "inner_rmse": rmse,
                }
            )
            key = (qlike, rmse, float(correction_lambda), float(alpha))
            if best is None or key < best:
                best = key

    if best is None:
        raise RuntimeError("readout selection produced no candidates")
    selected_lambda = float(best[2])
    selected_alpha = float(best[3])

    scaler = StandardScaler().fit(features[residual_train_mask])
    design = scaler.transform(features)
    model = Ridge(alpha=selected_alpha, fit_intercept=False)
    model.fit(design[residual_train_mask], residuals[residual_train_mask])
    raw_correction = np.asarray(model.predict(design), dtype=float)
    correction = selected_lambda * raw_correction
    prediction = har + correction
    selection = {
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "fit_intercept": False,
        "inner_fit_rows": int(inner_fit.sum()),
        "inner_tune_rows": int(inner_tune.sum()),
        "full_fit_rows": int(residual_train_mask.sum()),
        "coefficient_l2": float(np.linalg.norm(np.asarray(model.coef_, dtype=float))),
        "raw_correction_rms": float(np.sqrt(np.mean(raw_correction**2))),
        "scaled_correction_rms": float(np.sqrt(np.mean(correction**2))),
    }
    return prediction, correction, selection, pd.DataFrame(candidates)


def _baseline_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    train_mean = np.mean(y[train_mask], axis=0, keepdims=True)
    level = frame["level"].to_numpy(dtype=float)[:, None]
    return {
        "train_mean": np.repeat(train_mean, len(frame), axis=0),
        "naive_level": np.repeat(level, y.shape[1], axis=1),
        "har": _fit_har(frame, y, train_mask),
    }


def _prediction_cells(
    frame: pd.DataFrame,
    *,
    fold: int,
    model: str,
    representation: str,
    condition: str,
    interactions: str,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    validation_mask: np.ndarray,
) -> pd.DataFrame:
    selected = frame.loc[validation_mask].reset_index(drop=True)
    horizons = y.shape[1]
    truth = y[validation_mask].reshape(-1)
    baseline = har[validation_mask].reshape(-1)
    forecast = prediction[validation_mask].reshape(-1)
    delta = correction[validation_mask].reshape(-1)
    return pd.DataFrame(
        {
            "fold": np.repeat(int(fold), len(truth)),
            "model": np.repeat(model, len(truth)),
            "representation": np.repeat(representation, len(truth)),
            "condition": np.repeat(condition, len(truth)),
            "interactions": np.repeat(interactions, len(truth)),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "origin_date": np.repeat(selected["origin_date"].astype(str), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "forecast_horizon": np.tile(np.arange(1, horizons + 1), len(selected)),
            "y_true": truth,
            "har_prediction": baseline,
            "prediction": forecast,
            "har_residual": truth - baseline,
            "correction": delta,
        }
    )


def summarize_prediction_cells(cells: pd.DataFrame, *, pooled: bool) -> pd.DataFrame:
    group_columns = [
        "model",
        "representation",
        "condition",
        "interactions",
    ]
    if not pooled:
        group_columns.insert(0, "fold")

    rows: list[dict[str, object]] = []
    for keys, group in cells.groupby(group_columns, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        scopes: list[tuple[str, np.ndarray]] = [
            ("path", np.ones(len(group), dtype=bool)),
            ("control", group["label"].eq(0).to_numpy()),
            ("transition", group["label"].eq(1).to_numpy()),
        ]
        for horizon in sorted(group["forecast_horizon"].unique()):
            scopes.append(
                (
                    f"h{int(horizon)}",
                    group["forecast_horizon"].eq(int(horizon)).to_numpy(),
                )
            )
        for scope, mask in scopes:
            if not mask.any():
                continue
            local = group.loc[mask]
            residual_negative = local["har_residual"].lt(0.0)
            residual_positive = local["har_residual"].gt(0.0)
            rows.append(
                {
                    **base,
                    "scope": scope,
                    "cells": int(len(local)),
                    "samples": int(local["sample_id"].nunique()),
                    "qlike": _mean_qlike(
                        local["y_true"].to_numpy(), local["prediction"].to_numpy()
                    ),
                    "rmse": _rmse(
                        local["y_true"].to_numpy(), local["prediction"].to_numpy()
                    ),
                    "har_qlike": _mean_qlike(
                        local["y_true"].to_numpy(),
                        local["har_prediction"].to_numpy(),
                    ),
                    "har_rmse": _rmse(
                        local["y_true"].to_numpy(),
                        local["har_prediction"].to_numpy(),
                    ),
                    "mean_correction": float(local["correction"].mean()),
                    "correction_rms": float(
                        np.sqrt(np.mean(local["correction"].to_numpy() ** 2))
                    ),
                    "wrong_up_rate": (
                        float(local.loc[residual_negative, "correction"].gt(0.0).mean())
                        if residual_negative.any()
                        else np.nan
                    ),
                    "correct_up_rate": (
                        float(local.loc[residual_positive, "correction"].gt(0.0).mean())
                        if residual_positive.any()
                        else np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["qlike_delta_vs_har"] = result["qlike"] - result["har_qlike"]
    result["rmse_delta_vs_har"] = result["rmse"] - result["har_rmse"]
    return result


def _feature_diagnostic(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    *,
    fold: int,
    model: str,
    representation: str,
    condition: str,
    interactions: str,
    ordered_reference: np.ndarray | None,
) -> dict[str, object]:
    features = np.asarray(matrix, dtype=float)
    train = features[train_mask]
    centered = train - train.mean(axis=0, keepdims=True)
    active = centered.std(axis=0) > 1e-12
    reference_delta = np.nan
    if ordered_reference is not None:
        denominator = float(np.linalg.norm(ordered_reference[train_mask]))
        reference_delta = (
            float(np.linalg.norm(features[train_mask] - ordered_reference[train_mask]))
            / denominator
            if denominator > 1e-15
            else np.nan
        )
    return {
        "fold": int(fold),
        "model": model,
        "representation": representation,
        "condition": condition,
        "interactions": interactions,
        "rows": int(len(features)),
        "feature_width": int(features.shape[1]),
        "active_features": int(active.sum()),
        "effective_rank_train": float(effective_rank(train)),
        "ordered_relative_feature_change": reference_delta,
    }


def run_palindrome_real_task_relevance_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    config: PalindromeRealTaskConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
    run_id: str,
) -> Path:
    """Test what the memory-specialist palindrome contributes on the actual L5 task."""

    config.validate()
    candidate_features.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("assay requires an exact six-atom reservoir")
    if not np.isclose(reservoir.step_duration_us, 0.02):
        raise ValueError("the frozen palindrome relevance assay requires 0.02 us steps")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")

    schedule = _resolve_schedule(config.schedule_name)
    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "geometry": geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "drive_phase_rad": float(drive_phase_rad),
            "scientific_questions": [
                "Does the 0.02-us palindrome add validation information beyond a recomputed HAR baseline?",
                "Does ordered history outperform endpoint-preserving shuffled, recent-tail-only, and static controls?",
                "Does the instability channel add value beyond level-only under equal operator exposure?",
                "Do Rydberg interactions add value beyond the matched interaction-off feature map?",
                "Can any gain survive an intercept-free residual head and matched classical linear/quadratic histories?",
            ],
            "test_rows_allowed": False,
            "qrc_head_fit_intercept": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )

    cell_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    scaler_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=(int(config.lead),),
            max_per_class=int(config.max_per_class),
            seed=int(config.selection_seed),
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("real-task relevance assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        level = level[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no valid selected rows")
        validate_har_contract(frame)

        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if train.sum() < 15 or validation.sum() < 5:
            raise RuntimeError(
                f"fold {fold}: insufficient train/validation rows: "
                f"train={int(train.sum())}, val={int(validation.sum())}"
            )

        baselines = _baseline_predictions(frame, y, train)
        har = baselines["har"]
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )
        if residual_train.sum() < 15:
            raise RuntimeError(
                f"fold {fold}: insufficient causal HAR residual rows: "
                f"{int(residual_train.sum())}"
            )

        zeros = np.zeros_like(y)
        for baseline_name, prediction in baselines.items():
            model_name = f"baseline_{baseline_name}"
            baseline_correction = prediction - har
            cell_frames.append(
                _prediction_cells(
                    frame,
                    fold=int(fold),
                    model=model_name,
                    representation="baseline",
                    condition="not_applicable",
                    interactions="not_applicable",
                    y=y,
                    har=har,
                    prediction=prediction,
                    correction=baseline_correction if baseline_name != "har" else zeros,
                    validation_mask=validation,
                )
            )
            baseline_rows.append(
                {
                    "fold": int(fold),
                    "model": model_name,
                    "validation_rows": int(validation.sum()),
                    "qlike": _mean_qlike(y[validation], prediction[validation]),
                    "rmse": _rmse(y[validation], prediction[validation]),
                }
            )

        for representation in config.representations:
            raw = build_candidate_sequences(level, representation, candidate_features)
            scaler = fit_channel_scaler(
                raw,
                train,
                q_low=candidate_features.q_low,
                q_high=candidate_features.q_high,
            )
            encoded = transform_candidate_sequences(raw, scaler)
            scaler_rows.append(
                {
                    "fold": int(fold),
                    "representation": representation,
                    "train_rows": int(train.sum()),
                    "channel_0_median": float(scaler.medians[0]),
                    "channel_1_median": float(scaler.medians[1]),
                    "channel_0_half_range": float(scaler.half_ranges[0]),
                    "channel_1_half_range": float(scaler.half_ranges[1]),
                }
            )

            classical_matrices = {
                "classical_linear_ordered": encoded.reshape(len(encoded), -1),
                "classical_quadratic_ordered": elementwise_quadratic_matrix(encoded),
            }
            for model_name, matrix in classical_matrices.items():
                prediction, correction, selection, candidates = (
                    _fit_no_intercept_residual_model(
                        matrix,
                        frame=frame,
                        y=y,
                        har=har,
                        residuals=residuals,
                        residual_train_mask=residual_train,
                        config=config,
                    )
                )
                selection_rows.append(
                    {
                        "fold": int(fold),
                        "model": model_name,
                        "representation": representation,
                        "condition": "ordered",
                        "interactions": "classical",
                        **selection,
                    }
                )
                candidates.insert(0, "fold", int(fold))
                candidates.insert(1, "model", model_name)
                candidates.insert(2, "representation", representation)
                selection_frames.append(candidates)
                feature_rows.append(
                    _feature_diagnostic(
                        matrix,
                        train,
                        fold=int(fold),
                        model=model_name,
                        representation=representation,
                        condition="ordered",
                        interactions="classical",
                        ordered_reference=None,
                    )
                )
                cell_frames.append(
                    _prediction_cells(
                        frame,
                        fold=int(fold),
                        model=model_name,
                        representation=representation,
                        condition="ordered",
                        interactions="classical",
                        y=y,
                        har=har,
                        prediction=prediction,
                        correction=correction,
                        validation_mask=validation,
                    )
                )

            ordered_features: np.ndarray | None = None
            for condition_index, condition in enumerate(config.temporal_conditions):
                conditioned = apply_temporal_condition(
                    encoded,
                    condition,
                    recent_tail_steps=config.recent_tail_steps,
                    seed=(
                        int(config.selection_seed)
                        + 100_003 * int(fold)
                        + 1_009 * condition_index
                        + 17 * list(config.representations).index(representation)
                    ),
                )
                probabilities, metadata = evolve_palindrome_probabilities(
                    conditioned,
                    reservoir,
                    geometry,
                    schedule,
                    interaction_scale=float(interaction_scale),
                    interactions=True,
                    drive_phase_rad=float(drive_phase_rad),
                )
                matrix = np.asarray(
                    build_crossover_feature_banks(probabilities)["occupation_pair_raw"],
                    dtype=float,
                )
                if condition == "ordered":
                    ordered_features = matrix
                model_name = f"palindrome_{condition}_on"
                prediction, correction, selection, candidates = (
                    _fit_no_intercept_residual_model(
                        matrix,
                        frame=frame,
                        y=y,
                        har=har,
                        residuals=residuals,
                        residual_train_mask=residual_train,
                        config=config,
                    )
                )
                selection_rows.append(
                    {
                        "fold": int(fold),
                        "model": model_name,
                        "representation": representation,
                        "condition": condition,
                        "interactions": "on",
                        **selection,
                    }
                )
                candidates.insert(0, "fold", int(fold))
                candidates.insert(1, "model", model_name)
                candidates.insert(2, "representation", representation)
                selection_frames.append(candidates)
                feature_rows.append(
                    _feature_diagnostic(
                        matrix,
                        train,
                        fold=int(fold),
                        model=model_name,
                        representation=representation,
                        condition=condition,
                        interactions="on",
                        ordered_reference=(
                            ordered_features if condition != "ordered" else None
                        ),
                    )
                )
                metadata_rows.append(
                    {
                        "fold": int(fold),
                        "representation": representation,
                        "condition": condition,
                        "interactions": "on",
                        "feature_width": int(matrix.shape[1]),
                        "metadata": json.dumps(metadata, sort_keys=True),
                    }
                )
                cell_frames.append(
                    _prediction_cells(
                        frame,
                        fold=int(fold),
                        model=model_name,
                        representation=representation,
                        condition=condition,
                        interactions="on",
                        y=y,
                        har=har,
                        prediction=prediction,
                        correction=correction,
                        validation_mask=validation,
                    )
                )

            probabilities_off, metadata_off = evolve_palindrome_probabilities(
                encoded,
                reservoir,
                geometry,
                schedule,
                interaction_scale=float(interaction_scale),
                interactions=False,
                drive_phase_rad=float(drive_phase_rad),
            )
            off_matrix = np.asarray(
                build_crossover_feature_banks(probabilities_off)["occupation_pair_raw"],
                dtype=float,
            )
            off_model_name = "palindrome_ordered_off"
            prediction, correction, selection, candidates = (
                _fit_no_intercept_residual_model(
                    off_matrix,
                    frame=frame,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train,
                    config=config,
                )
            )
            selection_rows.append(
                {
                    "fold": int(fold),
                    "model": off_model_name,
                    "representation": representation,
                    "condition": "ordered",
                    "interactions": "off",
                    **selection,
                }
            )
            candidates.insert(0, "fold", int(fold))
            candidates.insert(1, "model", off_model_name)
            candidates.insert(2, "representation", representation)
            selection_frames.append(candidates)
            feature_rows.append(
                _feature_diagnostic(
                    off_matrix,
                    train,
                    fold=int(fold),
                    model=off_model_name,
                    representation=representation,
                    condition="ordered",
                    interactions="off",
                    ordered_reference=ordered_features,
                )
            )
            metadata_rows.append(
                {
                    "fold": int(fold),
                    "representation": representation,
                    "condition": "ordered",
                    "interactions": "off",
                    "feature_width": int(off_matrix.shape[1]),
                    "metadata": json.dumps(metadata_off, sort_keys=True),
                }
            )
            cell_frames.append(
                _prediction_cells(
                    frame,
                    fold=int(fold),
                    model=off_model_name,
                    representation=representation,
                    condition="ordered",
                    interactions="off",
                    y=y,
                    har=har,
                    prediction=prediction,
                    correction=correction,
                    validation_mask=validation,
                )
            )

    cells = pd.concat(cell_frames, ignore_index=True)
    fold_metrics = summarize_prediction_cells(cells, pooled=False)
    pooled_metrics = summarize_prediction_cells(cells, pooled=True)
    selections = pd.DataFrame(selection_rows)
    candidates = pd.concat(selection_frames, ignore_index=True)
    feature_diagnostics = pd.DataFrame(feature_rows)
    baselines = pd.DataFrame(baseline_rows)
    scalers = pd.DataFrame(scaler_rows)
    simulation_metadata = pd.DataFrame(metadata_rows)

    cells.to_csv(run_dir / "prediction_cells.csv.gz", index=False, compression="gzip")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    pooled_metrics.to_csv(run_dir / "pooled_metrics.csv", index=False)
    selections.to_csv(run_dir / "readout_selections.csv", index=False)
    candidates.to_csv(
        run_dir / "readout_candidates.csv.gz", index=False, compression="gzip"
    )
    feature_diagnostics.to_csv(run_dir / "feature_diagnostics.csv", index=False)
    baselines.to_csv(run_dir / "baseline_health.csv", index=False)
    scalers.to_csv(run_dir / "channel_scalers.csv", index=False)
    simulation_metadata.to_csv(run_dir / "simulation_metadata.csv", index=False)

    path_metrics = pooled_metrics.loc[pooled_metrics["scope"].eq("path")].copy()
    nonbaseline = path_metrics.loc[~path_metrics["model"].str.startswith("baseline_")]
    best = (
        nonbaseline.sort_values(
            ["qlike", "rmse", "model", "representation"], kind="mergesort"
        ).iloc[0]
        if not nonbaseline.empty
        else None
    )
    ordered = path_metrics.loc[
        path_metrics["model"].eq("palindrome_ordered_on")
    ].copy()
    ordered_beats_har = bool(
        (
            (ordered["qlike_delta_vs_har"] < 0.0)
            & (ordered["rmse_delta_vs_har"] <= 0.0)
        ).any()
    )
    summary = {
        "schema_version": 1,
        "status": "palindrome_real_task_relevance_assay_complete",
        "test_rows_used": 0,
        "qrc_head_fit_intercept": False,
        "har_feature_contract": list(HAR_FEATURES),
        "folds": [int(value) for value in config.folds],
        "lead": int(config.lead),
        "representations": list(config.representations),
        "temporal_conditions": list(config.temporal_conditions),
        "ordered_palindrome_beats_har_on_both_path_metrics_for_any_representation": (
            ordered_beats_har
        ),
        "best_nonbaseline_path_model": (
            {
                "model": str(best["model"]),
                "representation": str(best["representation"]),
                "condition": str(best["condition"]),
                "interactions": str(best["interactions"]),
                "qlike": float(best["qlike"]),
                "rmse": float(best["rmse"]),
                "qlike_delta_vs_har": float(best["qlike_delta_vs_har"]),
                "rmse_delta_vs_har": float(best["rmse_delta_vs_har"]),
            }
            if best is not None
            else None
        ),
        "interpretation_rules": [
            "Ordered must beat endpoint-preserving shuffled history before temporal memory is credited.",
            "Ordered interaction-on must beat ordered interaction-off before Rydberg interactions are credited.",
            "Level plus instability must beat level-only before the second channel is credited.",
            "Palindrome must beat matched classical linear and quadratic histories before the quantum feature map is credited.",
            "A QLIKE gain accompanied by material RMSE damage or wrong-direction calm corrections is not a clean success.",
        ],
        "files": {
            "prediction_cells": "prediction_cells.csv.gz",
            "fold_metrics": "fold_metrics.csv",
            "pooled_metrics": "pooled_metrics.csv",
            "readout_selections": "readout_selections.csv",
            "readout_candidates": "readout_candidates.csv.gz",
            "feature_diagnostics": "feature_diagnostics.csv",
            "baseline_health": "baseline_health.csv",
            "channel_scalers": "channel_scalers.csv",
            "simulation_metadata": "simulation_metadata.csv",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
