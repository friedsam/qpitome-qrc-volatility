from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
)
from transition_forecasting.qrc.bivariate_capacity_assay import (
    BivariateCapacityConfig,
    build_capacity_targets,
    capacity_summary,
    fit_capacity_readout,
    generate_bivariate_windows,
    squared_correlation,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    CrossoverSchedule,
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _baseline_predictions,
    _mean_qlike,
    _prediction_cells,
    _prequential_har_residuals,
    _rmse,
    evolve_palindrome_probabilities,
    summarize_prediction_cells,
    validate_har_contract,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    Representation,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

InteractionLabel = Literal["on", "off"]
_SUPPORTED_REPRESENTATIONS: tuple[Representation, ...] = (
    "level_only",
    "level_instability",
    "level_positive_shock_energy",
)


@dataclass(frozen=True)
class PalindromeDurationMemoryPerformanceConfig:
    """Joint operating-point assay for memory and lead-5 forecast relevance.

    The current palindrome orientation is held fixed by default so duration is the
    only new physical axis. The mirrored orientation can be requested explicitly.
    """

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 24
    sequence_length: int = 40
    representations: tuple[Representation, ...] = _SUPPORTED_REPRESENTATIONS
    durations_us: tuple[float, ...] = (0.010, 0.015, 0.020, 0.025, 0.030)
    schedule_names: tuple[str, ...] = ("crossover_Ahalf_B_Ahalf",)
    interaction_labels: tuple[InteractionLabel, ...] = ("on", "off")
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    correction_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    inner_holdout_fraction: float = 0.25
    prequential_blocks: int = 5
    selection_seed: int = 20260725
    synthetic_samples: int = 320
    synthetic_seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    synthetic_alphas: tuple[float, ...] = (
        1e-6,
        1e-4,
        1e-2,
        0.1,
        1.0,
        10.0,
        100.0,
    )
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if self.lead < 1:
            raise ValueError("lead must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.representations or set(self.representations).difference(
            _SUPPORTED_REPRESENTATIONS
        ):
            raise ValueError("unsupported duration-assay representation")
        if not self.durations_us or len(set(self.durations_us)) != len(self.durations_us):
            raise ValueError("durations_us must be nonempty and unique")
        if any(value <= 0 or not np.isfinite(value) for value in self.durations_us):
            raise ValueError("durations_us must be finite and positive")
        if not self.schedule_names or len(set(self.schedule_names)) != len(
            self.schedule_names
        ):
            raise ValueError("schedule_names must be nonempty and unique")
        for name in self.schedule_names:
            _resolve_schedule(name)
        if not self.interaction_labels or set(self.interaction_labels).difference(
            {"on", "off"}
        ):
            raise ValueError("interaction_labels must contain only on/off")
        if not self.memory_delays or any(value < 1 for value in self.memory_delays):
            raise ValueError("memory_delays must be positive")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the input history")
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
        if self.synthetic_samples < 100:
            raise ValueError("synthetic_samples must be at least 100")
        if not self.synthetic_seeds or len(set(self.synthetic_seeds)) != len(
            self.synthetic_seeds
        ):
            raise ValueError("synthetic_seeds must be nonempty and unique")
        if not self.synthetic_alphas or any(value <= 0 for value in self.synthetic_alphas):
            raise ValueError("synthetic_alphas must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_schedule(name: str) -> CrossoverSchedule:
    matches = [item for item in CROSSOVER_SCHEDULES if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous crossover schedule: {name!r}")
    matches[0].validate()
    return matches[0]


def _duration_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _fit_scaled_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    alpha: float,
    *,
    fit_intercept: bool,
) -> tuple[StandardScaler, Ridge, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    scaler = StandardScaler().fit(matrix[fit_mask])
    design = scaler.transform(matrix)
    model = Ridge(alpha=float(alpha), fit_intercept=bool(fit_intercept))
    model.fit(design[fit_mask], y[fit_mask])
    prediction = np.asarray(model.predict(design), dtype=float)
    return scaler, model, prediction


def fit_duration_selected_residual_model(
    matrices_by_duration: dict[float, np.ndarray],
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: PalindromeDurationMemoryPerformanceConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, object],
    pd.DataFrame,
    dict[float, dict[str, object]],
]:
    """Select duration, ridge alpha, and correction strength on inner time only.

    Fixed-duration outer results are returned as diagnostics. They are not used to
    choose the headline duration-selected model.
    """

    if set(float(value) for value in matrices_by_duration) != set(config.durations_us):
        raise ValueError("matrices_by_duration does not match configured durations")
    inner_fit, inner_tune = chronological_inner_split(
        frame["origin_date"].astype(str).to_numpy(),
        residual_train_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )

    candidate_rows: list[dict[str, float]] = []
    raw_by_duration_alpha: dict[tuple[float, float], np.ndarray] = {}
    for duration in sorted(matrices_by_duration):
        matrix = np.asarray(matrices_by_duration[duration], dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(frame) or not np.isfinite(matrix).all():
            raise ValueError("duration feature matrix is invalid or misaligned")
        for alpha in config.ridge_alphas:
            _, _, raw = _fit_scaled_ridge(
                matrix,
                residuals,
                inner_fit,
                float(alpha),
                fit_intercept=False,
            )
            raw_by_duration_alpha[(float(duration), float(alpha))] = raw
            for correction_lambda in config.correction_lambdas:
                prediction = har + float(correction_lambda) * raw
                candidate_rows.append(
                    {
                        "step_duration_us": float(duration),
                        "alpha": float(alpha),
                        "correction_lambda": float(correction_lambda),
                        "inner_qlike": _mean_qlike(
                            y[inner_tune], prediction[inner_tune]
                        ),
                        "inner_rmse": _rmse(y[inner_tune], prediction[inner_tune]),
                    }
                )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError("duration selection produced no candidates")

    def _candidate_key(row: pd.Series) -> tuple[float, ...]:
        return (
            float(row["inner_qlike"]),
            float(row["inner_rmse"]),
            float(row["correction_lambda"]),
            abs(float(row["step_duration_us"]) - 0.020),
            float(row["alpha"]),
            float(row["step_duration_us"]),
        )

    best_global = min((row for _, row in candidates.iterrows()), key=_candidate_key)
    fixed_results: dict[float, dict[str, object]] = {}
    for duration, local in candidates.groupby("step_duration_us", sort=True):
        choice = min((row for _, row in local.iterrows()), key=_candidate_key)
        alpha = float(choice["alpha"])
        correction_lambda = float(choice["correction_lambda"])
        matrix = np.asarray(matrices_by_duration[float(duration)], dtype=float)
        _, model, raw = _fit_scaled_ridge(
            matrix,
            residuals,
            residual_train_mask,
            alpha,
            fit_intercept=False,
        )
        correction = correction_lambda * raw
        prediction = har + correction
        fixed_results[float(duration)] = {
            "prediction": prediction,
            "correction": correction,
            "selected_alpha": alpha,
            "selected_lambda": correction_lambda,
            "outer_qlike": _mean_qlike(y[validation_mask], prediction[validation_mask]),
            "outer_rmse": _rmse(y[validation_mask], prediction[validation_mask]),
            "coefficient_l2": float(np.linalg.norm(np.asarray(model.coef_, dtype=float))),
            "raw_correction_rms": float(np.sqrt(np.mean(raw**2))),
            "scaled_correction_rms": float(np.sqrt(np.mean(correction**2))),
        }

    selected_duration = float(best_global["step_duration_us"])
    selected = fixed_results[selected_duration]
    selection = {
        "selected_duration_us": selected_duration,
        "selected_alpha": float(best_global["alpha"]),
        "selected_lambda": float(best_global["correction_lambda"]),
        "inner_qlike": float(best_global["inner_qlike"]),
        "inner_rmse": float(best_global["inner_rmse"]),
        "fit_intercept": False,
        "inner_fit_rows": int(inner_fit.sum()),
        "inner_tune_rows": int(inner_tune.sum()),
        "full_fit_rows": int(residual_train_mask.sum()),
        "coefficient_l2": float(selected["coefficient_l2"]),
        "raw_correction_rms": float(selected["raw_correction_rms"]),
        "scaled_correction_rms": float(selected["scaled_correction_rms"]),
    }
    return (
        np.asarray(selected["prediction"], dtype=float),
        np.asarray(selected["correction"], dtype=float),
        selection,
        candidates,
        fixed_results,
    )


def _readout_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    estimated = np.asarray(prediction, dtype=float).reshape(-1)
    return {
        "r2": float(r2_score(observed, estimated)),
        "corr2": squared_correlation(observed, estimated),
        "rmse": float(np.sqrt(np.mean((observed - estimated) ** 2))),
    }


def fit_real_history_memory(
    qrc_features: np.ndarray,
    encoded_sequences: np.ndarray,
    *,
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    memory_delays: tuple[int, ...],
    alphas: tuple[float, ...],
    inner_holdout_fraction: float,
) -> pd.DataFrame:
    """Measure lag recovery on financial histories beyond an endpoint baseline."""

    features = np.asarray(qrc_features, dtype=float)
    encoded = np.asarray(encoded_sequences, dtype=float)
    if features.ndim != 2 or len(features) != len(frame):
        raise ValueError("qrc_features must align with frame")
    if encoded.ndim != 3 or len(encoded) != len(frame) or encoded.shape[2] != 2:
        raise ValueError("encoded_sequences must have shape (rows, time, 2)")
    if not np.isfinite(features).all() or not np.isfinite(encoded).all():
        raise ValueError("memory inputs must be finite")

    inner_fit, inner_tune = chronological_inner_split(
        frame["origin_date"].astype(str).to_numpy(),
        train_mask,
        holdout_fraction=inner_holdout_fraction,
    )
    endpoint = encoded[:, -1, :]
    designs = {
        "endpoint": endpoint,
        "qrc": features,
        "qrc_plus_endpoint": np.concatenate([features, endpoint], axis=1),
    }
    rows: list[dict[str, object]] = []
    for channel in (0, 1):
        for delay in memory_delays:
            target = encoded[:, -(int(delay) + 1), channel]
            if float(np.std(target[train_mask])) <= 1e-12:
                for readout_name in designs:
                    rows.append(
                        {
                            "channel": int(channel + 1),
                            "delay": int(delay),
                            "readout": readout_name,
                            "status": "constant_target",
                            "selected_alpha": np.nan,
                            "validation_r2": np.nan,
                            "validation_corr2": np.nan,
                            "test_r2": np.nan,
                            "test_corr2": np.nan,
                            "test_rmse": np.nan,
                        }
                    )
                continue
            for readout_name, matrix in designs.items():
                choices: list[dict[str, float]] = []
                for alpha in alphas:
                    _, _, prediction = _fit_scaled_ridge(
                        matrix,
                        target,
                        inner_fit,
                        float(alpha),
                        fit_intercept=True,
                    )
                    metrics = _readout_metrics(target[inner_tune], prediction[inner_tune])
                    choices.append(
                        {
                            "alpha": float(alpha),
                            "r2": metrics["r2"],
                            "corr2": metrics["corr2"],
                            "rmse": metrics["rmse"],
                        }
                    )
                choice = min(
                    choices,
                    key=lambda row: (
                        -float(row["corr2"]),
                        -float(row["r2"]),
                        float(row["rmse"]),
                        float(row["alpha"]),
                    ),
                )
                _, _, prediction = _fit_scaled_ridge(
                    matrix,
                    target,
                    train_mask,
                    float(choice["alpha"]),
                    fit_intercept=True,
                )
                test_metrics = _readout_metrics(
                    target[validation_mask], prediction[validation_mask]
                )
                rows.append(
                    {
                        "channel": int(channel + 1),
                        "delay": int(delay),
                        "readout": readout_name,
                        "status": "ok",
                        "selected_alpha": float(choice["alpha"]),
                        "validation_r2": float(choice["r2"]),
                        "validation_corr2": float(choice["corr2"]),
                        "test_r2": test_metrics["r2"],
                        "test_corr2": test_metrics["corr2"],
                        "test_rmse": test_metrics["rmse"],
                    }
                )

    result = pd.DataFrame(rows)
    endpoint_metrics = result.loc[result["readout"].eq("endpoint"), [
        "channel",
        "delay",
        "test_r2",
        "test_corr2",
    ]].rename(
        columns={
            "test_r2": "endpoint_test_r2",
            "test_corr2": "endpoint_test_corr2",
        }
    )
    result = result.merge(endpoint_metrics, on=["channel", "delay"], how="left")
    result["incremental_r2_vs_endpoint"] = (
        result["test_r2"] - result["endpoint_test_r2"]
    )
    result["incremental_corr2_vs_endpoint"] = (
        result["test_corr2"] - result["endpoint_test_corr2"]
    )
    return result


def _feature_row(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    *,
    fold: int,
    representation: str,
    schedule: str,
    interactions: str,
    duration_us: float,
) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    train = values[train_mask]
    centered = train - train.mean(axis=0, keepdims=True)
    return {
        "fold": int(fold),
        "representation": representation,
        "schedule": schedule,
        "interactions": interactions,
        "step_duration_us": float(duration_us),
        "feature_width": int(values.shape[1]),
        "active_features": int((centered.std(axis=0) > 1e-12).sum()),
        "effective_rank_train": float(effective_rank(train)),
        "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
    }


def _run_synthetic_memory(
    *,
    config: PalindromeDurationMemoryPerformanceConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for seed in config.synthetic_seeds:
        capacity_config = BivariateCapacityConfig(
            samples=int(config.synthetic_samples),
            sequence_length=int(config.sequence_length),
            memory_delays=tuple(config.memory_delays),
            alphas=tuple(config.synthetic_alphas),
            interaction_scale=float(interaction_scale),
            seed=int(seed),
        )
        windows = generate_bivariate_windows(capacity_config)
        targets = build_capacity_targets(windows, capacity_config)
        for schedule_name in config.schedule_names:
            schedule = _resolve_schedule(schedule_name)
            for interactions in config.interaction_labels:
                enabled = interactions == "on"
                for duration in config.durations_us:
                    local_reservoir = replace(
                        reservoir, step_duration_us=float(duration)
                    )
                    probabilities, metadata = evolve_palindrome_probabilities(
                        windows,
                        local_reservoir,
                        geometry,
                        schedule,
                        interaction_scale=float(interaction_scale),
                        interactions=enabled,
                        drive_phase_rad=float(drive_phase_rad),
                    )
                    matrix = np.asarray(
                        build_crossover_feature_banks(probabilities)[
                            "occupation_pair_raw"
                        ],
                        dtype=float,
                    )
                    metrics, diagnostics, candidates = fit_capacity_readout(
                        matrix,
                        targets,
                        capacity_config,
                    )
                    for table in (metrics, candidates):
                        table.insert(0, "seed", int(seed))
                        table.insert(1, "schedule", schedule_name)
                        table.insert(2, "interactions", interactions)
                        table.insert(3, "step_duration_us", float(duration))
                    task_frames.append(metrics)
                    candidate_frames.append(candidates)
                    summary_rows.append(
                        {
                            "seed": int(seed),
                            "schedule": schedule_name,
                            "interactions": interactions,
                            "step_duration_us": float(duration),
                            **capacity_summary(metrics),
                            **diagnostics,
                        }
                    )
                    metadata_rows.append(
                        {
                            "seed": int(seed),
                            "schedule": schedule_name,
                            "interactions": interactions,
                            "step_duration_us": float(duration),
                            "metadata": json.dumps(metadata, sort_keys=True),
                        }
                    )
    tasks = pd.concat(task_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    aggregate = (
        summary.groupby(
            ["schedule", "interactions", "step_duration_us"],
            sort=True,
            as_index=False,
        )
        .mean(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    metadata = pd.DataFrame(metadata_rows)
    return tasks, candidates, summary, aggregate.merge(
        metadata.groupby(
            ["schedule", "interactions", "step_duration_us"], as_index=False
        ).size(),
        on=["schedule", "interactions", "step_duration_us"],
        how="left",
    )


def _memory_performance_correlations(
    diagnostic_metrics: pd.DataFrame,
    real_memory: pd.DataFrame,
    synthetic_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    path = diagnostic_metrics.loc[diagnostic_metrics["scope"].eq("path")].copy()
    path = (
        path.groupby(
            [
                "representation",
                "schedule",
                "interactions",
                "step_duration_us",
            ],
            as_index=False,
            sort=True,
        )[["qlike_delta_vs_har", "rmse_delta_vs_har"]]
        .mean()
    )
    real = real_memory.loc[
        real_memory["readout"].eq("qrc_plus_endpoint")
        & real_memory["status"].eq("ok")
    ].copy()
    real_aggregate = (
        real.groupby(
            [
                "representation",
                "schedule",
                "interactions",
                "step_duration_us",
            ],
            as_index=False,
            sort=True,
        )[
            [
                "test_corr2",
                "incremental_corr2_vs_endpoint",
                "test_r2",
                "incremental_r2_vs_endpoint",
            ]
        ]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for keys, local_path in path.groupby(
        ["representation", "schedule", "interactions"], sort=True
    ):
        representation, schedule, interactions = keys
        joined = local_path.merge(
            real_aggregate.loc[
                real_aggregate["representation"].eq(representation)
                & real_aggregate["schedule"].eq(schedule)
                & real_aggregate["interactions"].eq(interactions)
            ],
            on=[
                "representation",
                "schedule",
                "interactions",
                "step_duration_us",
            ],
            how="inner",
        ).merge(
            synthetic_aggregate.loc[
                synthetic_aggregate["schedule"].eq(schedule)
                & synthetic_aggregate["interactions"].eq(interactions)
            ],
            on=["schedule", "interactions", "step_duration_us"],
            how="left",
            suffixes=("", "_synthetic"),
        )
        memory_columns = [
            "test_corr2",
            "incremental_corr2_vs_endpoint",
            "test_r2",
            "incremental_r2_vs_endpoint",
            "memory_capacity_u1",
            "memory_capacity_u2",
            "mixing_capacity",
            "order_capacity",
            "mean_delay5_memory_capacity",
            "effective_rank_train_validation",
        ]
        for memory_metric in memory_columns:
            if memory_metric not in joined:
                continue
            valid = np.isfinite(joined[memory_metric]) & np.isfinite(
                joined["qlike_delta_vs_har"]
            )
            if int(valid.sum()) < 3:
                continue
            rows.append(
                {
                    "representation": representation,
                    "schedule": schedule,
                    "interactions": interactions,
                    "memory_metric": memory_metric,
                    "duration_points": int(valid.sum()),
                    "spearman_vs_qlike_delta": float(
                        joined.loc[valid, memory_metric].corr(
                            joined.loc[valid, "qlike_delta_vs_har"],
                            method="spearman",
                        )
                    ),
                    "spearman_vs_rmse_delta": float(
                        joined.loc[valid, memory_metric].corr(
                            joined.loc[valid, "rmse_delta_vs_har"],
                            method="spearman",
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_palindrome_duration_memory_performance_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    config: PalindromeDurationMemoryPerformanceConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
    run_id: str,
) -> Path:
    """Map duration-dependent memory and forecast behavior without test rows."""

    config.validate()
    candidate_features.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("duration assay requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir_template": reservoir.to_dict(),
            "geometry": geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "drive_phase_rad": float(drive_phase_rad),
            "scientific_questions": [
                "How does palindrome step duration change intrinsic lag memory and nonlinear capacity?",
                "How does duration change recoverable financial history beyond the current endpoint?",
                "Does leakage-safe duration selection improve lead-5 forecasts across folds?",
                "Do interactions become useful at a different duration than 0.02 us?",
            ],
            "test_rows_allowed": False,
            "duration_selected_on_outer_validation": False,
            "qrc_head_fit_intercept": False,
        },
        run_id=run_id,
    )

    synthetic_tasks, synthetic_candidates, synthetic_summary, synthetic_aggregate = (
        _run_synthetic_memory(
            config=config,
            reservoir=reservoir,
            geometry=geometry,
            interaction_scale=float(interaction_scale),
            drive_phase_rad=float(drive_phase_rad),
        )
    )

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )

    selected_cell_frames: list[pd.DataFrame] = []
    diagnostic_cell_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    real_memory_frames: list[pd.DataFrame] = []
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
            raise RuntimeError("duration assay must not receive test rows")
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
            cells = _prediction_cells(
                frame,
                fold=int(fold),
                model=f"baseline_{baseline_name}",
                representation="baseline",
                condition="not_applicable",
                interactions="not_applicable",
                y=y,
                har=har,
                prediction=prediction,
                correction=prediction - har if baseline_name != "har" else zeros,
                validation_mask=validation,
            )
            cells.insert(4, "schedule", "not_applicable")
            cells.insert(5, "step_duration_us", np.nan)
            selected_cell_frames.append(cells)
            baseline_rows.append(
                {
                    "fold": int(fold),
                    "model": f"baseline_{baseline_name}",
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
            for schedule_name in config.schedule_names:
                schedule = _resolve_schedule(schedule_name)
                for interactions in config.interaction_labels:
                    enabled = interactions == "on"
                    matrices: dict[float, np.ndarray] = {}
                    for duration in config.durations_us:
                        local_reservoir = replace(
                            reservoir, step_duration_us=float(duration)
                        )
                        probabilities, metadata = evolve_palindrome_probabilities(
                            encoded,
                            local_reservoir,
                            geometry,
                            schedule,
                            interaction_scale=float(interaction_scale),
                            interactions=enabled,
                            drive_phase_rad=float(drive_phase_rad),
                        )
                        matrix = np.asarray(
                            build_crossover_feature_banks(probabilities)[
                                "occupation_pair_raw"
                            ],
                            dtype=float,
                        )
                        matrices[float(duration)] = matrix
                        feature_rows.append(
                            _feature_row(
                                matrix,
                                train,
                                fold=int(fold),
                                representation=representation,
                                schedule=schedule_name,
                                interactions=interactions,
                                duration_us=float(duration),
                            )
                        )
                        memory = fit_real_history_memory(
                            matrix,
                            encoded,
                            frame=frame,
                            train_mask=train,
                            validation_mask=validation,
                            memory_delays=config.memory_delays,
                            alphas=config.ridge_alphas,
                            inner_holdout_fraction=config.inner_holdout_fraction,
                        )
                        memory.insert(0, "fold", int(fold))
                        memory.insert(1, "representation", representation)
                        memory.insert(2, "schedule", schedule_name)
                        memory.insert(3, "interactions", interactions)
                        memory.insert(4, "step_duration_us", float(duration))
                        real_memory_frames.append(memory)
                        metadata_rows.append(
                            {
                                "fold": int(fold),
                                "representation": representation,
                                "schedule": schedule_name,
                                "interactions": interactions,
                                "step_duration_us": float(duration),
                                "feature_width": int(matrix.shape[1]),
                                "metadata": json.dumps(metadata, sort_keys=True),
                            }
                        )

                    prediction, correction, selection, candidates, fixed = (
                        fit_duration_selected_residual_model(
                            matrices,
                            frame=frame,
                            y=y,
                            har=har,
                            residuals=residuals,
                            residual_train_mask=residual_train,
                            validation_mask=validation,
                            config=config,
                        )
                    )
                    model_name = (
                        f"palindrome_duration_selected_{schedule_name}_{interactions}"
                    )
                    selection_rows.append(
                        {
                            "fold": int(fold),
                            "model": model_name,
                            "representation": representation,
                            "schedule": schedule_name,
                            "interactions": interactions,
                            **selection,
                        }
                    )
                    candidates.insert(0, "fold", int(fold))
                    candidates.insert(1, "representation", representation)
                    candidates.insert(2, "schedule", schedule_name)
                    candidates.insert(3, "interactions", interactions)
                    candidate_frames.append(candidates)

                    cells = _prediction_cells(
                        frame,
                        fold=int(fold),
                        model=model_name,
                        representation=representation,
                        condition="ordered",
                        interactions=interactions,
                        y=y,
                        har=har,
                        prediction=prediction,
                        correction=correction,
                        validation_mask=validation,
                    )
                    cells.insert(4, "schedule", schedule_name)
                    cells.insert(5, "step_duration_us", selection["selected_duration_us"])
                    selected_cell_frames.append(cells)

                    for duration, result in fixed.items():
                        diagnostic_model = (
                            f"palindrome_fixed_dt_{_duration_tag(duration)}_"
                            f"{schedule_name}_{interactions}"
                        )
                        local_cells = _prediction_cells(
                            frame,
                            fold=int(fold),
                            model=diagnostic_model,
                            representation=representation,
                            condition="ordered_fixed_duration_diagnostic",
                            interactions=interactions,
                            y=y,
                            har=har,
                            prediction=np.asarray(result["prediction"], dtype=float),
                            correction=np.asarray(result["correction"], dtype=float),
                            validation_mask=validation,
                        )
                        local_cells.insert(4, "schedule", schedule_name)
                        local_cells.insert(5, "step_duration_us", float(duration))
                        local_cells["selected_alpha"] = float(result["selected_alpha"])
                        local_cells["selected_lambda"] = float(result["selected_lambda"])
                        diagnostic_cell_frames.append(local_cells)

    selected_cells = pd.concat(selected_cell_frames, ignore_index=True)
    diagnostic_cells = pd.concat(diagnostic_cell_frames, ignore_index=True)
    selected_fold_metrics = summarize_prediction_cells(selected_cells, pooled=False)
    selected_pooled_metrics = summarize_prediction_cells(selected_cells, pooled=True)
    diagnostic_fold_metrics = summarize_prediction_cells(diagnostic_cells, pooled=False)
    diagnostic_pooled_metrics = summarize_prediction_cells(diagnostic_cells, pooled=True)

    def _attach_duration_and_schedule(
        metrics: pd.DataFrame, cells: pd.DataFrame
    ) -> pd.DataFrame:
        lookup = cells[[
            "fold",
            "model",
            "representation",
            "schedule",
            "step_duration_us",
        ]].drop_duplicates()
        keys = ["fold", "model", "representation"] if "fold" in metrics else [
            "model",
            "representation",
        ]
        if "fold" not in metrics:
            lookup = lookup.drop(columns=["fold"]).drop_duplicates()
        return metrics.merge(lookup, on=keys, how="left")

    selected_fold_metrics = _attach_duration_and_schedule(
        selected_fold_metrics, selected_cells
    )
    diagnostic_fold_metrics = _attach_duration_and_schedule(
        diagnostic_fold_metrics, diagnostic_cells
    )
    diagnostic_pooled_metrics = _attach_duration_and_schedule(
        diagnostic_pooled_metrics, diagnostic_cells
    )

    selections = pd.DataFrame(selection_rows)
    duration_candidates = pd.concat(candidate_frames, ignore_index=True)
    feature_diagnostics = pd.DataFrame(feature_rows)
    real_memory = pd.concat(real_memory_frames, ignore_index=True)
    scalers = pd.DataFrame(scaler_rows)
    baselines = pd.DataFrame(baseline_rows)
    simulation_metadata = pd.DataFrame(metadata_rows)
    correlations = _memory_performance_correlations(
        diagnostic_fold_metrics,
        real_memory,
        synthetic_aggregate,
    )

    selected_cells.to_csv(
        run_dir / "selected_prediction_cells.csv.gz", index=False, compression="gzip"
    )
    diagnostic_cells.to_csv(
        run_dir / "diagnostic_fixed_duration_prediction_cells.csv.gz",
        index=False,
        compression="gzip",
    )
    selected_fold_metrics.to_csv(run_dir / "selected_fold_metrics.csv", index=False)
    selected_pooled_metrics.to_csv(run_dir / "selected_pooled_metrics.csv", index=False)
    diagnostic_fold_metrics.to_csv(
        run_dir / "diagnostic_fixed_duration_fold_metrics.csv", index=False
    )
    diagnostic_pooled_metrics.to_csv(
        run_dir / "diagnostic_fixed_duration_pooled_metrics.csv", index=False
    )
    selections.to_csv(run_dir / "duration_selections.csv", index=False)
    duration_candidates.to_csv(
        run_dir / "duration_candidates.csv.gz", index=False, compression="gzip"
    )
    feature_diagnostics.to_csv(run_dir / "feature_diagnostics.csv", index=False)
    real_memory.to_csv(run_dir / "real_history_memory.csv.gz", index=False, compression="gzip")
    synthetic_tasks.to_csv(
        run_dir / "synthetic_capacity_tasks.csv.gz", index=False, compression="gzip"
    )
    synthetic_candidates.to_csv(
        run_dir / "synthetic_capacity_candidates.csv.gz", index=False, compression="gzip"
    )
    synthetic_summary.to_csv(run_dir / "synthetic_capacity_summary.csv", index=False)
    synthetic_aggregate.to_csv(
        run_dir / "synthetic_capacity_aggregate.csv", index=False
    )
    correlations.to_csv(run_dir / "memory_performance_correlations.csv", index=False)
    baselines.to_csv(run_dir / "baseline_health.csv", index=False)
    scalers.to_csv(run_dir / "channel_scalers.csv", index=False)
    simulation_metadata.to_csv(run_dir / "simulation_metadata.csv", index=False)

    path = selected_pooled_metrics.loc[
        selected_pooled_metrics["scope"].eq("path")
        & ~selected_pooled_metrics["model"].str.startswith("baseline_")
    ].copy()
    best = (
        path.sort_values(
            ["qlike", "rmse", "model", "representation"], kind="mergesort"
        ).iloc[0]
        if not path.empty
        else None
    )
    duration_counts = (
        selections.groupby(
            ["representation", "schedule", "interactions", "selected_duration_us"],
            as_index=False,
        )
        .size()
        .to_dict(orient="records")
    )
    summary = {
        "schema_version": 1,
        "status": "palindrome_duration_memory_performance_assay_complete",
        "test_rows_used": 0,
        "duration_selected_on_outer_validation": False,
        "diagnostic_fixed_duration_outer_curves_are_exploratory": True,
        "qrc_head_fit_intercept": False,
        "har_feature_contract": list(HAR_FEATURES),
        "folds": [int(value) for value in config.folds],
        "representations": list(config.representations),
        "durations_us": [float(value) for value in config.durations_us],
        "schedule_names": list(config.schedule_names),
        "interaction_labels": list(config.interaction_labels),
        "selected_duration_counts": duration_counts,
        "best_duration_selected_path_model": (
            {
                "model": str(best["model"]),
                "representation": str(best["representation"]),
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
            "Only duration selected on the chronological inner tune split is a headline forecast result.",
            "Fixed-duration outer curves are diagnostic and must not be used as post-hoc model selection.",
            "Financial lag recovery receives credit only relative to the endpoint baseline.",
            "Synthetic uniform-input capacity describes intrinsic reservoir behavior, not a financial input by itself.",
            "A duration is not promoted if improvement remains concentrated in one fold or one market episode.",
            "Interaction-on must beat interaction-off under the same representation and duration-selection protocol before interactions receive credit.",
        ],
        "files": {
            "selected_prediction_cells": "selected_prediction_cells.csv.gz",
            "diagnostic_prediction_cells": "diagnostic_fixed_duration_prediction_cells.csv.gz",
            "selected_fold_metrics": "selected_fold_metrics.csv",
            "selected_pooled_metrics": "selected_pooled_metrics.csv",
            "diagnostic_fold_metrics": "diagnostic_fixed_duration_fold_metrics.csv",
            "diagnostic_pooled_metrics": "diagnostic_fixed_duration_pooled_metrics.csv",
            "duration_selections": "duration_selections.csv",
            "duration_candidates": "duration_candidates.csv.gz",
            "feature_diagnostics": "feature_diagnostics.csv",
            "real_history_memory": "real_history_memory.csv.gz",
            "synthetic_capacity_tasks": "synthetic_capacity_tasks.csv.gz",
            "synthetic_capacity_summary": "synthetic_capacity_summary.csv",
            "synthetic_capacity_aggregate": "synthetic_capacity_aggregate.csv",
            "memory_performance_correlations": "memory_performance_correlations.csv",
            "baseline_health": "baseline_health.csv",
            "channel_scalers": "channel_scalers.csv",
            "simulation_metadata": "simulation_metadata.csv",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
