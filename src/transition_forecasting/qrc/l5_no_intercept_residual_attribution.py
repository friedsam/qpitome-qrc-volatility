from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.l5_residual_foundation import (
    L5ResidualFoundationConfig,
    build_fold_residual_registry,
    selected_residual_paths,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import (
    _evolve_segment_batch,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    CrossoverSchedule,
    _branch_drive,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    validate_probe_probabilities,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    elementwise_quadratic_matrix,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _fresh_states,
    _resolve_probe_steps,
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


PRIMARY_MODEL = "palindrome_ordered_interacting"
MODEL_NAMES = (
    PRIMARY_MODEL,
    "palindrome_interaction_off",
    "palindrome_endpoint_preserving_shuffle",
    "raw_input_linear",
    "raw_input_quadratic",
)


@dataclass(frozen=True)
class L5NoInterceptResidualAttributionConfig:
    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    selection_folds: tuple[int, ...] = (4, 5, 6)
    confirmation_folds: tuple[int, ...] = (7, 8)
    lead: int = 5
    target_horizon: int = 10
    min_har_fit_rows: int = 40
    ridge_alpha: float = 100.0
    sequence_length: int = 40
    representation: str = "level_instability"
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    interaction_scale: float = 1.25
    palindrome_schedule: str = "crossover_Ahalf_B_Ahalf"
    row_permutations: int = 256
    target_permutations: int = 128
    seed: int = 20260725
    control_mean_abs_limit: float = 0.05
    control_q90_abs_limit: float = 0.15
    attribution_alpha: float = 0.05

    def validate(self) -> None:
        selection = set(self.selection_folds)
        confirmation = set(self.confirmation_folds)
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if selection & confirmation or selection | confirmation != set(self.folds):
            raise ValueError("selection and confirmation folds must partition folds")
        if any(int(fold) <= 3 for fold in self.folds):
            raise ValueError("assay is restricted to development folds 4+")
        if self.lead != 5 or self.target_horizon != 10:
            raise ValueError("assay is frozen to L5 and the ten-step target path")
        if self.min_har_fit_rows < 40:
            raise ValueError("min_har_fit_rows must be at least 40")
        if self.ridge_alpha <= 0 or self.sequence_length < 5:
            raise ValueError("invalid readout or sequence configuration")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if self.palindrome_schedule not in {item.name for item in CROSSOVER_SCHEDULES}:
            raise ValueError("unknown palindrome schedule")
        if self.row_permutations < 1 or self.target_permutations < 1:
            raise ValueError("attribution permutation counts must be positive")
        if self.control_mean_abs_limit <= 0 or self.control_q90_abs_limit <= 0:
            raise ValueError("control correction limits must be positive")
        if not 0 < self.attribution_alpha < 1:
            raise ValueError("attribution_alpha must lie in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _schedule(name: str) -> CrossoverSchedule:
    for schedule in CROSSOVER_SCHEDULES:
        if schedule.name == name:
            return schedule
    raise ValueError(f"unknown crossover schedule: {name}")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _qlike_cells(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = np.clip(
        2.0 * (np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float)),
        -40.0,
        40.0,
    )
    ratio = np.exp(difference)
    return ratio - difference - 1.0


def _rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def _segment_mask(horizon: np.ndarray, segment: str) -> np.ndarray:
    values = np.asarray(horizon, dtype=int)
    if segment == "all":
        return np.ones(len(values), dtype=bool)
    if segment == "pre_onset":
        return values <= 4
    if segment == "boundary":
        return values == 5
    if segment == "through_boundary":
        return values <= 5
    if segment == "post_onset":
        return values >= 6
    raise ValueError(f"unknown segment: {segment}")


def endpoint_preserving_shuffle(values: np.ndarray, *, seed: int) -> np.ndarray:
    windows = np.asarray(values, dtype=float)
    if windows.ndim != 3 or windows.shape[1] < 2:
        raise ValueError("windows must have shape (samples, time, channels)")
    rng = np.random.default_rng(int(seed))
    result = windows.copy()
    for row in range(len(result)):
        order = rng.permutation(windows.shape[1] - 1)
        result[row, :-1, :] = windows[row, order, :]
    if not np.array_equal(result[:, -1, :], windows[:, -1, :]):
        raise RuntimeError("endpoint-preserving shuffle changed the endpoint")
    return result


def evolve_crossover_control_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    schedule: CrossoverSchedule,
    *,
    interaction_scale: float,
    interactions: bool,
    drive_phase_rad: float = 0.0,
) -> np.ndarray:
    """Reproduce crossover evolution while allowing an exact interaction-off control."""

    schedule.validate()
    reservoir.validate()
    geometry.validate()
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("control evolution requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
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
                float(drive_phase_rad),
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
    return validate_probe_probabilities(
        np.stack(blocks, axis=1), n_atoms=precomputed.n_atoms
    )


def _fit_no_intercept_readout(
    features: np.ndarray,
    residual: np.ndarray,
    fit_mask: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(residual, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if matrix.ndim != 2 or target.ndim != 2 or len(matrix) != len(target):
        raise ValueError("feature and residual matrices must align")
    if fit.shape != (len(matrix),) or not fit.any():
        raise ValueError("fit mask must select aligned rows")
    if not np.isfinite(matrix).all() or not np.isfinite(target[fit]).all():
        raise ValueError("non-finite features or fitted residuals")

    scaler = StandardScaler().fit(matrix[fit])
    transformed = scaler.transform(matrix)
    model = Ridge(alpha=float(alpha), fit_intercept=False).fit(
        transformed[fit], target[fit]
    )
    correction = np.asarray(model.predict(transformed), dtype=float)
    coefficients = np.asarray(model.coef_, dtype=float)
    reconstructed = transformed @ coefficients.T
    intercept = np.asarray(model.intercept_, dtype=float)
    zero_prediction = np.zeros_like(transformed) @ coefficients.T
    feature_std = np.std(matrix[fit], axis=0)
    diagnostics = {
        "fit_rows": int(fit.sum()),
        "feature_width": int(matrix.shape[1]),
        "ridge_alpha": float(alpha),
        "fit_intercept": False,
        "intercept_max_abs": float(np.max(np.abs(np.atleast_1d(intercept)))),
        "manual_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - correction))
        ),
        "zero_feature_max_abs_correction": float(np.max(np.abs(zero_prediction))),
        "constant_feature_columns": int(np.sum(feature_std <= 1e-12)),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "feature_matrix_sha256": _array_sha256(matrix),
        "standardized_feature_sha256": _array_sha256(transformed),
        "correction_sha256": _array_sha256(correction),
    }
    if diagnostics["intercept_max_abs"] != 0.0:
        raise RuntimeError("no-intercept readout acquired a nonzero intercept")
    if diagnostics["manual_reconstruction_max_abs_error"] > 1e-12:
        raise RuntimeError("QRC correction cannot be reconstructed from features")
    if diagnostics["zero_feature_max_abs_correction"] != 0.0:
        raise RuntimeError("zero QRC features did not reproduce the HAR baseline")
    return correction, diagnostics, transformed, coefficients


def _cell_table(
    frame: pd.DataFrame,
    truth: np.ndarray,
    baseline: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    model_name: str,
    fold_group: str,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = truth.shape[1]
    count = len(selected)
    horizon = np.tile(np.arange(1, horizons + 1), count)
    observed = truth[validation].reshape(-1)
    har = baseline[validation].reshape(-1)
    delta = correction[validation].reshape(-1)
    return pd.DataFrame(
        {
            "fold": np.repeat(int(fold), count * horizons),
            "fold_group": np.repeat(str(fold_group), count * horizons),
            "model": np.repeat(str(model_name), count * horizons),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "episode_id": np.repeat(selected["episode_id"].astype(str), horizons),
            "origin_date": np.repeat(selected["origin_date"].astype(str), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "horizon": horizon,
            "y_true": observed,
            "har_prediction": har,
            "har_residual": observed - har,
            "qrc_correction": delta,
            "augmented_prediction": har + delta,
        }
    )


def _metric_payload(cells: pd.DataFrame) -> dict[str, float]:
    truth = cells["y_true"].to_numpy(dtype=float)
    baseline = cells["har_prediction"].to_numpy(dtype=float)
    prediction = cells["augmented_prediction"].to_numpy(dtype=float)
    correction = cells["qrc_correction"].to_numpy(dtype=float)
    residual = cells["har_residual"].to_numpy(dtype=float)
    return {
        "cells": int(len(cells)),
        "samples": int(cells["sample_id"].nunique()),
        "episodes": int(cells["episode_id"].nunique()),
        "har_qlike": float(_qlike_cells(truth, baseline).mean()),
        "augmented_qlike": float(_qlike_cells(truth, prediction).mean()),
        "qlike_delta": float(
            _qlike_cells(truth, prediction).mean()
            - _qlike_cells(truth, baseline).mean()
        ),
        "har_rmse": _rmse(truth, baseline),
        "augmented_rmse": _rmse(truth, prediction),
        "rmse_delta": _rmse(truth, prediction) - _rmse(truth, baseline),
        "mean_correction": float(np.mean(correction)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "q90_abs_correction": float(np.quantile(np.abs(correction), 0.90)),
        "correction_residual_correlation": _safe_correlation(correction, residual),
    }


def build_metric_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    populations = (
        ("all", np.ones(len(cells), dtype=bool)),
        ("control", cells["label"].eq(0).to_numpy()),
        ("transition", cells["label"].eq(1).to_numpy()),
    )
    for (fold, model), group in cells.groupby(["fold", "model"], sort=True):
        for population, population_mask in populations:
            local_population_mask = population_mask[group.index]
            local_population = group.loc[local_population_mask]
            if local_population.empty:
                continue
            for segment in (
                "all",
                "pre_onset",
                "boundary",
                "through_boundary",
                "post_onset",
            ):
                local = local_population.loc[
                    _segment_mask(local_population["horizon"].to_numpy(), segment)
                ]
                if local.empty:
                    continue
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "fold_group": str(local["fold_group"].iloc[0]),
                        "model": str(model),
                        "population": population,
                        "segment": segment,
                        **_metric_payload(local),
                    }
                )
            for horizon, local in local_population.groupby("horizon", sort=True):
                horizon_rows.append(
                    {
                        "fold": int(fold),
                        "fold_group": str(local["fold_group"].iloc[0]),
                        "model": str(model),
                        "population": population,
                        "horizon": int(horizon),
                        **_metric_payload(local),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(horizon_rows)


def _curve_table(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby(
            ["fold_group", "model", "label", "horizon"], as_index=False, sort=True
        )
        .agg(
            samples=("sample_id", "nunique"),
            episodes=("episode_id", "nunique"),
            realized=("y_true", "mean"),
            har=("har_prediction", "mean"),
            qrc_correction=("qrc_correction", "mean"),
            har_plus_qrc=("augmented_prediction", "mean"),
        )
    )


def _episode_weighted_curve_table(cells: pd.DataFrame) -> pd.DataFrame:
    episode = (
        cells.groupby(
            ["fold_group", "model", "label", "episode_id", "horizon"],
            as_index=False,
            sort=True,
        )
        .agg(
            realized=("y_true", "mean"),
            har=("har_prediction", "mean"),
            qrc_correction=("qrc_correction", "mean"),
            har_plus_qrc=("augmented_prediction", "mean"),
        )
    )
    return (
        episode.groupby(
            ["fold_group", "model", "label", "horizon"], as_index=False, sort=True
        )
        .agg(
            episodes=("episode_id", "nunique"),
            realized=("realized", "mean"),
            har=("har", "mean"),
            qrc_correction=("qrc_correction", "mean"),
            har_plus_qrc=("har_plus_qrc", "mean"),
        )
    )


def _null_metric_payload(
    frame: pd.DataFrame,
    truth: np.ndarray,
    baseline: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
) -> dict[str, float]:
    control = validation & frame["label"].eq(0).to_numpy()
    transition = validation & frame["label"].eq(1).to_numpy()
    all_rows = validation

    def metrics(mask: np.ndarray) -> tuple[float, float, float, float]:
        observed = truth[mask]
        har = baseline[mask]
        prediction = har + correction[mask]
        return (
            float(_qlike_cells(observed, prediction).mean() - _qlike_cells(observed, har).mean()),
            float(_rmse(observed, prediction) - _rmse(observed, har)),
            float(np.mean(correction[mask, :5])),
            _safe_correlation(correction[mask], observed - har),
        )

    all_qlike, all_rmse, all_mean, all_correlation = metrics(all_rows)
    control_qlike, control_rmse, control_mean, _ = metrics(control)
    transition_qlike, transition_rmse, transition_mean, _ = metrics(transition)
    return {
        "all_qlike_delta": all_qlike,
        "all_rmse_delta": all_rmse,
        "all_mean_correction_h1_h5": all_mean,
        "all_correction_residual_correlation": all_correlation,
        "control_qlike_delta": control_qlike,
        "control_rmse_delta": control_rmse,
        "control_mean_correction_h1_h5": control_mean,
        "transition_qlike_delta": transition_qlike,
        "transition_rmse_delta": transition_rmse,
        "transition_mean_correction_h1_h5": transition_mean,
        "transition_minus_control_correction_h1_h5": transition_mean - control_mean,
    }


def _plot_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    primary = curves.loc[curves["model"].eq(PRIMARY_MODEL)]
    for fold_group in ("selection", "confirmation"):
        for label, population in ((0, "control"), (1, "transition")):
            local = primary.loc[
                primary["fold_group"].eq(fold_group) & primary["label"].eq(label)
            ].sort_values("horizon")
            if local.empty:
                continue
            figure, axis = plt.subplots(figsize=(10.5, 6.2))
            axis.plot(local["horizon"], local["realized"], marker="o", label="Realized")
            axis.plot(local["horizon"], local["har"], marker="o", label="HAR")
            axis.plot(
                local["horizon"],
                local["har_plus_qrc"],
                marker="o",
                label="HAR + no-intercept QRC",
            )
            axis.axvline(5, linestyle="--", linewidth=1.2)
            axis.set_title(f"L5 {population} paths — {fold_group} folds")
            axis.set_xlabel("Forecast horizon")
            axis.set_ylabel("Mean future log volatility")
            axis.set_xticks(range(1, 11))
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                output_dir / f"{fold_group}_{population}_paths.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

        local = primary.loc[primary["fold_group"].eq(fold_group)].copy()
        if local.empty:
            continue
        figure, axis = plt.subplots(figsize=(10.5, 6.2))
        for label, population in ((0, "Control"), (1, "Transition")):
            population_rows = local.loc[local["label"].eq(label)].sort_values("horizon")
            axis.plot(
                population_rows["horizon"],
                population_rows["qrc_correction"],
                marker="o",
                label=population,
            )
        axis.axhline(0.0, linewidth=1.0)
        axis.axvline(5, linestyle="--", linewidth=1.2)
        axis.set_title(f"No-intercept QRC correction paths — {fold_group} folds")
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel("Mean QRC correction")
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{fold_group}_correction_paths.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def _null_p_value(values: pd.Series, observed: float, *, larger_is_better: bool) -> float:
    finite = values[np.isfinite(values)].to_numpy(dtype=float)
    if not len(finite):
        return float("nan")
    if larger_is_better:
        return float((1 + np.sum(finite >= observed)) / (len(finite) + 1))
    return float((1 + np.sum(finite <= observed)) / (len(finite) + 1))


def run_l5_no_intercept_residual_attribution(
    *,
    fold_dir: Path,
    results_root: Path,
    config: L5NoInterceptResidualAttributionConfig = (
        L5NoInterceptResidualAttributionConfig()
    ),
    candidate_features: CandidateFeatureConfig = CandidateFeatureConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    config.validate()
    candidate_features.validate()
    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    reservoir_config = reservoir or TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.020,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seed,
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    reservoir_config.validate()
    geometry_config.validate()
    if reservoir_config.n_atoms != 6 or reservoir_config.shots is not None:
        raise ValueError("assay requires an exact six-atom reservoir")

    residual_config = L5ResidualFoundationConfig(
        folds=config.folds,
        selection_folds=config.selection_folds,
        confirmation_folds=config.confirmation_folds,
        lead=config.lead,
        target_horizon=config.target_horizon,
        min_fit_rows=config.min_har_fit_rows,
        ridge_alpha=config.ridge_alpha,
        models=("mse_har",),
    )
    schedule = _schedule(config.palindrome_schedule)
    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir_config.to_dict(),
            "geometry": geometry_config.to_dict(),
            "baseline": "plain target-available causal MSE-HAR",
            "qrc_readout_fit_intercept": False,
            "qlike_offset": False,
            "post_hoc_calibration": False,
            "test_rows_allowed": False,
            "qrc_row_selection_before_baseline": False,
        },
        run_id=run_id,
    )

    cell_tables: list[pd.DataFrame] = []
    residual_registries: list[pd.DataFrame] = []
    readout_diagnostics: list[dict[str, object]] = []
    fold_diagnostics: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    cache: dict[str, np.ndarray] = {}

    for fold in config.folds:
        registry, residual_diagnostic = build_fold_residual_registry(
            dataset.manifest,
            fold=int(fold),
            config=residual_config,
        )
        registry = registry.loc[registry["model"].eq("mse_har")].copy()
        residual_registries.append(registry)
        keys, truth, baseline, residual = selected_residual_paths(
            registry,
            model_name="mse_har",
            target_horizon=config.target_horizon,
        )
        if keys["fold_split"].eq("test").any():
            raise RuntimeError("residual registry contains test rows")
        local_manifest = dataset.manifest.loc[
            dataset.manifest["fold"].eq(int(fold))
            & dataset.manifest["lead"].eq(config.lead)
            & dataset.manifest["fold_split"].isin(("train", "val"))
        ].copy()
        if local_manifest.duplicated("sample_id").any():
            raise ValueError(f"fold {fold}: duplicate sample_id in fold manifest")
        local_manifest = local_manifest.sort_values(
            ["origin_date", "fold_split", "label", "episode_id", "sample_id"]
        ).reset_index(drop=True)
        tensor_rows = local_manifest["_tensor_row"].to_numpy(dtype=int)
        levels = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        if not np.isfinite(levels).all():
            raise ValueError(f"fold {fold}: non-finite level windows")
        train_all = local_manifest["fold_split"].eq("train").to_numpy()
        raw_sequences = build_candidate_sequences(
            levels,
            config.representation,
            candidate_features,
        )
        channel_scaler = fit_channel_scaler(
            raw_sequences,
            train_all,
            q_low=candidate_features.q_low,
            q_high=candidate_features.q_high,
        )
        encoded = transform_candidate_sequences(raw_sequences, channel_scaler)

        aligned = keys.merge(
            local_manifest[
                ["sample_id", "_tensor_row", "label", "episode_id", "origin_date", "fold_split"]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_manifest"),
        )
        if aligned["_tensor_row"].isna().any():
            raise RuntimeError(f"fold {fold}: residual rows do not align to tensors")
        for column in ("label", "episode_id", "origin_date", "fold_split"):
            other = f"{column}_manifest"
            if other in aligned and not aligned[column].astype(str).equals(
                aligned[other].astype(str)
            ):
                raise RuntimeError(f"fold {fold}: residual/tensor mismatch in {column}")
        row_lookup = {
            str(sample_id): int(index)
            for index, sample_id in enumerate(local_manifest["sample_id"].astype(str))
        }
        aligned_indices = np.asarray(
            [row_lookup[str(value)] for value in aligned["sample_id"]], dtype=int
        )
        encoded_aligned = encoded[aligned_indices]
        fit_mask = aligned["fold_split"].eq("train").to_numpy()
        validation = aligned["fold_split"].eq("val").to_numpy()
        if not fit_mask.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty residual fit or validation split")

        ordered_probabilities, ordered_metadata = evolve_crossover_probabilities(
            encoded_aligned,
            replace(reservoir_config, shot_seed=config.seed + int(fold)),
            geometry_config,
            schedule,
            interaction_scale=config.interaction_scale,
            drive_phase_rad=0.0,
        )
        control_on = evolve_crossover_control_probabilities(
            encoded_aligned,
            replace(reservoir_config, shot_seed=config.seed + int(fold)),
            geometry_config,
            schedule,
            interaction_scale=config.interaction_scale,
            interactions=True,
        )
        on_equivalence = float(np.max(np.abs(ordered_probabilities - control_on)))
        if on_equivalence > 1e-12:
            raise RuntimeError("interaction-control implementation disagrees with primary QRC")
        off_probabilities = evolve_crossover_control_probabilities(
            encoded_aligned,
            replace(reservoir_config, shot_seed=config.seed + int(fold)),
            geometry_config,
            schedule,
            interaction_scale=config.interaction_scale,
            interactions=False,
        )
        shuffled = endpoint_preserving_shuffle(
            encoded_aligned, seed=config.seed + 1000 + int(fold)
        )
        shuffled_probabilities, _ = evolve_crossover_probabilities(
            shuffled,
            replace(reservoir_config, shot_seed=config.seed + int(fold)),
            geometry_config,
            schedule,
            interaction_scale=config.interaction_scale,
            drive_phase_rad=0.0,
        )
        feature_matrices = {
            PRIMARY_MODEL: build_crossover_feature_banks(ordered_probabilities)[
                "occupation_pair_raw"
            ],
            "palindrome_interaction_off": build_crossover_feature_banks(
                off_probabilities
            )["occupation_pair_raw"],
            "palindrome_endpoint_preserving_shuffle": build_crossover_feature_banks(
                shuffled_probabilities
            )["occupation_pair_raw"],
            "raw_input_linear": encoded_aligned.reshape(len(encoded_aligned), -1),
            "raw_input_quadratic": elementwise_quadratic_matrix(encoded_aligned),
        }

        fitted: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for model_name, features in feature_matrices.items():
            correction, diagnostics, transformed, coefficients = _fit_no_intercept_readout(
                features,
                residual,
                fit_mask,
                alpha=config.ridge_alpha,
            )
            diagnostics.update(
                {
                    "fold": int(fold),
                    "fold_group": (
                        "selection"
                        if int(fold) in config.selection_folds
                        else "confirmation"
                    ),
                    "model": model_name,
                    "baseline_sha256": _array_sha256(baseline),
                    "truth_sha256": _array_sha256(truth),
                    "residual_sha256": _array_sha256(residual),
                    "input_scaler": channel_scaler.to_dict(),
                    "primary_control_probability_max_abs_error": on_equivalence,
                }
            )
            validation_correction = correction[validation]
            horizon_mean = validation_correction.mean(axis=0, keepdims=True)
            denominator = float(np.sum(validation_correction**2))
            diagnostics["validation_mean_component_energy_fraction"] = (
                float(len(validation_correction) * np.sum(horizon_mean**2) / denominator)
                if denominator > 0
                else 0.0
            )
            diagnostics["validation_control_q90_abs_correction_h1_h5"] = float(
                np.quantile(
                    np.abs(
                        correction[
                            validation & aligned["label"].eq(0).to_numpy(), :5
                        ]
                    ),
                    0.90,
                )
            )
            readout_diagnostics.append(diagnostics)
            fitted[model_name] = (correction, transformed, coefficients)
            cell_tables.append(
                _cell_table(
                    aligned,
                    truth,
                    baseline,
                    correction,
                    validation,
                    fold=int(fold),
                    model_name=model_name,
                    fold_group=(
                        "selection"
                        if int(fold) in config.selection_folds
                        else "confirmation"
                    ),
                )
            )
            cache[f"fold_{fold}__{model_name}"] = np.asarray(features, dtype=float)

        primary_correction, transformed, _ = fitted[PRIMARY_MODEL]
        rng = np.random.default_rng(config.seed + 10000 * int(fold))
        validation_indices = np.flatnonzero(validation)
        for iteration in range(config.row_permutations):
            permutation = rng.permutation(validation_indices)
            permuted = np.zeros_like(primary_correction)
            permuted[validation_indices] = primary_correction[permutation]
            null_rows.append(
                {
                    "fold": int(fold),
                    "fold_group": (
                        "selection"
                        if int(fold) in config.selection_folds
                        else "confirmation"
                    ),
                    "null": "validation_qrc_row_permutation",
                    "iteration": int(iteration),
                    **_null_metric_payload(
                        aligned, truth, baseline, permuted, validation
                    ),
                }
            )
        fit_indices = np.flatnonzero(fit_mask)
        for iteration in range(config.target_permutations):
            permutation = rng.permutation(fit_indices)
            model = Ridge(alpha=config.ridge_alpha, fit_intercept=False).fit(
                transformed[fit_mask], residual[permutation]
            )
            permuted = np.asarray(model.predict(transformed), dtype=float)
            null_rows.append(
                {
                    "fold": int(fold),
                    "fold_group": (
                        "selection"
                        if int(fold) in config.selection_folds
                        else "confirmation"
                    ),
                    "null": "training_residual_row_permutation",
                    "iteration": int(iteration),
                    **_null_metric_payload(
                        aligned, truth, baseline, permuted, validation
                    ),
                }
            )

        fold_diagnostics.append(
            {
                **residual_diagnostic,
                "fold": int(fold),
                "residual_rows": int(len(aligned)),
                "residual_fit_rows": int(fit_mask.sum()),
                "validation_rows": int(validation.sum()),
                "test_rows_used": 0,
                "ordered_metadata": ordered_metadata,
                "input_scaler": channel_scaler.to_dict(),
                "encoded_clip_fraction": float(
                    np.mean(np.abs(encoded_aligned) >= 1.0 - 1e-12)
                ),
            }
        )

    cells = pd.concat(cell_tables, ignore_index=True)
    residual_registry = pd.concat(residual_registries, ignore_index=True)
    fold_metrics, horizon_metrics = build_metric_tables(cells)
    curves = _curve_table(cells)
    episode_curves = _episode_weighted_curve_table(cells)
    diagnostics = pd.DataFrame(readout_diagnostics)
    nulls = pd.DataFrame(null_rows)

    residual_registry.to_csv(run_dir / "plain_mse_har_residual_registry.csv", index=False)
    cells.to_csv(run_dir / "validation_cells.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    curves.to_csv(run_dir / "forecast_and_correction_curves.csv", index=False)
    episode_curves.to_csv(run_dir / "episode_weighted_curves.csv", index=False)
    diagnostics.to_csv(run_dir / "feature_attribution_diagnostics.csv", index=False)
    nulls.to_csv(run_dir / "attribution_null_metrics.csv", index=False)
    np.savez_compressed(run_dir / "feature_cache.npz", **cache)
    (run_dir / "fold_diagnostics.json").write_text(
        json.dumps(fold_diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    _plot_curves(curves, run_dir)

    primary_metrics = fold_metrics.loc[
        fold_metrics["model"].eq(PRIMARY_MODEL)
        & fold_metrics["segment"].eq("through_boundary")
    ].copy()
    confirmation = primary_metrics.loc[
        primary_metrics["fold"].isin(config.confirmation_folds)
    ]
    transition = confirmation.loc[confirmation["population"].eq("transition")]
    control = confirmation.loc[confirmation["population"].eq("control")]
    observed_by_fold = []
    for fold in config.confirmation_folds:
        local_cells = cells.loc[
            cells["fold"].eq(int(fold)) & cells["model"].eq(PRIMARY_MODEL)
        ]
        local_frame = local_cells.drop_duplicates("sample_id")[
            ["sample_id", "episode_id", "label", "origin_date"]
        ]
        observed_by_fold.append(
            {
                "fold": int(fold),
                "transition_mean_correction_h1_h5": float(
                    local_cells.loc[
                        local_cells["label"].eq(1)
                        & local_cells["horizon"].le(5),
                        "qrc_correction",
                    ].mean()
                ),
                "control_mean_correction_h1_h5": float(
                    local_cells.loc[
                        local_cells["label"].eq(0)
                        & local_cells["horizon"].le(5),
                        "qrc_correction",
                    ].mean()
                ),
                "samples": int(local_frame["sample_id"].nunique()),
                "episodes": int(local_frame["episode_id"].nunique()),
            }
        )

    observed = nulls.iloc[0:0].copy()
    observed_rows: list[dict[str, object]] = []
    for fold in config.folds:
        local = cells.loc[
            cells["fold"].eq(int(fold)) & cells["model"].eq(PRIMARY_MODEL)
        ]
        keys = local.drop_duplicates("sample_id").sort_values("sample_id")
        truth = local.pivot(index="sample_id", columns="horizon", values="y_true").loc[
            keys["sample_id"]
        ].to_numpy()
        baseline = local.pivot(
            index="sample_id", columns="horizon", values="har_prediction"
        ).loc[keys["sample_id"]].to_numpy()
        correction = local.pivot(
            index="sample_id", columns="horizon", values="qrc_correction"
        ).loc[keys["sample_id"]].to_numpy()
        frame = keys[["sample_id", "label"]].reset_index(drop=True)
        validation = np.ones(len(frame), dtype=bool)
        observed_rows.append(
            {
                "fold": int(fold),
                "fold_group": (
                    "selection" if int(fold) in config.selection_folds else "confirmation"
                ),
                "null": "observed",
                "iteration": -1,
                **_null_metric_payload(frame, truth, baseline, correction, validation),
            }
        )
    observed = pd.DataFrame(observed_rows)

    null_summary: list[dict[str, object]] = []
    for fold_group in ("selection", "confirmation"):
        observed_group = observed.loc[observed["fold_group"].eq(fold_group)].mean(
            numeric_only=True
        )
        for null_name, group in nulls.loc[
            nulls["fold_group"].eq(fold_group)
        ].groupby("null", sort=True):
            aggregated = group.groupby("iteration").mean(numeric_only=True)
            for metric, larger_is_better in (
                ("all_correction_residual_correlation", True),
                ("transition_minus_control_correction_h1_h5", True),
                ("transition_mean_correction_h1_h5", True),
                ("transition_qlike_delta", False),
                ("transition_rmse_delta", False),
            ):
                value = float(observed_group[metric])
                null_summary.append(
                    {
                        "fold_group": fold_group,
                        "null": str(null_name),
                        "metric": metric,
                        "observed": value,
                        "null_q05": float(aggregated[metric].quantile(0.05)),
                        "null_median": float(aggregated[metric].median()),
                        "null_q95": float(aggregated[metric].quantile(0.95)),
                        "p_value": _null_p_value(
                            aggregated[metric],
                            value,
                            larger_is_better=larger_is_better,
                        ),
                    }
                )
    null_summary_frame = pd.DataFrame(null_summary)
    null_summary_frame.to_csv(run_dir / "attribution_null_summary.csv", index=False)

    confirmation_null = null_summary_frame.loc[
        null_summary_frame["fold_group"].eq("confirmation")
        & null_summary_frame["metric"].eq("all_correction_residual_correlation")
    ]
    attribution_pass = bool(
        len(confirmation_null)
        and confirmation_null["p_value"].lt(config.attribution_alpha).all()
    )
    control_pass = bool(
        len(control) == len(config.confirmation_folds)
        and control["mean_correction"].abs().le(config.control_mean_abs_limit).all()
        and control["q90_abs_correction"].le(config.control_q90_abs_limit).all()
    )
    transition_positive = bool(
        len(transition) == len(config.confirmation_folds)
        and transition["mean_correction"].gt(0.0).all()
    )
    transition_loss_pass = bool(
        len(transition) == len(config.confirmation_folds)
        and transition["qlike_delta"].lt(0.0).all()
        and transition["rmse_delta"].lt(0.0).all()
    )
    gap_pass = all(
        item["transition_mean_correction_h1_h5"]
        > item["control_mean_correction_h1_h5"]
        for item in observed_by_fold
    )
    summary = {
        "status": "l5_no_intercept_residual_attribution_complete",
        "financial_test_rows_used": 0,
        "baseline": "plain target-available causal MSE-HAR",
        "qrc_correction_identity": (
            "HAR_plus_QRC = HAR + StandardScaler(QRC_features) @ coefficients.T"
        ),
        "prohibited_components": [
            "QRC intercept",
            "constant feature",
            "QLIKE offset",
            "segmented lambda",
            "post-hoc calibration",
        ],
        "confirmation_observed_by_fold": observed_by_fold,
        "success_pattern": {
            "control_small_and_bounded": control_pass,
            "transition_positive_through_L5_each_fold": transition_positive,
            "transition_qlike_and_rmse_improve_each_fold": transition_loss_pass,
            "transition_correction_exceeds_control_each_fold": bool(gap_pass),
            "feature_attribution_nulls_rejected": attribution_pass,
            "overall_pass": bool(
                control_pass
                and transition_positive
                and transition_loss_pass
                and gap_pass
                and attribution_pass
            ),
        },
        "interpretation_rule": (
            "A significant feature-attribution result is not sufficient. Promotion requires "
            "the predeclared control/transition correction pattern and foldwise confirmation."
        ),
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
