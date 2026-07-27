"""Submission-oriented noise robustness assay for the frozen palindrome QRC."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _mean_qlike,
    _resolve_schedule,
    _rmse,
    evolve_palindrome_probabilities,
    validate_har_contract,
)
from transition_forecasting.qrc.palindrome_rydberg_noise import (
    build_noisy_palindrome_probabilities,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)
from transition_forecasting.qrc.temporal_rydberg_noise import TemporalNoiseSpec


@dataclass(frozen=True)
class PalindromeNoiseAssayConfig:
    """Frozen bounded protocol; scenario filtering is execution-only, not tuning."""

    fold: int = 5
    lead: int = 5
    sequence_length: int = 40
    max_per_class: int = 2
    selection_seed: int = 20260726
    representation: str = "level_instability"
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    schedule_name: str = "crossover_Ahalf_B_Ahalf"
    feature_bank: str = "six_mode_density_curvature"
    ridge_alpha: float = 100.0
    correction_lambda: float = 1.0
    prequential_blocks: int = 5
    amplitude_damping_t1_us: tuple[float, ...] = (200.0, 100.0, 50.0)
    dephasing_t2_us: tuple[float, ...] = (100.0, 50.0)
    depolarizing_probabilities: tuple[float, ...] = (0.005, 0.01, 0.03)
    include_combined_moderate: bool = True
    selected_scenarios: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.fold < 1 or self.lead < 1:
            raise ValueError("fold and lead must be positive")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.representation != "level_instability":
            raise ValueError("submission noise assay requires level_instability")
        if self.feature_bank != "six_mode_density_curvature":
            raise ValueError("submission noise assay requires six_mode_density_curvature")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.correction_lambda < 0:
            raise ValueError("correction_lambda must be nonnegative")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if any(value <= 0 for value in self.amplitude_damping_t1_us):
            raise ValueError("all T1 values must be positive")
        if any(value <= 0 for value in self.dephasing_t2_us):
            raise ValueError("all T2 values must be positive")
        if any(not 0.0 < value < 1.0 for value in self.depolarizing_probabilities):
            raise ValueError("depolarizing probabilities must lie in (0, 1)")
        if len(set(self.selected_scenarios)) != len(self.selected_scenarios):
            raise ValueError("selected_scenarios must be unique")
        _resolve_schedule(self.schedule_name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _all_noise_scenarios(config: PalindromeNoiseAssayConfig) -> tuple[TemporalNoiseSpec, ...]:
    scenarios: list[TemporalNoiseSpec] = [TemporalNoiseSpec(name="ideal_density")]
    scenarios.extend(
        TemporalNoiseSpec(
            name=f"amplitude_damping_T1_{value:g}us",
            amplitude_damping_t1_us=float(value),
        )
        for value in config.amplitude_damping_t1_us
    )
    scenarios.extend(
        TemporalNoiseSpec(
            name=f"dephasing_T2_{value:g}us",
            dephasing_t2_us=float(value),
        )
        for value in config.dephasing_t2_us
    )
    scenarios.extend(
        TemporalNoiseSpec(
            name=f"depolarizing_p_{value:g}",
            depolarizing_probability=float(value),
        )
        for value in config.depolarizing_probabilities
    )
    if config.include_combined_moderate:
        scenarios.append(
            TemporalNoiseSpec(
                name="combined_T1_100us_T2_50us_p_0.01",
                amplitude_damping_t1_us=100.0,
                dephasing_t2_us=50.0,
                depolarizing_probability=0.01,
            )
        )
    for scenario in scenarios:
        scenario.validate()
    return tuple(scenarios)


def noise_scenarios(config: PalindromeNoiseAssayConfig) -> tuple[TemporalNoiseSpec, ...]:
    """Return ideal plus all requested scenarios in canonical order."""

    config.validate()
    scenarios = _all_noise_scenarios(config)
    if not config.selected_scenarios:
        return scenarios
    lookup = {scenario.name: scenario for scenario in scenarios}
    unknown = sorted(set(config.selected_scenarios).difference(lookup))
    if unknown:
        raise ValueError(f"unknown selected noise scenarios: {unknown}")
    return (lookup["ideal_density"],) + tuple(
        lookup[name]
        for name in config.selected_scenarios
        if name != "ideal_density"
    )


def _channel_name(scenario: TemporalNoiseSpec) -> str:
    active = sum(
        [
            scenario.amplitude_damping_t1_us is not None,
            scenario.dephasing_t2_us is not None,
            scenario.depolarizing_probability > 0.0,
        ]
    )
    if scenario.is_ideal:
        return "ideal"
    if active > 1:
        return "combined"
    if scenario.amplitude_damping_t1_us is not None:
        return "amplitude_damping"
    if scenario.dephasing_t2_us is not None:
        return "dephasing"
    return "depolarizing"


def _feature_distortion(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    clean = np.asarray(reference, dtype=float)
    noisy = np.asarray(candidate, dtype=float)
    if clean.shape != noisy.shape:
        raise ValueError("feature distortion inputs must share one shape")
    difference = noisy - clean
    clean_scale = max(float(np.mean(np.abs(clean))), 1e-12)
    flat_clean = clean.reshape(-1)
    flat_noisy = noisy.reshape(-1)
    correlation = (
        float(np.corrcoef(flat_clean, flat_noisy)[0, 1])
        if np.std(flat_clean) > 0 and np.std(flat_noisy) > 0
        else np.nan
    )
    return {
        "feature_mae": float(np.mean(np.abs(difference))),
        "feature_rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_feature_mae": float(np.mean(np.abs(difference)) / clean_scale),
        "feature_correlation": correlation,
        "max_absolute_feature_change": float(np.max(np.abs(difference), initial=0.0)),
    }


def _sign_accuracy(actual_residual: np.ndarray, correction: np.ndarray) -> float:
    actual = np.asarray(actual_residual, dtype=float).reshape(-1)
    predicted = np.asarray(correction, dtype=float).reshape(-1)
    informative = np.abs(actual) > 1e-12
    if not informative.any():
        return float("nan")
    return float(np.mean(np.sign(actual[informative]) == np.sign(predicted[informative])))


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    scenario: TemporalNoiseSpec,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    y_val = y[validation]
    har_val = har[validation]
    pred_val = prediction[validation]
    correction_val = correction[validation]
    for sample_index, metadata in selected.iterrows():
        for horizon in range(y.shape[1]):
            rows.append(
                {
                    "scenario": scenario.name,
                    "sample_id": str(metadata["sample_id"]),
                    "fold": int(metadata["fold"]),
                    "lead": int(metadata["lead"]),
                    "label": int(metadata["label"]),
                    "origin_date": str(metadata["origin_date"]),
                    "horizon": int(horizon + 1),
                    "y_true": float(y_val[sample_index, horizon]),
                    "har_prediction": float(har_val[sample_index, horizon]),
                    "prediction": float(pred_val[sample_index, horizon]),
                    "correction": float(correction_val[sample_index, horizon]),
                }
            )
    return pd.DataFrame(rows)


def _scenario_metric_row(
    *,
    scenario: TemporalNoiseSpec,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
    clean_features: np.ndarray,
    features: np.ndarray,
    elapsed_seconds: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    y_val = y[validation]
    har_val = har[validation]
    prediction_val = prediction[validation]
    correction_val = correction[validation]
    actual_residual = y_val - har_val
    return {
        "scenario": scenario.name,
        "channel": _channel_name(scenario),
        "amplitude_damping_t1_us": scenario.amplitude_damping_t1_us,
        "dephasing_t2_us": scenario.dephasing_t2_us,
        "depolarizing_probability": scenario.depolarizing_probability,
        "validation_qlike": _mean_qlike(y_val, prediction_val),
        "validation_rmse": _rmse(y_val, prediction_val),
        "har_validation_qlike": _mean_qlike(y_val, har_val),
        "har_validation_rmse": _rmse(y_val, har_val),
        "residual_sign_accuracy": _sign_accuracy(actual_residual, correction_val),
        "mean_correction": float(np.mean(correction_val)),
        "correction_rms": float(np.sqrt(np.mean(correction_val**2))),
        "wall_seconds": float(elapsed_seconds),
        "seconds_per_sample": float(elapsed_seconds / len(features)),
        "total_substeps": int(metadata["total_substeps"]),
        "max_trace_drift_before_renormalization": float(
            metadata["max_trace_drift_before_renormalization"]
        ),
        "max_hermiticity_error": float(metadata["max_hermiticity_error"]),
        **_feature_distortion(clean_features, features),
    }


def _render_plots(metrics: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    local = metrics.copy()
    outputs: list[str] = []
    positions = np.arange(len(local))

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.bar(positions - 0.18, local["delta_qlike_vs_ideal"], width=0.36, label="ΔQLIKE")
    axis.bar(positions + 0.18, local["delta_rmse_vs_ideal"], width=0.36, label="ΔRMSE")
    axis.axhline(0.0, linewidth=1.0)
    axis.set(
        title="Frozen palindrome-head degradation under local noise",
        xlabel="Noise scenario",
        ylabel="Metric change versus ideal density simulation",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(local["scenario"], rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "forecast_metric_change_vs_ideal.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.bar(positions, local["relative_feature_mae"])
    axis.set(
        title="Six-mode palindrome feature distortion under noise",
        xlabel="Noise scenario",
        ylabel="Mean absolute feature change / mean |ideal feature|",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(local["scenario"], rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    filename = "feature_distortion_by_noise.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.bar(positions, local["feature_correlation"])
    axis.set(
        title="Correlation with ideal six-mode features",
        xlabel="Noise scenario",
        ylabel="Feature correlation",
        ylim=(0.0, 1.01),
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(local["scenario"], rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    filename = "feature_correlation_by_noise.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_palindrome_noise_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: PalindromeNoiseAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
    run_id: str,
) -> Path:
    """Run a fixed-head density-matrix robustness study without test rows."""

    assay.validate()
    candidate_features.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("noise assay requires an exact six-atom reservoir")
    if not np.isclose(reservoir.step_duration_us, 0.02):
        raise ValueError("noise assay requires 0.02-us palindrome steps")
    if tuple(reservoir.probe_fractions) != (0.25, 0.5, 1.0):
        raise ValueError("noise assay requires probes at 1/4, 1/2, and endpoint")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=assay.level_channel_name,
        fallback=assay.fallback_level_channel,
    )
    frame = _select_rows_for_fold(
        dataset.manifest,
        fold=int(assay.fold),
        leads=(int(assay.lead),),
        max_per_class=int(assay.max_per_class),
        seed=int(assay.selection_seed),
    )
    if frame["fold_split"].eq("test").any():
        raise RuntimeError("noise assay must not receive test rows")
    tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
    level = extract_level_windows(
        dataset,
        tensor_rows,
        sequence_length=int(assay.sequence_length),
        level_channel=level_channel,
    )
    usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
    frame = frame.loc[usable].reset_index(drop=True)
    level = level[usable]
    tensor_rows = tensor_rows[usable]
    if frame.empty:
        raise RuntimeError("noise assay selected no valid rows")
    validate_har_contract(frame)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if train.sum() < 4 or validation.sum() < 2:
        raise RuntimeError(
            "noise assay requires at least four train and two validation rows"
        )
    har = _fit_har(frame, y, train)
    residuals, residual_train = _prequential_har_residuals(
        frame,
        y,
        train,
        blocks=int(assay.prequential_blocks),
    )
    if residual_train.sum() < 4:
        raise RuntimeError("noise assay has insufficient causal residual-fit rows")

    raw_sequences = build_candidate_sequences(
        level,
        assay.representation,
        candidate_features,
    )
    channel_scaler = fit_channel_scaler(
        raw_sequences,
        train,
        q_low=candidate_features.q_low,
        q_high=candidate_features.q_high,
    )
    windows = transform_candidate_sequences(raw_sequences, channel_scaler)
    schedule = _resolve_schedule(assay.schedule_name)

    clean_started = time.perf_counter()
    clean_probabilities, clean_metadata = evolve_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=float(interaction_scale),
        interactions=True,
        drive_phase_rad=float(drive_phase_rad),
    )
    clean_seconds = float(time.perf_counter() - clean_started)
    clean_features = np.asarray(
        build_crossover_feature_banks(clean_probabilities)[assay.feature_bank],
        dtype=float,
    )
    feature_scaler = StandardScaler().fit(clean_features[residual_train])
    clean_design = feature_scaler.transform(clean_features)
    readout = Ridge(alpha=float(assay.ridge_alpha), fit_intercept=False)
    readout.fit(clean_design[residual_train], residuals[residual_train])
    clean_raw_correction = np.asarray(readout.predict(clean_design), dtype=float)
    clean_correction = float(assay.correction_lambda) * clean_raw_correction
    clean_prediction = har + clean_correction

    scenarios = noise_scenarios(assay)
    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "assay": assay.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "geometry": geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "drive_phase_rad": float(drive_phase_rad),
            "noise_scenarios": [scenario.to_dict() for scenario in scenarios],
            "readout_policy": (
                "StandardScaler and Ridge(alpha=100, fit_intercept=False) fit once "
                "on ideal six-mode causal HAR residuals and frozen for every scenario"
            ),
            "test_rows_allowed": False,
            "scientific_scope": "Final-palindrome robustness/compliance; not model selection",
        },
        run_id=run_id,
    )
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "frozen_readout.npz",
        feature_scaler_mean=feature_scaler.mean_,
        feature_scaler_scale=feature_scaler.scale_,
        ridge_coef=np.asarray(readout.coef_, dtype=float),
        ridge_alpha=np.asarray([assay.ridge_alpha], dtype=float),
        correction_lambda=np.asarray([assay.correction_lambda], dtype=float),
        residual_train_mask=residual_train,
    )

    scenario_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    parity: dict[str, float] = {}
    for scenario in scenarios:
        started = time.perf_counter()
        probabilities, metadata = build_noisy_palindrome_probabilities(
            windows,
            reservoir,
            geometry,
            schedule,
            scenario,
            interaction_scale=float(interaction_scale),
            drive_phase_rad=float(drive_phase_rad),
            interactions=True,
        )
        elapsed = float(time.perf_counter() - started)
        features = np.asarray(
            build_crossover_feature_banks(probabilities)[assay.feature_bank],
            dtype=float,
        )
        if scenario.is_ideal:
            parity = _feature_distortion(clean_features, features)
            if parity["max_absolute_feature_change"] > 1e-8:
                raise RuntimeError(
                    "ideal density palindrome does not match the statevector reference"
                )
        standardized = feature_scaler.transform(features)
        raw_correction = np.asarray(readout.predict(standardized), dtype=float)
        correction = float(assay.correction_lambda) * raw_correction
        prediction = har + correction
        scenario_rows.append(
            _scenario_metric_row(
                scenario=scenario,
                y=y,
                har=har,
                prediction=prediction,
                correction=correction,
                validation=validation,
                clean_features=clean_features,
                features=features,
                elapsed_seconds=elapsed,
                metadata=metadata,
            )
        )
        prediction_frames.append(
            _prediction_rows(
                frame,
                scenario=scenario,
                y=y,
                har=har,
                prediction=prediction,
                correction=correction,
                validation=validation,
            )
        )
        np.savez_compressed(
            feature_dir / f"{scenario.name}.npz",
            probabilities=probabilities,
            features=features,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            train_mask=train,
            residual_train_mask=residual_train,
            validation_mask=validation,
        )
        (feature_dir / f"{scenario.name}_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    metrics = pd.DataFrame(scenario_rows)
    ideal = metrics.loc[metrics["channel"].eq("ideal")].iloc[0]
    metrics["delta_qlike_vs_ideal"] = (
        metrics["validation_qlike"] - float(ideal["validation_qlike"])
    )
    metrics["delta_rmse_vs_ideal"] = (
        metrics["validation_rmse"] - float(ideal["validation_rmse"])
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    plots = _render_plots(metrics, run_dir / "plots")

    retained_columns = [
        "sample_id",
        "fold",
        "fold_split",
        "lead",
        "label",
        "origin_date",
    ]
    if "episode_id" in frame.columns:
        retained_columns.append("episode_id")
    retained = frame[retained_columns].copy()
    retained["tensor_row"] = tensor_rows
    retained["train_mask"] = train
    retained["residual_train_mask"] = residual_train
    retained["validation_mask"] = validation
    retained.to_csv(run_dir / "retained_samples.csv", index=False)
    metrics.to_csv(run_dir / "noise_metrics.csv", index=False)
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    ideal_qrc_qlike = _mean_qlike(y[validation], clean_prediction[validation])
    ideal_qrc_rmse = _rmse(y[validation], clean_prediction[validation])
    har_qlike = _mean_qlike(y[validation], har[validation])
    har_rmse = _rmse(y[validation], har[validation])
    summary = {
        "schema_version": 1,
        "status": "palindrome_noise_assay_complete",
        "test_rows_used": 0,
        "fold": int(assay.fold),
        "lead": int(assay.lead),
        "samples": int(len(frame)),
        "train_samples": int(train.sum()),
        "residual_train_samples": int(residual_train.sum()),
        "validation_samples": int(validation.sum()),
        "architecture": {
            "representation": assay.representation,
            "schedule": assay.schedule_name,
            "feature_bank": assay.feature_bank,
            "feature_width": int(clean_features.shape[1]),
            "ridge_alpha": float(assay.ridge_alpha),
            "fit_intercept": False,
            "correction_lambda": float(assay.correction_lambda),
            "interaction_scale": float(interaction_scale),
            "step_duration_us": float(reservoir.step_duration_us),
            "probe_fractions": list(reservoir.probe_fractions),
        },
        "ideal_statevector_metrics": {
            "validation_qlike": ideal_qrc_qlike,
            "validation_rmse": ideal_qrc_rmse,
            "qlike_delta_vs_har": ideal_qrc_qlike - har_qlike,
            "rmse_delta_vs_har": ideal_qrc_rmse - har_rmse,
            "wall_seconds": clean_seconds,
        },
        "statevector_density_feature_parity": parity,
        "clean_statevector_metadata": clean_metadata,
        "channel_scaler": channel_scaler.to_dict(),
        "scenarios_completed": metrics["scenario"].astype(str).tolist(),
        "plots": plots,
        "known_limitations": [
            "One fixed development fold and a deliberately small balanced panel bound density-matrix runtime.",
            "Noise channels are local Markovian approximations and do not constitute a calibrated Aquila device model.",
            "The clean no-intercept head is frozen across scenarios; noisy readout retraining is intentionally excluded.",
            "Depolarizing noise is a generic comparability channel, whereas T1/T2 are physically motivated coherence channels.",
            "No hardware performance is inferred from the simulator noise curves.",
        ],
        "files": {
            "metrics": "noise_metrics.csv",
            "predictions": "predictions.csv.gz",
            "retained_samples": "retained_samples.csv",
            "frozen_readout": "frozen_readout.npz",
            "features": "features/",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
