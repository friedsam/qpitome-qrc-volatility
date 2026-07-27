from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
)
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_sequence_models import har_predictions, metrics
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
    fit_level_rate_scaler,
    transform_level_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_noise import (
    TemporalNoiseSpec,
    build_noisy_temporal_rydberg_features,
)


@dataclass(frozen=True)
class TemporalNoiseAssayConfig:
    fold: int = 5
    lead: int = 5
    sequence_length: int = 40
    max_per_class: int = 2
    seed: int = 20260722
    ridge_alpha: float = 100.0
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    amplitude_damping_t1_us: tuple[float, ...] = (200.0, 100.0, 50.0)
    dephasing_t2_us: tuple[float, ...] = (100.0, 50.0)
    depolarizing_probabilities: tuple[float, ...] = (0.005, 0.01, 0.03)

    def validate(self) -> None:
        if self.fold < 1 or self.lead < 1:
            raise ValueError("fold and lead must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if any(value <= 0 for value in self.amplitude_damping_t1_us):
            raise ValueError("all T1 values must be positive")
        if any(value <= 0 for value in self.dephasing_t2_us):
            raise ValueError("all T2 values must be positive")
        if any(not 0.0 < value < 1.0 for value in self.depolarizing_probabilities):
            raise ValueError("depolarizing probabilities must lie in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def noise_scenarios(config: TemporalNoiseAssayConfig) -> tuple[TemporalNoiseSpec, ...]:
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
    for scenario in scenarios:
        scenario.validate()
    return tuple(scenarios)


def _feature_distortion(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    clean = np.asarray(reference, dtype=float)
    noisy = np.asarray(candidate, dtype=float)
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
        "max_absolute_feature_change": float(np.max(np.abs(difference))),
    }


def _sign_accuracy(actual_residual: np.ndarray, predicted_correction: np.ndarray) -> float:
    actual = np.asarray(actual_residual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_correction, dtype=float).reshape(-1)
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
    validation: np.ndarray,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    y_val = y[validation]
    har_val = har[validation]
    prediction_val = prediction[validation]
    for sample_index, metadata in selected.iterrows():
        for horizon in range(y.shape[1]):
            rows.append(
                {
                    "scenario": scenario.name,
                    "sample_id": str(metadata["sample_id"]),
                    "fold": int(metadata["fold"]),
                    "lead": int(metadata["lead"]),
                    "label": int(metadata["label"]),
                    "episode_id": str(metadata["episode_id"]),
                    "origin_date": str(metadata["origin_date"]),
                    "horizon": int(horizon + 1),
                    "y_true": float(y_val[sample_index, horizon]),
                    "har_pred": float(har_val[sample_index, horizon]),
                    "y_pred": float(prediction_val[sample_index, horizon]),
                    "correction": float(
                        prediction_val[sample_index, horizon]
                        - har_val[sample_index, horizon]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _scenario_metric_row(
    *,
    scenario: TemporalNoiseSpec,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    validation: np.ndarray,
    clean_features: np.ndarray,
    features: np.ndarray,
    elapsed_seconds: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    qlike, rmse = metrics(y, prediction, validation)
    y_val = y[validation]
    har_val = har[validation]
    prediction_val = prediction[validation]
    correction = prediction_val - har_val
    actual_residual = y_val - har_val
    return {
        "scenario": scenario.name,
        "channel": (
            "ideal"
            if scenario.is_ideal
            else "amplitude_damping"
            if scenario.amplitude_damping_t1_us is not None
            else "dephasing"
            if scenario.dephasing_t2_us is not None
            else "depolarizing"
        ),
        "amplitude_damping_t1_us": scenario.amplitude_damping_t1_us,
        "dephasing_t2_us": scenario.dephasing_t2_us,
        "depolarizing_probability": scenario.depolarizing_probability,
        "val_qlike": float(qlike),
        "val_rmse": float(rmse),
        "residual_sign_accuracy": _sign_accuracy(actual_residual, correction),
        "mean_correction": float(np.mean(correction)),
        "mean_absolute_correction": float(np.mean(np.abs(correction))),
        "wall_seconds": float(elapsed_seconds),
        "seconds_per_sample": float(elapsed_seconds / len(features)),
        "max_trace_drift_before_renormalization": float(
            metadata["max_trace_drift_before_renormalization"]
        ),
        "max_hermiticity_error": float(metadata["max_hermiticity_error"]),
        **_feature_distortion(clean_features, features),
    }


def _render_plots(metrics_frame: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    local = metrics_frame.copy()
    ideal = local.loc[local["channel"].eq("ideal")].iloc[0]
    local["delta_qlike_vs_ideal"] = local["val_qlike"] - float(ideal["val_qlike"])
    local["delta_rmse_vs_ideal"] = local["val_rmse"] - float(ideal["val_rmse"])

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    positions = np.arange(len(local))
    axis.bar(positions - 0.18, local["delta_qlike_vs_ideal"], width=0.36, label="ΔQLIKE")
    axis.bar(positions + 0.18, local["delta_rmse_vs_ideal"], width=0.36, label="ΔRMSE")
    axis.axhline(0.0, linewidth=1.0)
    axis.set(
        title="Frozen-readout forecast degradation under noise",
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
        title="Reservoir-feature distortion under noise",
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
    axis.bar(positions, local["residual_sign_accuracy"])
    axis.axhline(0.5, linestyle="--", linewidth=1.0, label="Chance reference")
    axis.set(
        title="Residual-sign agreement of the frozen clean readout",
        xlabel="Noise scenario",
        ylabel="Validation sign accuracy",
        ylim=(0.0, 1.0),
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(local["scenario"], rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "residual_sign_accuracy_under_noise.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_temporal_rydberg_noise_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: TemporalNoiseAssayConfig = TemporalNoiseAssayConfig(),
    reservoir: TemporalRydbergChainConfig = TemporalRydbergChainConfig(),
    run_id: str | None = None,
) -> Path:
    """Run a fixed-readout density-matrix noise sensitivity study.

    The readout is fit once on ideal statevector features and then held fixed for
    every noisy scenario. The assay is a robustness/compliance study, not model
    selection, and it never evaluates test rows.
    """

    assay.validate()
    reservoir.validate()
    if reservoir.shots is not None:
        raise ValueError("noise assay requires shots=None")
    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=assay.level_channel_name,
        fallback=assay.fallback_level_channel,
    )
    selected = select_balanced_episode_rows(
        dataset.manifest,
        fold=int(assay.fold),
        lead=int(assay.lead),
        max_per_class=int(assay.max_per_class),
        splits=DEVELOPMENT_FOLD_SPLITS,
        seed=int(assay.seed),
    )
    if selected["fold_split"].eq("test").any():
        raise RuntimeError("noise assay must not receive test rows")
    tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
    level = extract_level_windows(
        dataset,
        tensor_rows,
        sequence_length=int(assay.sequence_length),
        level_channel=level_channel,
    )
    usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
    frame = selected.loc[usable].reset_index(drop=True)
    level = level[usable]
    tensor_rows = tensor_rows[usable]
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if not train.any() or not validation.any():
        raise RuntimeError("noise assay requires train and validation rows")

    scaler = fit_level_rate_scaler(level, train)
    windows = transform_level_windows(level, scaler)
    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = har_predictions(frame, y, train)

    clean_features, clean_metadata = build_temporal_rydberg_chain_features(
        windows,
        reservoir,
        condition="ordered",
    )
    feature_scaler = StandardScaler().fit(clean_features[train])
    clean_standardized = feature_scaler.transform(clean_features)
    readout = Ridge(alpha=float(assay.ridge_alpha))
    readout.fit(clean_standardized[train], (y - har)[train])

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "assay": assay.to_dict(),
            "reservoir": reservoir.to_dict(),
            "noise_scenarios": [scenario.to_dict() for scenario in noise_scenarios(assay)],
            "readout_policy": (
                "Ridge fit once on ideal statevector train features and frozen for all noise scenarios"
            ),
            "test_rows_allowed": False,
            "scientific_scope": (
                "Noise sensitivity and challenge compliance; not financial model selection"
            ),
        },
        run_id=run_id,
    )
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    parity: dict[str, float] = {}
    for scenario in noise_scenarios(assay):
        started = time.perf_counter()
        features, metadata = build_noisy_temporal_rydberg_features(
            windows,
            replace(reservoir, shots=None),
            scenario,
        )
        elapsed = float(time.perf_counter() - started)
        if scenario.is_ideal:
            parity = _feature_distortion(clean_features, features)
        standardized = feature_scaler.transform(features)
        correction = readout.predict(standardized)
        prediction = har + correction
        scenario_rows.append(
            _scenario_metric_row(
                scenario=scenario,
                y=y,
                har=har,
                prediction=prediction,
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
                validation=validation,
            )
        )
        np.savez_compressed(
            feature_dir / f"{scenario.name}.npz",
            features=features,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            train_mask=train,
            validation_mask=validation,
        )
        (feature_dir / f"{scenario.name}_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    metrics_frame = pd.DataFrame(scenario_rows)
    ideal_row = metrics_frame.loc[metrics_frame["channel"].eq("ideal")].iloc[0]
    metrics_frame["delta_qlike_vs_ideal"] = (
        metrics_frame["val_qlike"] - float(ideal_row["val_qlike"])
    )
    metrics_frame["delta_rmse_vs_ideal"] = (
        metrics_frame["val_rmse"] - float(ideal_row["val_rmse"])
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    plots = _render_plots(metrics_frame, run_dir / "plots")

    retained = frame[
        [
            "sample_id",
            "fold",
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
        ]
    ].copy()
    retained["tensor_row"] = tensor_rows
    retained.to_csv(run_dir / "retained_samples.csv", index=False)
    metrics_frame.to_csv(run_dir / "noise_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")

    summary = {
        "schema_version": 1,
        "status": "temporal_rydberg_noise_assay_complete",
        "test_rows_used": 0,
        "fold": int(assay.fold),
        "lead": int(assay.lead),
        "samples": int(len(frame)),
        "train_samples": int(train.sum()),
        "validation_samples": int(validation.sum()),
        "statevector_density_parity": parity,
        "clean_statevector_metadata": clean_metadata,
        "plots": plots,
        "known_limitations": [
            "One fixed development fold and a deliberately small balanced panel are used to control density-matrix runtime.",
            "The frozen Ridge readout quantifies output sensitivity; it is not promoted as the final financial model.",
            "Depolarizing noise is a comparability channel, whereas T1/T2 are physically motivated neutral-atom channels.",
            "No hardware result is inferred from the simulator noise curves.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
