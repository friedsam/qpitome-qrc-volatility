"""Finite-shot sensitivity for the frozen six-atom A/B/A palindrome QRC.

The exact statevector probabilities, feature scaler, and no-intercept Ridge head
are generated once and frozen. Independent multinomial measurement replicates
then quantify whether finite shots preserve the transition-versus-control
correction gap used as the crisis-warning signal. Pooled QLIKE/RMSE are retained
as secondary diagnostics only.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
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
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    sample_probe_probabilities,
)
from transition_forecasting.qrc.palindrome_noise_assay import (
    _feature_distortion,
    _sign_accuracy,
)
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _mean_qlike,
    _resolve_schedule,
    _rmse,
    evolve_palindrome_probabilities,
    validate_har_contract,
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


@dataclass(frozen=True)
class PalindromeShotAssayConfig:
    fold: int = 5
    lead: int = 5
    sequence_length: int = 40
    max_per_class: int = 12
    selection_seed: int = 20260726
    representation: str = "level_instability"
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    schedule_name: str = "crossover_Ahalf_B_Ahalf"
    feature_bank: str = "six_mode_density_curvature"
    ridge_alpha: float = 100.0
    correction_lambda: float = 1.0
    prequential_blocks: int = 5
    shot_counts: tuple[int, ...] = (100, 250, 500, 1000, 2000, 5000)
    measurement_seeds: tuple[int, ...] = (
        20260722,
        20260723,
        20260724,
        20260725,
        20260726,
    )

    def validate(self) -> None:
        if self.fold < 1 or self.lead < 1:
            raise ValueError("fold and lead must be positive")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.representation != "level_instability":
            raise ValueError("submission shot assay requires level_instability")
        if self.feature_bank != "six_mode_density_curvature":
            raise ValueError("submission shot assay requires six_mode_density_curvature")
        if self.ridge_alpha <= 0 or self.correction_lambda < 0:
            raise ValueError("invalid readout parameters")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not self.shot_counts or any(int(value) < 1 for value in self.shot_counts):
            raise ValueError("shot_counts must be positive and nonempty")
        if len(set(self.shot_counts)) != len(self.shot_counts):
            raise ValueError("shot_counts must be unique")
        if not self.measurement_seeds:
            raise ValueError("measurement_seeds cannot be empty")
        if len(set(self.measurement_seeds)) != len(self.measurement_seeds):
            raise ValueError("measurement_seeds must be unique")
        _resolve_schedule(self.schedule_name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(candidate, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("correlation arrays must share one shape")
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _sign_agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(candidate, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("sign-agreement arrays must share one shape")
    informative = np.abs(left) > 1e-12
    if not informative.any():
        return float("nan")
    return float(np.mean(np.sign(left[informative]) == np.sign(right[informative])))


def transition_control_direction_rows(
    *,
    labels: np.ndarray,
    exact_correction: np.ndarray,
    candidate_correction: np.ndarray,
    shot_count: int,
    measurement_seed: int,
) -> list[dict[str, object]]:
    """Summarize transition-minus-control correction separation by path and horizon."""

    target = np.asarray(labels, dtype=int).reshape(-1)
    exact = np.asarray(exact_correction, dtype=float)
    candidate = np.asarray(candidate_correction, dtype=float)
    if exact.shape != candidate.shape or exact.ndim != 2:
        raise ValueError("corrections must share shape (samples, horizons)")
    if target.shape != (len(exact),):
        raise ValueError("labels must align with correction rows")
    if set(np.unique(target)).difference({0, 1}):
        raise ValueError("direction metrics require binary control/transition labels")
    control = target == 0
    transition = target == 1
    if not control.any() or not transition.any():
        raise ValueError("direction metrics require both control and transition rows")

    scopes: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("path_mean", exact.mean(axis=1), candidate.mean(axis=1))
    ]
    scopes.extend(
        (f"h{horizon + 1}", exact[:, horizon], candidate[:, horizon])
        for horizon in range(exact.shape[1])
    )

    rows: list[dict[str, object]] = []
    for scope, exact_values, candidate_values in scopes:
        exact_control = float(np.mean(exact_values[control]))
        exact_transition = float(np.mean(exact_values[transition]))
        candidate_control = float(np.mean(candidate_values[control]))
        candidate_transition = float(np.mean(candidate_values[transition]))
        exact_gap = exact_transition - exact_control
        candidate_gap = candidate_transition - candidate_control
        orientation = float(np.sign(exact_gap)) if abs(exact_gap) > 1e-12 else 0.0
        rows.append(
            {
                "shot_count": int(shot_count),
                "measurement_seed": int(measurement_seed),
                "scope": scope,
                "control_samples": int(control.sum()),
                "transition_samples": int(transition.sum()),
                "exact_control_mean_correction": exact_control,
                "exact_transition_mean_correction": exact_transition,
                "exact_transition_minus_control_gap": exact_gap,
                "control_mean_correction": candidate_control,
                "transition_mean_correction": candidate_transition,
                "transition_minus_control_gap": candidate_gap,
                "gap_absolute_error": abs(candidate_gap - exact_gap),
                "gap_relative_to_exact": (
                    candidate_gap / exact_gap if abs(exact_gap) > 1e-12 else np.nan
                ),
                "gap_direction_preserved": bool(
                    abs(exact_gap) > 1e-12 and np.sign(candidate_gap) == np.sign(exact_gap)
                ),
                "control_sign_agreement": _sign_agreement(
                    exact_values[control], candidate_values[control]
                ),
                "transition_sign_agreement": _sign_agreement(
                    exact_values[transition], candidate_values[transition]
                ),
                "all_sample_sign_agreement": _sign_agreement(
                    exact_values, candidate_values
                ),
                "oriented_transition_warning_rate": (
                    float(np.mean(orientation * candidate_values[transition] > 0.0))
                    if orientation != 0.0
                    else np.nan
                ),
                "oriented_control_false_alarm_rate": (
                    float(np.mean(orientation * candidate_values[control] > 0.0))
                    if orientation != 0.0
                    else np.nan
                ),
            }
        )
    return rows


def _shot_metric_row(
    *,
    shot_count: int,
    measurement_seed: int,
    y: np.ndarray,
    har: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
    exact_prediction: np.ndarray,
    exact_correction: np.ndarray,
    exact_features: np.ndarray,
    sampled_features: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, object]:
    y_val = y[validation]
    har_val = har[validation]
    prediction_val = prediction[validation]
    correction_val = correction[validation]
    exact_prediction_val = exact_prediction[validation]
    exact_correction_val = exact_correction[validation]
    feature_metrics = _feature_distortion(exact_features, sampled_features)
    return {
        "shot_count": int(shot_count),
        "measurement_seed": int(measurement_seed),
        "validation_qlike": _mean_qlike(y_val, prediction_val),
        "validation_rmse": _rmse(y_val, prediction_val),
        "qlike_delta_vs_exact": _mean_qlike(y_val, prediction_val)
        - _mean_qlike(y_val, exact_prediction_val),
        "rmse_delta_vs_exact": _rmse(y_val, prediction_val)
        - _rmse(y_val, exact_prediction_val),
        "residual_sign_accuracy": _sign_accuracy(y_val - har_val, correction_val),
        "correction_correlation_vs_exact": _correlation(
            exact_correction_val, correction_val
        ),
        "prediction_correlation_vs_exact": _correlation(
            exact_prediction_val, prediction_val
        ),
        "correction_mae_vs_exact": float(
            np.mean(np.abs(correction_val - exact_correction_val))
        ),
        "correction_rmse_vs_exact": float(
            np.sqrt(np.mean((correction_val - exact_correction_val) ** 2))
        ),
        "cell_sign_agreement_vs_exact": _sign_agreement(
            exact_correction_val, correction_val
        ),
        "sampling_seconds": float(elapsed_seconds),
        **feature_metrics,
    }


def _aggregate_shots(
    shot_metrics: pd.DataFrame,
    direction_metrics: pd.DataFrame,
) -> pd.DataFrame:
    path = direction_metrics.loc[direction_metrics["scope"].eq("path_mean")].copy()
    merged = shot_metrics.merge(
        path,
        on=["shot_count", "measurement_seed"],
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for shot_count, group in merged.groupby("shot_count", sort=True):
        rows.append(
            {
                "shot_count": int(shot_count),
                "seeds": int(group["measurement_seed"].nunique()),
                "all_seed_gap_direction_preserved": bool(
                    group["gap_direction_preserved"].all()
                ),
                "gap_direction_preservation_rate": float(
                    group["gap_direction_preserved"].mean()
                ),
                "mean_gap_relative_to_exact": float(
                    group["gap_relative_to_exact"].mean()
                ),
                "std_gap_relative_to_exact": float(
                    group["gap_relative_to_exact"].std(ddof=0)
                ),
                "max_gap_absolute_error": float(group["gap_absolute_error"].max()),
                "mean_transition_sign_agreement": float(
                    group["transition_sign_agreement"].mean()
                ),
                "minimum_transition_sign_agreement": float(
                    group["transition_sign_agreement"].min()
                ),
                "mean_control_sign_agreement": float(
                    group["control_sign_agreement"].mean()
                ),
                "minimum_control_sign_agreement": float(
                    group["control_sign_agreement"].min()
                ),
                "mean_correction_correlation_vs_exact": float(
                    group["correction_correlation_vs_exact"].mean()
                ),
                "minimum_correction_correlation_vs_exact": float(
                    group["correction_correlation_vs_exact"].min()
                ),
                "mean_relative_feature_mae": float(
                    group["relative_feature_mae"].mean()
                ),
                "maximum_relative_feature_mae": float(
                    group["relative_feature_mae"].max()
                ),
                "mean_qlike_delta_vs_exact": float(
                    group["qlike_delta_vs_exact"].mean()
                ),
                "maximum_absolute_qlike_delta_vs_exact": float(
                    group["qlike_delta_vs_exact"].abs().max()
                ),
                "mean_rmse_delta_vs_exact": float(
                    group["rmse_delta_vs_exact"].mean()
                ),
                "maximum_absolute_rmse_delta_vs_exact": float(
                    group["rmse_delta_vs_exact"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("shot_count").reset_index(drop=True)


def _render_plots(
    summary: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.semilogx(
        summary["shot_count"],
        summary["mean_gap_relative_to_exact"],
        marker="o",
        label="Mean across seeds",
    )
    axis.fill_between(
        summary["shot_count"],
        summary["mean_gap_relative_to_exact"]
        - summary["std_gap_relative_to_exact"],
        summary["mean_gap_relative_to_exact"]
        + summary["std_gap_relative_to_exact"],
        alpha=0.2,
    )
    axis.axhline(1.0, linestyle="--", label="Exact gap")
    axis.set(
        xlabel="Shots per sample and probe",
        ylabel="Transition-control correction gap / exact gap",
        title="Finite-shot preservation of the crisis-warning gap",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "warning_gap_preservation_vs_shots.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.semilogx(
        summary["shot_count"],
        summary["mean_correction_correlation_vs_exact"],
        marker="o",
    )
    axis.set(
        xlabel="Shots per sample and probe",
        ylabel="Correction correlation with exact statevector",
        title="Finite-shot correction stability",
        ylim=(0.0, 1.01),
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "correction_correlation_vs_shots.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    path = direction_metrics.loc[direction_metrics["scope"].eq("path_mean")]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    grouped = path.groupby("shot_count", sort=True)
    shots = np.asarray(sorted(path["shot_count"].unique()), dtype=float)
    mean_gap = grouped["transition_minus_control_gap"].mean().reindex(shots).to_numpy()
    std_gap = grouped["transition_minus_control_gap"].std(ddof=0).reindex(shots).to_numpy()
    exact_gap = float(path["exact_transition_minus_control_gap"].iloc[0])
    axis.semilogx(shots, mean_gap, marker="o", label="Finite-shot mean")
    axis.fill_between(shots, mean_gap - std_gap, mean_gap + std_gap, alpha=0.2)
    axis.axhline(exact_gap, linestyle="--", label="Exact-state reference")
    axis.set(
        xlabel="Shots per sample and probe",
        ylabel="Transition mean correction − control mean correction",
        title="Transition-specific directional correction",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "transition_control_gap_vs_shots.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_palindrome_shot_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: PalindromeShotAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
    run_id: str,
) -> Path:
    """Measure finite-shot preservation of the final transition-warning signal."""

    assay.validate()
    candidate_features.validate()
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("shot assay requires exact six-atom probabilities")
    if not np.isclose(reservoir.step_duration_us, 0.02):
        raise ValueError("shot assay requires 0.02-us palindrome steps")
    if tuple(reservoir.probe_fractions) != (0.25, 0.5, 1.0):
        raise ValueError("shot assay requires probes at 1/4, 1/2, and endpoint")
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
        raise RuntimeError("shot assay must not receive test rows")
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
        raise RuntimeError("shot assay selected no valid rows")
    validate_har_contract(frame)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if train.sum() < 4 or validation.sum() < 2:
        raise RuntimeError("shot assay has insufficient train/validation rows")
    har = _fit_har(frame, y, train)
    residuals, residual_train = _prequential_har_residuals(
        frame,
        y,
        train,
        blocks=int(assay.prequential_blocks),
    )
    if residual_train.sum() < 4:
        raise RuntimeError("shot assay has insufficient causal residual-fit rows")

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

    exact_started = time.perf_counter()
    exact_probabilities, exact_metadata = evolve_palindrome_probabilities(
        windows,
        reservoir,
        geometry,
        schedule,
        interaction_scale=float(interaction_scale),
        interactions=True,
        drive_phase_rad=float(drive_phase_rad),
    )
    exact_seconds = float(time.perf_counter() - exact_started)
    exact_features = np.asarray(
        build_crossover_feature_banks(exact_probabilities)[assay.feature_bank],
        dtype=float,
    )
    if exact_features.shape[1] != 6:
        raise RuntimeError("shot assay feature width drifted from six")
    feature_scaler = StandardScaler().fit(exact_features[residual_train])
    exact_design = feature_scaler.transform(exact_features)
    readout = Ridge(alpha=float(assay.ridge_alpha), fit_intercept=False)
    readout.fit(exact_design[residual_train], residuals[residual_train])
    exact_correction = float(assay.correction_lambda) * np.asarray(
        readout.predict(exact_design), dtype=float
    )
    exact_prediction = har + exact_correction

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
            "readout_policy": (
                "StandardScaler and Ridge(alpha=100, fit_intercept=False) fit once "
                "on exact-state six-mode causal HAR residuals and frozen for every "
                "shot count and measurement seed"
            ),
            "primary_endpoint": (
                "preservation of validation transition-minus-control correction gap"
            ),
            "secondary_endpoints": "pooled QLIKE/RMSE and feature distortion",
            "test_rows_allowed": False,
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
    np.savez_compressed(
        feature_dir / "exact_reference.npz",
        probabilities=exact_probabilities,
        features=exact_features,
        sample_id=frame["sample_id"].astype(str).to_numpy(),
        train_mask=train,
        validation_mask=validation,
    )

    labels_val = frame.loc[validation, "label"].to_numpy(dtype=int)
    exact_correction_val = exact_correction[validation]
    exact_direction = pd.DataFrame(
        transition_control_direction_rows(
            labels=labels_val,
            exact_correction=exact_correction_val,
            candidate_correction=exact_correction_val,
            shot_count=0,
            measurement_seed=0,
        )
    )

    shot_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    probe_steps = tuple(int(value) for value in exact_metadata["probe_steps"])
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    for shot_count in assay.shot_counts:
        for measurement_seed in assay.measurement_seeds:
            started = time.perf_counter()
            sampled_probabilities = sample_probe_probabilities(
                exact_probabilities,
                shots=int(shot_count),
                base_seed=int(measurement_seed),
                fold=int(assay.fold),
                sample_ids=sample_ids,
                probe_steps=probe_steps,
            )
            sampled_features = np.asarray(
                build_crossover_feature_banks(sampled_probabilities)[assay.feature_bank],
                dtype=float,
            )
            sampled_design = feature_scaler.transform(sampled_features)
            correction = float(assay.correction_lambda) * np.asarray(
                readout.predict(sampled_design), dtype=float
            )
            prediction = har + correction
            elapsed = float(time.perf_counter() - started)
            shot_rows.append(
                _shot_metric_row(
                    shot_count=int(shot_count),
                    measurement_seed=int(measurement_seed),
                    y=y,
                    har=har,
                    prediction=prediction,
                    correction=correction,
                    validation=validation,
                    exact_prediction=exact_prediction,
                    exact_correction=exact_correction,
                    exact_features=exact_features,
                    sampled_features=sampled_features,
                    elapsed_seconds=elapsed,
                )
            )
            direction_rows.extend(
                transition_control_direction_rows(
                    labels=labels_val,
                    exact_correction=exact_correction_val,
                    candidate_correction=correction[validation],
                    shot_count=int(shot_count),
                    measurement_seed=int(measurement_seed),
                )
            )
            selected = frame.loc[validation].reset_index(drop=True)
            for row_index, row in selected.iterrows():
                for horizon in range(y.shape[1]):
                    prediction_rows.append(
                        {
                            "shot_count": int(shot_count),
                            "measurement_seed": int(measurement_seed),
                            "sample_id": str(row["sample_id"]),
                            "label": int(row["label"]),
                            "horizon": int(horizon + 1),
                            "y_true": float(y[validation][row_index, horizon]),
                            "har_prediction": float(har[validation][row_index, horizon]),
                            "exact_prediction": float(
                                exact_prediction[validation][row_index, horizon]
                            ),
                            "prediction": float(prediction[validation][row_index, horizon]),
                            "exact_correction": float(
                                exact_correction[validation][row_index, horizon]
                            ),
                            "correction": float(correction[validation][row_index, horizon]),
                        }
                    )

    shot_metrics = pd.DataFrame(shot_rows).sort_values(
        ["shot_count", "measurement_seed"]
    ).reset_index(drop=True)
    direction_metrics = pd.DataFrame(direction_rows).sort_values(
        ["shot_count", "measurement_seed", "scope"]
    ).reset_index(drop=True)
    shot_summary = _aggregate_shots(shot_metrics, direction_metrics)
    plots = _render_plots(shot_summary, direction_metrics, run_dir / "plots")

    retained = frame[
        ["sample_id", "fold", "fold_split", "lead", "label", "episode_id", "origin_date"]
    ].copy()
    retained["tensor_row"] = tensor_rows
    retained.to_csv(run_dir / "retained_samples.csv", index=False)
    shot_metrics.to_csv(run_dir / "shot_metrics.csv", index=False)
    direction_metrics.to_csv(run_dir / "direction_metrics.csv", index=False)
    exact_direction.to_csv(run_dir / "exact_direction_reference.csv", index=False)
    shot_summary.to_csv(run_dir / "shot_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "predictions.csv.gz", index=False, compression="gzip"
    )

    exact_y = y[validation]
    exact_har = har[validation]
    exact_prediction_val = exact_prediction[validation]
    path_exact = exact_direction.loc[exact_direction["scope"].eq("path_mean")].iloc[0]
    qualifying = shot_summary.loc[shot_summary["all_seed_gap_direction_preserved"]]
    summary = {
        "schema_version": 1,
        "status": "palindrome_shot_assay_complete",
        "test_rows_used": 0,
        "fold": int(assay.fold),
        "lead": int(assay.lead),
        "samples": int(len(frame)),
        "train_samples": int(train.sum()),
        "validation_samples": int(validation.sum()),
        "residual_train_samples": int(residual_train.sum()),
        "shot_counts": [int(value) for value in assay.shot_counts],
        "measurement_seeds": [int(value) for value in assay.measurement_seeds],
        "replicates": int(len(shot_metrics)),
        "feature_width": int(exact_features.shape[1]),
        "exact_reference": {
            "validation_qlike": _mean_qlike(exact_y, exact_prediction_val),
            "validation_rmse": _rmse(exact_y, exact_prediction_val),
            "qlike_delta_vs_har": _mean_qlike(exact_y, exact_prediction_val)
            - _mean_qlike(exact_y, exact_har),
            "rmse_delta_vs_har": _rmse(exact_y, exact_prediction_val)
            - _rmse(exact_y, exact_har),
            "transition_mean_correction": float(
                path_exact["exact_transition_mean_correction"]
            ),
            "control_mean_correction": float(
                path_exact["exact_control_mean_correction"]
            ),
            "transition_minus_control_gap": float(
                path_exact["exact_transition_minus_control_gap"]
            ),
            "statevector_seconds": exact_seconds,
        },
        "minimum_shots_all_seed_gap_direction_preserved": (
            int(qualifying["shot_count"].min()) if not qualifying.empty else None
        ),
        "primary_endpoint": (
            "transition-minus-control mean correction gap on validation rows"
        ),
        "plots": plots,
        "files": {
            "shot_metrics": "shot_metrics.csv",
            "direction_metrics": "direction_metrics.csv",
            "exact_direction_reference": "exact_direction_reference.csv",
            "shot_summary": "shot_summary.csv",
            "predictions": "predictions.csv.gz",
            "retained_samples": "retained_samples.csv",
            "frozen_readout": "frozen_readout.npz",
            "features": "features/",
        },
        "known_limitations": [
            "The assay samples ideal computational-basis probabilities and does not model readout confusion or hardware drift.",
            "The exact-state feature scaler and readout are frozen; finite-shot retraining is intentionally excluded.",
            "The primary directional endpoint compares matched transition and control rows, not a within-episode pre/post causal estimate.",
            "One fixed development fold is used; no test rows are evaluated.",
            "Pooled QLIKE and RMSE are secondary because calm and transition offsets can cancel.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
