"""Reservoir-size scaling study for the final A/B/A palindrome QRC.

Exact statevector performance is measured on a fixed development panel for a
bounded set of atom counts. Resource requirements are reported analytically for
all requested counts, including sizes beyond the exact-simulation boundary.
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
from transition_forecasting.qrc.bivariate_capacity_dynamics import _evolve_segment_batch
from transition_forecasting.qrc.bivariate_crossover_assay import _branch_drive
from transition_forecasting.qrc.ladder_mode_readout_tools import ladder_mode_weights
from transition_forecasting.qrc.palindrome_noise_assay import _sign_accuracy
from transition_forecasting.qrc.palindrome_real_task_relevance_assay import (
    _mean_qlike,
    _resolve_schedule,
    _rmse,
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
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _Precomputed,
    _fresh_states,
    _occupation_bits,
    _resolve_probe_steps,
    effective_rank,
    interaction_matrix,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    staggered_ladder_positions,
)


@dataclass(frozen=True)
class PalindromeScalingConfig:
    fold: int = 5
    lead: int = 5
    sequence_length: int = 40
    max_per_class: int = 12
    selection_seed: int = 20260726
    representation: str = "level_instability"
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    schedule_name: str = "crossover_Ahalf_B_Ahalf"
    feature_bank: str = "size_normalized_density_curvature"
    ridge_alpha: float = 100.0
    correction_lambda: float = 1.0
    prequential_blocks: int = 5
    exact_atom_counts: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12)
    resource_min_atoms: int = 5
    resource_max_atoms: int = 20
    runtime_fit_tail: int = 4

    def validate(self) -> None:
        if self.fold < 1 or self.lead < 1:
            raise ValueError("fold and lead must be positive")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.representation != "level_instability":
            raise ValueError("submission scaling assay requires level_instability")
        if self.feature_bank != "size_normalized_density_curvature":
            raise ValueError("scaling assay requires fixed density-curvature features")
        if self.ridge_alpha <= 0 or self.correction_lambda < 0:
            raise ValueError("invalid readout parameters")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not self.exact_atom_counts:
            raise ValueError("exact_atom_counts cannot be empty")
        if len(set(self.exact_atom_counts)) != len(self.exact_atom_counts):
            raise ValueError("exact_atom_counts must be unique")
        if any(value < 5 for value in self.exact_atom_counts):
            raise ValueError("exact_atom_counts must be at least five")
        if self.resource_min_atoms < 5 or self.resource_max_atoms < self.resource_min_atoms:
            raise ValueError("invalid resource atom range")
        if min(self.exact_atom_counts) < self.resource_min_atoms:
            raise ValueError("exact atom count lies below resource range")
        if max(self.exact_atom_counts) > self.resource_max_atoms:
            raise ValueError("exact atom count lies above resource range")
        if self.runtime_fit_tail < 2:
            raise ValueError("runtime_fit_tail must be at least two")
        _resolve_schedule(self.schedule_name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generalized_staggered_ladder_positions(
    n_atoms: int,
    geometry: StaggeredLadderGeometryConfig,
) -> np.ndarray:
    """Extend the incumbent two-row staggered ladder without changing N=6."""

    geometry.validate()
    n = int(n_atoms)
    if n < 5:
        raise ValueError("generalized ladder requires at least five atoms")
    if n == 6:
        return staggered_ladder_positions(geometry)

    top_count = (n + 1) // 2
    bottom_count = n // 2
    longitudinal = float(geometry.longitudinal_spacing_um)
    bottom_spacing = longitudinal * float(geometry.bottom_spacing_scale)
    stagger = longitudinal * float(geometry.stagger_fraction)
    half_row = 0.5 * float(geometry.row_spacing_um)

    top = np.column_stack(
        [np.arange(top_count, dtype=float) * longitudinal, np.full(top_count, half_row)]
    )
    bottom = np.column_stack(
        [
            stagger + np.arange(bottom_count, dtype=float) * bottom_spacing,
            np.full(bottom_count, -half_row),
        ]
    )
    positions = np.vstack([top, bottom])
    reference_fraction = float(geometry.defect_site) / 5.0
    defect_site = int(np.clip(round(reference_fraction * (n - 1)), 0, n - 1))
    positions[defect_site, 0] += float(geometry.defect_dx_um)
    positions[defect_site, 1] += float(geometry.defect_dy_um)
    positions -= positions.mean(axis=0, keepdims=True)

    distances = np.sqrt(
        np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    )
    np.fill_diagonal(distances, np.inf)
    if not np.isfinite(positions).all() or float(distances.min()) <= 0.0:
        raise ValueError("generalized ladder contains invalid or repeated positions")
    return positions


def _row_curvature(count: int) -> np.ndarray | None:
    if count < 3:
        return None
    coordinate = np.linspace(-1.0, 1.0, int(count))
    constant = np.ones(int(count), dtype=float)
    curvature = coordinate**2
    for basis in (constant, coordinate):
        denominator = float(np.dot(basis, basis))
        if denominator > 0:
            curvature -= float(np.dot(curvature, basis) / denominator) * basis
    norm = float(np.linalg.norm(curvature))
    if norm <= 1e-12:
        return None
    return curvature / norm


def size_normalized_mode_weights(n_atoms: int) -> np.ndarray:
    """Return fixed-width density and longitudinal-curvature observables."""

    n = int(n_atoms)
    if n < 5:
        raise ValueError("mode weights require at least five atoms")
    if n == 6:
        return np.asarray(ladder_mode_weights()[:, (0, 2)], dtype=float)

    constant = np.ones(n, dtype=float) / np.sqrt(float(n))
    top_count = (n + 1) // 2
    bottom_count = n // 2
    row_specs = ((0, top_count), (top_count, bottom_count))
    row_modes: list[np.ndarray] = []
    for offset, count in row_specs:
        local = _row_curvature(count)
        if local is None:
            continue
        mode = np.zeros(n, dtype=float)
        mode[offset : offset + count] = local
        row_modes.append(mode)
    if not row_modes:
        raise RuntimeError("no row supports a curvature observable")
    curvature = np.sum(row_modes, axis=0) / np.sqrt(float(len(row_modes)))
    curvature -= float(np.dot(curvature, constant)) * constant
    curvature /= np.linalg.norm(curvature)
    weights = np.column_stack([constant, curvature])
    if not np.allclose(weights.T @ weights, np.eye(2), atol=1e-12, rtol=0.0):
        raise RuntimeError("scaling mode weights are not orthonormal")
    return weights


def _scaling_precompute(
    n_atoms: int,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
) -> _Precomputed:
    positions = generalized_staggered_ladder_positions(n_atoms, geometry)
    coupling = interaction_matrix(
        positions,
        float(reservoir.c6_rad_um6_per_us) * float(interaction_scale),
    )
    bits = _occupation_bits(int(n_atoms))
    interaction_energy = np.einsum(
        "si,ij,sj->s",
        bits,
        np.triu(coupling, k=1),
        bits,
    )
    return _Precomputed(
        positions=positions,
        interaction_matrix=coupling,
        occupation_bits=bits,
        total_occupation=bits.sum(axis=1),
        interaction_energy=interaction_energy,
        nearest_pairs=tuple(),
        nearest_pair_bits=np.zeros((len(bits), 0), dtype=float),
        long_pairs=tuple(),
        long_pair_bits=np.zeros((len(bits), 0), dtype=float),
    )


def evolve_scaling_palindrome_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    schedule_name: str,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(windows, dtype=float)
    reservoir.validate()
    geometry.validate()
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms < 5 or reservoir.shots is not None:
        raise ValueError("scaling evolution requires at least five exact atoms")
    if interaction_scale <= 0:
        raise ValueError("interaction_scale must be positive")
    schedule = _resolve_schedule(schedule_name)
    precomputed = _scaling_precompute(
        reservoir.n_atoms,
        reservoir,
        geometry,
        interaction_scale,
    )
    states = _fresh_states(len(values), reservoir.n_atoms)
    probe_steps = _resolve_probe_steps(values.shape[1], reservoir.probe_fractions)
    blocks: list[np.ndarray] = []
    total_substeps = 0

    for step in range(values.shape[1]):
        for branch, fraction in schedule.segments:
            omega, delta, phase = _branch_drive(
                values,
                step,
                branch,
                reservoir,
                drive_phase_rad,
            )
            duration = float(reservoir.step_duration_us * fraction)
            scale = max(
                float(np.max(np.abs(omega), initial=0.0)),
                float(np.max(np.abs(delta), initial=0.0)),
                float(np.max(np.abs(precomputed.interaction_energy), initial=0.0)),
                1e-12,
            )
            total_substeps += int(
                np.clip(
                    np.ceil(scale * duration / reservoir.max_phase_per_substep),
                    1,
                    reservoir.max_substeps_per_step,
                )
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                duration,
                reservoir,
                precomputed,
                interactions=True,
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    probabilities = np.stack(blocks, axis=1)
    return probabilities, {
        "n_atoms": int(reservoir.n_atoms),
        "state_dimension": int(2**reservoir.n_atoms),
        "probe_steps": [int(value) for value in probe_steps],
        "total_substeps": int(total_substeps),
        "positions_um": precomputed.positions.tolist(),
        "interaction_matrix_rad_us": precomputed.interaction_matrix.tolist(),
        "reservoir": reservoir.to_dict(),
        "geometry": geometry.to_dict(),
        "schedule": schedule.name,
        "interaction_scale": float(interaction_scale),
    }


def probabilities_to_scaling_features(
    probabilities: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    expected_states = 2 ** int(n_atoms)
    if values.ndim != 3 or values.shape[2] != expected_states:
        raise ValueError("probabilities have an incompatible shape")
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError("probabilities are invalid")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    bits = _occupation_bits(int(n_atoms))
    occupations = np.einsum("rps,sa->rpa", values, bits)
    weights = size_normalized_mode_weights(int(n_atoms))
    modes = np.einsum("rpa,am->rpm", occupations, weights)
    features = modes.reshape(len(values), -1)
    expected_width = 2 * values.shape[1]
    if features.shape != (len(values), expected_width):
        raise RuntimeError("unexpected scaling feature width")
    return features


def resource_scaling_table(
    min_atoms: int,
    max_atoms: int,
    *,
    samples: int,
    exact_atom_counts: tuple[int, ...],
) -> pd.DataFrame:
    exact = set(int(value) for value in exact_atom_counts)
    rows = []
    for n_atoms in range(int(min_atoms), int(max_atoms) + 1):
        dimension = 2**n_atoms
        state_bytes = 16 * dimension
        rows.append(
            {
                "n_atoms": n_atoms,
                "state_dimension": dimension,
                "complex_state_bytes_per_sample": state_bytes,
                "complex_state_mib_per_sample": state_bytes / 2**20,
                "raw_batch_state_gib": state_bytes * int(samples) / 2**30,
                "probability_bytes_per_sample": 8 * dimension,
                "exact_simulated": n_atoms in exact,
            }
        )
    return pd.DataFrame(rows)


def _runtime_projection(
    resources: pd.DataFrame,
    exact_metrics: pd.DataFrame,
    *,
    samples: int,
    tail: int,
) -> pd.DataFrame:
    output = resources.copy()
    output["measured_wall_seconds"] = np.nan
    output["projected_wall_seconds"] = np.nan
    output["projection_kind"] = "resource_only"
    measured = exact_metrics.sort_values("n_atoms")
    for _, row in measured.iterrows():
        mask = output["n_atoms"].eq(int(row["n_atoms"]))
        output.loc[mask, "measured_wall_seconds"] = float(row["wall_seconds"])
        output.loc[mask, "projected_wall_seconds"] = float(row["wall_seconds"])
        output.loc[mask, "projection_kind"] = "measured_exact"

    fit = measured.tail(min(int(tail), len(measured)))
    if len(fit) >= 2 and np.all(fit["wall_seconds"].to_numpy() > 0):
        coefficient = np.polyfit(
            fit["n_atoms"].to_numpy(dtype=float),
            np.log2(fit["wall_seconds"].to_numpy(dtype=float)),
            deg=1,
        )
        boundary = int(measured["n_atoms"].max())
        mask = output["n_atoms"].gt(boundary)
        output.loc[mask, "projected_wall_seconds"] = 2.0 ** np.polyval(
            coefficient,
            output.loc[mask, "n_atoms"].to_numpy(dtype=float),
        )
        output.loc[mask, "projection_kind"] = "empirical_log2_extrapolation"
        output["runtime_fit_slope_log2_per_atom"] = float(coefficient[0])
        output["runtime_fit_intercept_log2_seconds"] = float(coefficient[1])
    output["projected_seconds_per_sample"] = (
        output["projected_wall_seconds"] / float(samples)
    )
    return output


def _render_plots(metrics: pd.DataFrame, resources: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(metrics["n_atoms"], metrics["validation_qlike"], marker="o", label="QRC")
    axis.axhline(float(metrics["har_validation_qlike"].iloc[0]), linestyle="--", label="HAR")
    axis.set(xlabel="Atoms", ylabel="Validation QLIKE", title="Exact palindrome performance scaling")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "validation_qlike_vs_atoms.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    measured = resources["measured_wall_seconds"].notna()
    axis.semilogy(
        resources["n_atoms"],
        resources["projected_wall_seconds"],
        linestyle="--",
        marker="o",
        label="Measured / extrapolated",
    )
    axis.scatter(
        resources.loc[measured, "n_atoms"],
        resources.loc[measured, "measured_wall_seconds"],
        label="Measured exact",
    )
    axis.set(xlabel="Atoms", ylabel="Wall seconds", title="Statevector runtime scaling")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "runtime_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.semilogy(
        resources["n_atoms"],
        resources["raw_batch_state_gib"],
        marker="o",
    )
    axis.set(
        xlabel="Atoms",
        ylabel="Raw batched state memory (GiB)",
        title="Analytical statevector memory scaling",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "statevector_memory_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_palindrome_scaling_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: PalindromeScalingConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir_template: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    drive_phase_rad: float,
    run_id: str,
) -> Path:
    assay.validate()
    candidate_features.validate()
    reservoir_template.validate()
    geometry.validate()
    if reservoir_template.shots is not None:
        raise ValueError("scaling assay requires exact statevectors")
    if not np.isclose(reservoir_template.step_duration_us, 0.02):
        raise ValueError("scaling assay requires 0.02-us palindrome steps")
    if tuple(reservoir_template.probe_fractions) != (0.25, 0.5, 1.0):
        raise ValueError("scaling assay requires three incumbent probe times")

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
        raise RuntimeError("scaling assay must not receive test rows")
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
    validate_har_contract(frame)

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    har = _fit_har(frame, y, train)
    residuals, residual_train = _prequential_har_residuals(
        frame,
        y,
        train,
        blocks=int(assay.prequential_blocks),
    )
    if residual_train.sum() < 4:
        raise RuntimeError("scaling assay has insufficient causal residual-fit rows")

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

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "assay": assay.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir_template": reservoir_template.to_dict(),
            "geometry": geometry.to_dict(),
            "interaction_scale": float(interaction_scale),
            "drive_phase_rad": float(drive_phase_rad),
            "readout_policy": (
                "For each exact atom count, StandardScaler and Ridge(alpha=100, "
                "fit_intercept=False) are fit on the same causal HAR-residual rows; "
                "feature width remains six."
            ),
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for n_atoms in assay.exact_atom_counts:
        reservoir = TemporalRydbergChainConfig(
            **{
                **reservoir_template.to_dict(),
                "n_atoms": int(n_atoms),
                "defect_edge": min(1, int(n_atoms) - 2),
            }
        )
        started = time.perf_counter()
        probabilities, metadata = evolve_scaling_palindrome_probabilities(
            windows,
            reservoir,
            geometry,
            schedule_name=assay.schedule_name,
            interaction_scale=float(interaction_scale),
            drive_phase_rad=float(drive_phase_rad),
        )
        elapsed = float(time.perf_counter() - started)
        features = probabilities_to_scaling_features(probabilities, int(n_atoms))
        if features.shape[1] != 6:
            raise RuntimeError("scaling readout width drifted from six")
        feature_scaler = StandardScaler().fit(features[residual_train])
        design = feature_scaler.transform(features)
        readout = Ridge(alpha=float(assay.ridge_alpha), fit_intercept=False)
        readout.fit(design[residual_train], residuals[residual_train])
        correction = float(assay.correction_lambda) * np.asarray(
            readout.predict(design), dtype=float
        )
        prediction = har + correction
        y_val = y[validation]
        har_val = har[validation]
        prediction_val = prediction[validation]
        correction_val = correction[validation]
        qrc_qlike = _mean_qlike(y_val, prediction_val)
        qrc_rmse = _rmse(y_val, prediction_val)
        har_qlike = _mean_qlike(y_val, har_val)
        har_rmse = _rmse(y_val, har_val)
        metric_rows.append(
            {
                "n_atoms": int(n_atoms),
                "state_dimension": int(2**int(n_atoms)),
                "feature_width": int(features.shape[1]),
                "validation_qlike": qrc_qlike,
                "validation_rmse": qrc_rmse,
                "har_validation_qlike": har_qlike,
                "har_validation_rmse": har_rmse,
                "qlike_delta_vs_har": qrc_qlike - har_qlike,
                "rmse_delta_vs_har": qrc_rmse - har_rmse,
                "residual_sign_accuracy": _sign_accuracy(y_val - har_val, correction_val),
                "effective_rank_residual_train": effective_rank(features[residual_train]),
                "feature_std": float(np.std(features)),
                "correction_rms": float(np.sqrt(np.mean(correction_val**2))),
                "ridge_coef_norm": float(np.linalg.norm(readout.coef_)),
                "wall_seconds": elapsed,
                "seconds_per_sample": elapsed / len(features),
                "total_substeps": int(metadata["total_substeps"]),
                "raw_batch_state_gib": 16 * (2**int(n_atoms)) * len(features) / 2**30,
            }
        )
        selected = frame.loc[validation].reset_index(drop=True)
        for row_index, row in selected.iterrows():
            for horizon in range(y.shape[1]):
                prediction_rows.append(
                    {
                        "n_atoms": int(n_atoms),
                        "sample_id": str(row["sample_id"]),
                        "horizon": int(horizon + 1),
                        "y_true": float(y_val[row_index, horizon]),
                        "har_prediction": float(har_val[row_index, horizon]),
                        "prediction": float(prediction_val[row_index, horizon]),
                        "correction": float(correction_val[row_index, horizon]),
                    }
                )
        np.savez_compressed(
            feature_dir / f"n_atoms_{int(n_atoms)}.npz",
            probabilities=probabilities,
            features=features,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            train_mask=train,
            validation_mask=validation,
            residual_train_mask=residual_train,
            feature_scaler_mean=feature_scaler.mean_,
            feature_scaler_scale=feature_scaler.scale_,
            ridge_coef=np.asarray(readout.coef_, dtype=float),
        )
        (feature_dir / f"n_atoms_{int(n_atoms)}_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    metrics = pd.DataFrame(metric_rows).sort_values("n_atoms").reset_index(drop=True)
    resources = resource_scaling_table(
        assay.resource_min_atoms,
        assay.resource_max_atoms,
        samples=len(frame),
        exact_atom_counts=assay.exact_atom_counts,
    )
    resources = _runtime_projection(
        resources,
        metrics,
        samples=len(frame),
        tail=assay.runtime_fit_tail,
    )
    plots = _render_plots(metrics, resources, run_dir / "plots")

    retained = frame[
        ["sample_id", "fold", "fold_split", "lead", "label", "episode_id", "origin_date"]
    ].copy()
    retained["tensor_row"] = tensor_rows
    retained.to_csv(run_dir / "retained_samples.csv", index=False)
    metrics.to_csv(run_dir / "scaling_metrics.csv", index=False)
    resources.to_csv(run_dir / "resource_scaling.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "predictions.csv.gz", index=False, compression="gzip"
    )

    best = metrics.sort_values(["validation_qlike", "validation_rmse", "n_atoms"]).iloc[0]
    six = metrics.loc[metrics["n_atoms"].eq(6)]
    summary = {
        "schema_version": 1,
        "status": "palindrome_scaling_assay_complete",
        "test_rows_used": 0,
        "fold": int(assay.fold),
        "lead": int(assay.lead),
        "samples": int(len(frame)),
        "train_samples": int(train.sum()),
        "validation_samples": int(validation.sum()),
        "residual_train_samples": int(residual_train.sum()),
        "exact_atom_counts": [int(value) for value in assay.exact_atom_counts],
        "resource_atom_range": [int(assay.resource_min_atoms), int(assay.resource_max_atoms)],
        "feature_width_fixed": 6,
        "best_exact": {
            "n_atoms": int(best["n_atoms"]),
            "validation_qlike": float(best["validation_qlike"]),
            "validation_rmse": float(best["validation_rmse"]),
            "qlike_delta_vs_har": float(best["qlike_delta_vs_har"]),
        },
        "six_atom_reference": (
            {
                "validation_qlike": float(six.iloc[0]["validation_qlike"]),
                "validation_rmse": float(six.iloc[0]["validation_rmse"]),
                "qlike_delta_vs_har": float(six.iloc[0]["qlike_delta_vs_har"]),
            }
            if not six.empty
            else None
        ),
        "plots": plots,
        "files": {
            "metrics": "scaling_metrics.csv",
            "resources": "resource_scaling.csv",
            "predictions": "predictions.csv.gz",
            "retained_samples": "retained_samples.csv",
            "features": "features/",
        },
        "known_limitations": [
            "Forecast metrics are measured only for the bounded exact-statevector atom counts.",
            "Rows above the exact boundary contain analytical memory and empirical runtime projections, not forecast-performance extrapolations.",
            "The generalized two-row ladder preserves the six-atom geometry exactly but necessarily adds sites at larger sizes.",
            "The density-curvature readout remains six-dimensional to avoid conflating reservoir size with classical readout width.",
            "One fixed development fold is used; no test rows are evaluated.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
