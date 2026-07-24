from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
    probabilities_to_occupations,
    probabilities_to_symmetric_modes,
    validate_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _evolve_step_batch,
    _fresh_states,
    _resolve_probe_steps,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)

EncodingName = Literal["simultaneous", "sequential_noncommuting"]
FeatureBankName = Literal[
    "raw_input_linear",
    "six_mode_density_curvature",
    "nine_mode_symmetric",
    "all_occupations",
    "full_one_two_body",
]

QRC_FEATURE_BANKS: tuple[FeatureBankName, ...] = (
    "six_mode_density_curvature",
    "nine_mode_symmetric",
    "all_occupations",
    "full_one_two_body",
)
ENCODINGS: tuple[EncodingName, ...] = (
    "simultaneous",
    "sequential_noncommuting",
)


@dataclass(frozen=True)
class BivariateCapacityConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    interaction_scale: float = 1.25
    sequential_first_slot_fraction: float = 0.50
    sequential_second_phase_rad: float = float(np.pi / 2.0)
    seed: int = 20260724

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("samples must be at least 100")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.memory_delays or any(delay < 1 for delay in self.memory_delays):
            raise ValueError("memory_delays must be positive and nonempty")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the synthetic window")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive and nonempty")
        if not 0.4 <= self.train_fraction < 0.9:
            raise ValueError("train_fraction must lie in [0.4, 0.9)")
        if not 0.05 <= self.validation_fraction < 0.4:
            raise ValueError("validation_fraction must lie in [0.05, 0.4)")
        if self.train_fraction + self.validation_fraction >= 0.95:
            raise ValueError("train and validation fractions leave too little test data")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not 0.1 <= self.sequential_first_slot_fraction <= 0.9:
            raise ValueError("sequential_first_slot_fraction must lie in [0.1, 0.9]")
        if not np.isfinite(self.sequential_second_phase_rad):
            raise ValueError("sequential_second_phase_rad must be finite")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticTargets:
    values: np.ndarray
    metadata: pd.DataFrame


def generate_bivariate_windows(config: BivariateCapacityConfig) -> np.ndarray:
    config.validate()
    rng = np.random.default_rng(config.seed)
    return rng.uniform(
        -1.0,
        1.0,
        size=(config.samples, config.sequence_length, 2),
    ).astype(float)


def _delayed(windows: np.ndarray, channel: int, delay: int) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    if delay < 0 or delay >= values.shape[1]:
        raise ValueError("delay lies outside the input window")
    return values[:, -(delay + 1), int(channel)]


def build_capacity_targets(
    windows: np.ndarray,
    config: BivariateCapacityConfig,
) -> SyntheticTargets:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if values.shape[1] != config.sequence_length:
        raise ValueError("window length differs from capacity configuration")

    columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for channel in (0, 1):
        for delay in config.memory_delays:
            columns.append(_delayed(values, channel, int(delay)))
            rows.append(
                {
                    "task": f"memory_u{channel + 1}_d{int(delay)}",
                    "group": "memory",
                    "channel": int(channel + 1),
                    "delay_a": int(delay),
                    "delay_b": np.nan,
                }
            )

    mixing_specs = (
        ("mix_same_d1", 1, 1),
        ("mix_u1d1_u2d2", 1, 2),
        ("mix_u1d2_u2d1", 2, 1),
        ("mix_u1d2_u2d5", 2, 5),
        ("mix_u1d5_u2d2", 5, 2),
    )
    for name, delay_a, delay_b in mixing_specs:
        columns.append(
            _delayed(values, 0, delay_a) * _delayed(values, 1, delay_b)
        )
        rows.append(
            {
                "task": name,
                "group": "mixing",
                "channel": np.nan,
                "delay_a": int(delay_a),
                "delay_b": int(delay_b),
            }
        )

    order = (
        _delayed(values, 0, 1) * _delayed(values, 1, 2)
        - _delayed(values, 1, 1) * _delayed(values, 0, 2)
    )
    columns.append(order)
    rows.append(
        {
            "task": "antisymmetric_order_d1_d2",
            "group": "order",
            "channel": np.nan,
            "delay_a": 1,
            "delay_b": 2,
        }
    )

    target = np.column_stack(columns)
    metadata = pd.DataFrame(rows)
    if target.shape != (len(values), len(metadata)) or not np.isfinite(target).all():
        raise RuntimeError("capacity target construction failed")
    return SyntheticTargets(values=target, metadata=metadata)


def _apply_global_rxy_batch(
    states: np.ndarray,
    angles: np.ndarray,
    phases: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Apply a sample-specific global rotation around a phase-selected XY axis."""

    state = np.asarray(states, dtype=complex)
    angle = np.asarray(angles, dtype=float)
    phase = np.asarray(phases, dtype=float)
    if state.ndim != 2 or state.shape[1] != 2**n_atoms:
        raise ValueError("states have an incompatible shape")
    if angle.shape != (len(state),) or phase.shape != (len(state),):
        raise ValueError("angles and phases must align with state rows")

    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    phase_plus = np.exp(1j * phase)
    phase_minus = np.conjugate(phase_plus)
    result = state
    for qubit in range(n_atoms):
        left = 2**qubit
        right = 2 ** (n_atoms - 1 - qubit)
        view = result.reshape(-1, left, 2, right)
        state_zero = view[:, :, 0, :].copy()
        state_one = view[:, :, 1, :].copy()
        c = cosine[:, None, None]
        s = sine[:, None, None]
        plus = phase_plus[:, None, None]
        minus = phase_minus[:, None, None]
        view[:, :, 0, :] = c * state_zero - 1j * s * plus * state_one
        view[:, :, 1, :] = -1j * s * minus * state_zero + c * state_one
        result = view.reshape(-1, 2**n_atoms)
    return result


def _evolve_segment_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    phase: np.ndarray,
    duration_us: float,
    reservoir: TemporalRydbergChainConfig,
    precomputed: object,
    *,
    interactions: bool,
) -> np.ndarray:
    """Strang-split one arbitrary-duration global AHS segment."""

    if duration_us <= 0:
        raise ValueError("duration_us must be positive")
    omega_values = np.asarray(omega, dtype=float)
    delta_values = np.asarray(delta, dtype=float)
    phase_values = np.asarray(phase, dtype=float)
    samples = len(states)
    if any(array.shape != (samples,) for array in (omega_values, delta_values, phase_values)):
        raise ValueError("drive arrays must align with state rows")

    interaction_energy = (
        np.asarray(precomputed.interaction_energy, dtype=float)
        if interactions
        else np.zeros_like(precomputed.interaction_energy, dtype=float)
    )
    scale = max(
        float(np.max(np.abs(omega_values), initial=0.0)),
        float(np.max(np.abs(delta_values), initial=0.0)),
        float(np.max(np.abs(interaction_energy), initial=0.0)),
        1e-12,
    )
    substeps = int(
        np.clip(
            np.ceil(scale * float(duration_us) / reservoir.max_phase_per_substep),
            1,
            reservoir.max_substeps_per_step,
        )
    )
    dt = float(duration_us) / substeps
    diagonal = (
        interaction_energy[None, :]
        - delta_values[:, None] * precomputed.total_occupation[None, :]
    )
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    angles = omega_values * dt

    result = np.asarray(states, dtype=complex) * half
    for substep in range(substeps):
        result = _apply_global_rxy_batch(
            result,
            angles,
            phase_values,
            int(precomputed.n_atoms),
        )
        result *= half if substep == substeps - 1 else full
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise RuntimeError("sequential segment normalization failed")
    return result / norm


def evolve_sequential_probe_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    config: BivariateCapacityConfig,
    *,
    interaction_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Encode u1 and u2 in consecutive noncommuting global pulse slots.

    Slot A uses u1 in global detuning with an X-axis drive. Slot B uses u2 in
    global Rabi amplitude with a Y-axis drive. The two slot durations sum to the
    incumbent per-observation duration.
    """

    reservoir.validate()
    geometry.validate()
    config.validate()
    values = np.asarray(windows, dtype=float)
    if (
        values.ndim != 3
        or values.shape[2] != 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("sequential probability extraction requires six exact atoms")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, steps, _ = values.shape
    first_duration = (
        reservoir.step_duration_us * config.sequential_first_slot_fraction
    )
    second_duration = reservoir.step_duration_us - first_duration
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    interactions = interaction_scale > 0
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []

    for step in range(steps):
        u1 = values[:, step, 0]
        u2 = values[:, step, 1]
        states = _evolve_segment_batch(
            states,
            np.full(samples, reservoir.omega_base_rad_us, dtype=float),
            reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * u1,
            np.zeros(samples, dtype=float),
            first_duration,
            reservoir,
            precomputed,
            interactions=interactions,
        )
        omega_second = reservoir.omega_base_rad_us * (
            1.0 + reservoir.omega_mod_fraction * u2
        )
        if np.any(omega_second <= 0):
            raise ValueError("sequential amplitude encoding became nonpositive")
        states = _evolve_segment_batch(
            states,
            omega_second,
            np.full(samples, reservoir.delta_center_rad_us, dtype=float),
            np.full(samples, config.sequential_second_phase_rad, dtype=float),
            second_duration,
            reservoir,
            precomputed,
            interactions=interactions,
        )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1),
        n_atoms=precomputed.n_atoms,
    )
    metadata = {
        "encoding": "sequential_noncommuting",
        "probe_steps": [int(value) for value in probe_steps],
        "interaction_scale": float(interaction_scale),
        "first_slot_duration_us": float(first_duration),
        "second_slot_duration_us": float(second_duration),
        "second_slot_phase_rad": float(config.sequential_second_phase_rad),
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }
    return stacked, metadata


def probabilities_to_full_low_order(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return all six occupations, all 15 pairs, and all 15 connected pairs."""

    values = validate_probe_probabilities(probabilities)
    occupations = probabilities_to_occupations(values)
    pairs = tuple(itertools.combinations(range(6), 2))
    states = np.arange(64)
    bits = np.stack(
        [((states >> (5 - site)) & 1) for site in range(6)],
        axis=1,
    ).astype(float)
    pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in pairs],
        axis=1,
    )
    pair_expectation = np.einsum("rps,sk->rpk", values, pair_bits)
    connected = pair_expectation - np.stack(
        [
            occupations[:, :, left] * occupations[:, :, right]
            for left, right in pairs
        ],
        axis=2,
    )
    block = np.concatenate([occupations, pair_expectation, connected], axis=2)
    names: list[str] = []
    for probe in range(values.shape[1]):
        names.extend(f"probe_{probe}_occupation_site_{site}" for site in range(6))
        names.extend(
            f"probe_{probe}_pair_{left}_{right}" for left, right in pairs
        )
        names.extend(
            f"probe_{probe}_connected_{left}_{right}" for left, right in pairs
        )
    return block.reshape(len(values), -1), tuple(names)


def build_feature_banks(
    probabilities: np.ndarray,
) -> dict[FeatureBankName, np.ndarray]:
    values = validate_probe_probabilities(probabilities)
    symmetric = probabilities_to_symmetric_modes(values)
    six_indices = np.asarray(
        [index for probe in range(values.shape[1]) for index in (3 * probe, 3 * probe + 2)],
        dtype=int,
    )
    occupations = probabilities_to_occupations(values).reshape(len(values), -1)
    full, _ = probabilities_to_full_low_order(values)
    banks: dict[FeatureBankName, np.ndarray] = {
        "six_mode_density_curvature": symmetric[:, six_indices],
        "nine_mode_symmetric": symmetric,
        "all_occupations": occupations,
        "full_one_two_body": full,
    }
    if any(not np.isfinite(matrix).all() for matrix in banks.values()):
        raise RuntimeError("feature extraction produced non-finite values")
    return banks


def split_masks(config: BivariateCapacityConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(np.floor(config.samples * config.train_fraction))
    validation_end = train_end + int(
        np.floor(config.samples * config.validation_fraction)
    )
    indices = np.arange(config.samples)
    train = indices < train_end
    validation = (indices >= train_end) & (indices < validation_end)
    test = indices >= validation_end
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) < 20:
        raise ValueError("synthetic split is too small")
    return train, validation, test


def squared_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if np.std(observed) <= 1e-12 or np.std(predicted) <= 1e-12:
        return 0.0
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation * correlation, 0.0, 1.0))


def fit_capacity_readout(
    features: np.ndarray,
    targets: SyntheticTargets,
    config: BivariateCapacityConfig,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    matrix = np.asarray(features, dtype=float)
    y = np.asarray(targets.values, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(y) or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite matrix aligned with targets")
    train, validation, test = split_masks(config)
    candidates: list[dict[str, float]] = []
    for alpha in config.alphas:
        scaler = StandardScaler().fit(matrix[train])
        model = Ridge(alpha=float(alpha)).fit(scaler.transform(matrix[train]), y[train])
        prediction = model.predict(scaler.transform(matrix[validation]))
        corr2 = np.asarray(
            [
                squared_correlation(y[validation, task], prediction[:, task])
                for task in range(y.shape[1])
            ]
        )
        r2 = np.asarray(
            [
                r2_score(y[validation, task], prediction[:, task])
                for task in range(y.shape[1])
            ]
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_mean_corr2": float(np.mean(corr2)),
                "validation_mean_r2": float(np.mean(r2)),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            -row["validation_mean_corr2"],
            -row["validation_mean_r2"],
            row["alpha"],
        ),
    )

    fit = train | validation
    scaler = StandardScaler().fit(matrix[fit])
    model = Ridge(alpha=float(selected["alpha"])).fit(
        scaler.transform(matrix[fit]), y[fit]
    )
    prediction = model.predict(scaler.transform(matrix[test]))
    rows: list[dict[str, object]] = []
    for task_index, task in targets.metadata.iterrows():
        observed = y[test, task_index]
        estimated = prediction[:, task_index]
        rows.append(
            {
                **task.to_dict(),
                "test_samples": int(test.sum()),
                "corr2": squared_correlation(observed, estimated),
                "r2": float(r2_score(observed, estimated)),
                "rmse": float(np.sqrt(np.mean((observed - estimated) ** 2))),
            }
        )
    diagnostics = {
        **selected,
        "feature_count": float(matrix.shape[1]),
        "effective_rank_train_validation": float(effective_rank(matrix[fit])),
        "numerical_rank_train_validation": float(
            np.linalg.matrix_rank(matrix[fit] - matrix[fit].mean(axis=0, keepdims=True))
        ),
    }
    return pd.DataFrame(rows), diagnostics, pd.DataFrame(candidates)


def capacity_summary(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    mixing = task_metrics.loc[task_metrics["group"].eq("mixing")]
    order = task_metrics.loc[task_metrics["group"].eq("order")]
    delay_five = memory.loc[memory["delay_a"].eq(5)]
    return {
        "memory_capacity_u1": float(memory.loc[memory["channel"].eq(1), "corr2"].sum()),
        "memory_capacity_u2": float(memory.loc[memory["channel"].eq(2), "corr2"].sum()),
        "mixing_capacity": float(mixing["corr2"].sum()),
        "order_capacity": float(order["corr2"].sum()),
        "mean_memory_corr2": float(memory["corr2"].mean()),
        "mean_mixing_corr2": float(mixing["corr2"].mean()),
        "mean_order_corr2": float(order["corr2"].mean()),
        "mean_delay5_memory_corr2": float(delay_five["corr2"].mean()),
        "mean_task_r2": float(task_metrics["r2"].mean()),
    }


def _render_plots(
    task_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    interacting = task_metrics.loc[
        task_metrics["interaction"].eq("on")
        & task_metrics["feature_bank"].isin(
            ["six_mode_density_curvature", "all_occupations", "full_one_two_body"]
        )
        & task_metrics["group"].eq("memory")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True, sharey=True)
    for row, encoding in enumerate(ENCODINGS):
        for column, channel in enumerate((1, 2)):
            axis = axes[row, column]
            local = interacting.loc[
                interacting["encoding"].eq(encoding)
                & interacting["channel"].eq(channel)
            ]
            for bank, group in local.groupby("feature_bank", sort=True):
                axis.plot(group["delay_a"], group["corr2"], marker="o", label=bank)
            axis.set_title(f"{encoding}, channel {channel}")
            axis.set_xlabel("Delay")
            axis.set_ylabel("Squared correlation")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    figure.suptitle("Bivariate memory curves with interactions")
    figure.tight_layout()
    filename = "memory_curves.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    qrc = summary.loc[~summary["encoding"].eq("raw_input")].copy()
    labels = (
        qrc["encoding"]
        + " / "
        + qrc["interaction"]
        + " / "
        + qrc["feature_bank"]
    )
    figure, axis = plt.subplots(figsize=(11.5, max(6.0, 0.31 * len(qrc))))
    axis.barh(np.arange(len(qrc)), qrc["mixing_capacity"])
    axis.set_yticks(np.arange(len(qrc)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Sum of test squared correlations")
    axis.set_title("Cross-channel mixing capacity")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    filename = "mixing_capacity.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.5, max(6.0, 0.31 * len(qrc))))
    axis.barh(np.arange(len(qrc)), qrc["order_capacity"])
    axis.set_yticks(np.arange(len(qrc)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Test squared correlation")
    axis.set_title("Order-sensitive bivariate capacity")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    filename = "order_capacity.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    selected = qrc.loc[
        qrc["interaction"].eq("on")
        & qrc["feature_bank"].isin(
            ["six_mode_density_curvature", "full_one_two_body"]
        )
    ].copy()
    selected["case"] = selected["encoding"] + " / " + selected["feature_bank"]
    x = np.arange(len(selected))
    width = 0.26
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    axis.bar(x - width, selected["mean_delay5_memory_corr2"], width, label="delay-5 memory")
    axis.bar(x, selected["mean_mixing_corr2"], width, label="mean mixing")
    axis.bar(x + width, selected["mean_order_corr2"], width, label="order")
    axis.set_xticks(x)
    axis.set_xticklabels(selected["case"], rotation=25, ha="right")
    axis.set_ylabel("Squared correlation")
    axis.set_title("Compression and encoding diagnosis")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "compression_encoding_diagnosis.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for bank, group in qrc.groupby("feature_bank", sort=True):
        axis.scatter(group["effective_rank_train_validation"], group["mixing_capacity"], label=bank)
    axis.set_xlabel("Effective feature rank")
    axis.set_ylabel("Mixing capacity")
    axis.set_title("Does additional observable rank create usable nonlinear mixing?")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    filename = "rank_vs_mixing.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def _one_summary(
    summary: pd.DataFrame,
    *,
    encoding: str,
    interaction: str,
    feature_bank: str,
) -> pd.Series:
    rows = summary.loc[
        summary["encoding"].eq(encoding)
        & summary["interaction"].eq(interaction)
        & summary["feature_bank"].eq(feature_bank)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one capacity row for {encoding}/{interaction}/{feature_bank}"
        )
    return rows.iloc[0]


def run_bivariate_capacity_assay(
    *,
    results_root: Path,
    config: BivariateCapacityConfig = BivariateCapacityConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    config.validate()
    reservoir_config = reservoir or TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.03,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seed,
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    reservoir_config.validate()
    geometry_config.validate()
    if reservoir_config.n_atoms != 6 or reservoir_config.shots is not None:
        raise ValueError("capacity assay requires the exact six-atom ladder")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": reservoir_config.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "scientific_questions": [
                "Does the incumbent reservoir retain each independent channel?",
                "Does it form linearly decodable nonlinear cross-channel products?",
                "Does it retain order-sensitive bivariate information?",
                "Does six-mode compression discard capacity present in atom/pair observables?",
                "Do sequential noncommuting slots improve mixing over simultaneous encoding?",
                "Are interactions required for any observed mixing capacity?",
            ],
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    windows = generate_bivariate_windows(config)
    targets = build_capacity_targets(windows, config)
    np.savez_compressed(
        run_dir / "synthetic_inputs_targets.npz",
        windows=windows,
        targets=targets.values,
        task_names=targets.metadata["task"].astype(str).to_numpy(),
    )
    targets.metadata.to_csv(run_dir / "task_definitions.csv", index=False)

    task_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    raw_features = windows.reshape(len(windows), -1)
    raw_tasks, raw_diagnostics, raw_candidates = fit_capacity_readout(
        raw_features, targets, config
    )
    raw_tasks.insert(0, "encoding", "raw_input")
    raw_tasks.insert(1, "interaction", "none")
    raw_tasks.insert(2, "feature_bank", "raw_input_linear")
    task_frames.append(raw_tasks)
    summary_rows.append(
        {
            "encoding": "raw_input",
            "interaction": "none",
            "feature_bank": "raw_input_linear",
            **raw_diagnostics,
            **capacity_summary(raw_tasks),
        }
    )
    selection_rows.append(
        {
            "encoding": "raw_input",
            "interaction": "none",
            "feature_bank": "raw_input_linear",
            **raw_diagnostics,
        }
    )
    raw_candidates.to_csv(
        selection_dir / "raw_input__none__raw_input_linear.csv", index=False
    )

    simulation_metadata: dict[str, object] = {}
    for encoding in ENCODINGS:
        for interaction_name, interaction_scale in (
            ("on", config.interaction_scale),
            ("off", 0.0),
        ):
            if encoding == "simultaneous":
                probabilities, metadata = evolve_ladder_probe_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    interaction_scale=float(interaction_scale),
                    condition="ordered",
                )
                metadata = {**metadata, "encoding": encoding}
            else:
                probabilities, metadata = evolve_sequential_probe_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    config,
                    interaction_scale=float(interaction_scale),
                )
            case = f"{encoding}__interaction_{interaction_name}"
            simulation_metadata[case] = metadata
            np.savez_compressed(
                probability_dir / f"{case}.npz",
                probabilities=probabilities,
                windows=windows,
            )
            banks = build_feature_banks(probabilities)
            for bank_name in QRC_FEATURE_BANKS:
                matrix = banks[bank_name]
                task_table, diagnostics, candidates = fit_capacity_readout(
                    matrix, targets, config
                )
                task_table.insert(0, "encoding", encoding)
                task_table.insert(1, "interaction", interaction_name)
                task_table.insert(2, "feature_bank", bank_name)
                task_frames.append(task_table)
                summary_rows.append(
                    {
                        "encoding": encoding,
                        "interaction": interaction_name,
                        "feature_bank": bank_name,
                        **diagnostics,
                        **capacity_summary(task_table),
                    }
                )
                selection_rows.append(
                    {
                        "encoding": encoding,
                        "interaction": interaction_name,
                        "feature_bank": bank_name,
                        **diagnostics,
                    }
                )
                candidates.to_csv(
                    selection_dir / f"{case}__{bank_name}.csv", index=False
                )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    selections = pd.DataFrame(selection_rows)
    task_metrics.to_csv(run_dir / "task_metrics.csv", index=False)
    summary.to_csv(run_dir / "capacity_summary.csv", index=False)
    selections.to_csv(run_dir / "selected_readouts.csv", index=False)
    plots = _render_plots(task_metrics, summary, run_dir / "plots")

    simultaneous_six = _one_summary(
        summary,
        encoding="simultaneous",
        interaction="on",
        feature_bank="six_mode_density_curvature",
    )
    simultaneous_full = _one_summary(
        summary,
        encoding="simultaneous",
        interaction="on",
        feature_bank="full_one_two_body",
    )
    sequential_full = _one_summary(
        summary,
        encoding="sequential_noncommuting",
        interaction="on",
        feature_bank="full_one_two_body",
    )
    sequential_off = _one_summary(
        summary,
        encoding="sequential_noncommuting",
        interaction="off",
        feature_bank="full_one_two_body",
    )
    diagnostic = {
        "status": "bivariate_capacity_assay_complete",
        "financial_data_used": False,
        "financial_test_rows_used": 0,
        "compression_mixing_gain_full_minus_six": float(
            simultaneous_full["mixing_capacity"]
            - simultaneous_six["mixing_capacity"]
        ),
        "compression_order_gain_full_minus_six": float(
            simultaneous_full["order_capacity"]
            - simultaneous_six["order_capacity"]
        ),
        "encoding_mixing_gain_sequential_minus_simultaneous": float(
            sequential_full["mixing_capacity"]
            - simultaneous_full["mixing_capacity"]
        ),
        "encoding_order_gain_sequential_minus_simultaneous": float(
            sequential_full["order_capacity"]
            - simultaneous_full["order_capacity"]
        ),
        "interaction_mixing_gain_sequential_on_minus_off": float(
            sequential_full["mixing_capacity"]
            - sequential_off["mixing_capacity"]
        ),
        "simultaneous_six_delay5_memory": float(
            simultaneous_six["mean_delay5_memory_corr2"]
        ),
        "simultaneous_full_delay5_memory": float(
            simultaneous_full["mean_delay5_memory_corr2"]
        ),
        "sequential_full_delay5_memory": float(
            sequential_full["mean_delay5_memory_corr2"]
        ),
        "interpretation": {
            "compression": (
                "Positive full-minus-six gains indicate that the six-mode bank discards "
                "linearly decodable capacity present in the same quantum probabilities."
            ),
            "encoding": (
                "Positive sequential-minus-simultaneous gains indicate that explicit "
                "noncommuting channel slots improve temporal mixing."
            ),
            "interactions": (
                "Positive interacting-minus-off gains indicate that atom interactions, "
                "rather than encoding alone, create usable mixing capacity."
            ),
            "financial_failure": (
                "If delay-5 memory and nonlinear mixing both succeed while the financial "
                "model fails, the remaining bottleneck is the financial input/target. If "
                "capacity fails here, repair the encoding, timescale, or readout bank first."
            ),
        },
        "plots": plots,
        "simulation_metadata": simulation_metadata,
    }
    (run_dir / "summary.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return run_dir
