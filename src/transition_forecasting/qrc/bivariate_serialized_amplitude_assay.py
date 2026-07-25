from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.bivariate_capacity_assay import (
    BivariateCapacityConfig,
    SyntheticTargets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import _evolve_segment_batch
from transition_forecasting.qrc.bivariate_crossover_assay import build_crossover_feature_banks
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    BivariateCrossoverStabilityConfig,
    generate_exchange_symmetric_windows,
    paired_permutation_orders,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    validate_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _fresh_states,
    _resolve_probe_steps,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)

PROGRAMS: dict[str, tuple[int, ...]] = {
    "single_u1": (0,),
    "single_u2": (1,),
    "serialized_u1_u2": (0, 1),
    "serialized_u2_u1": (1, 0),
}
REPRESENTATIONS = (
    "single_u1",
    "single_u2",
    "single_channels_concat",
    "serialized_forward",
    "serialized_reverse",
    "serialized_mirrors_concat",
)


@dataclass(frozen=True)
class BivariateSerializedAmplitudeConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_duration_us: float = 0.015
    interaction_scale: float = 1.25
    drive_phase_rad: float = 0.0
    permutations: int = 32

    def validate(self) -> None:
        if self.samples < 160 or self.samples % 2:
            raise ValueError("samples must be even and at least 160")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.memory_delays or any(delay < 1 for delay in self.memory_delays):
            raise ValueError("memory_delays must be positive")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the sequence")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.step_duration_us <= 0:
            raise ValueError("step_duration_us must be positive")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not np.isfinite(self.drive_phase_rad):
            raise ValueError("drive_phase_rad must be finite")
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        train_end = int(np.floor(self.samples * 0.60))
        validation_end = train_end + int(np.floor(self.samples * 0.20))
        if train_end % 2 or validation_end % 2:
            raise ValueError("split boundaries must preserve exchange pairs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def stability_config(self) -> BivariateCrossoverStabilityConfig:
        return BivariateCrossoverStabilityConfig(
            samples=self.samples,
            sequence_length=self.sequence_length,
            memory_delays=self.memory_delays,
            alphas=self.alphas,
            seeds=self.seeds,
            interaction_scale=self.interaction_scale,
            drive_phase_rad=self.drive_phase_rad,
            permutations=self.permutations,
        )


def _delayed(windows: np.ndarray, channel: int, delay: int) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    if delay < 0 or delay >= values.shape[1]:
        raise ValueError("delay lies outside the input window")
    return values[:, -(delay + 1), int(channel)]


def _legendre_two(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 0.5 * (3.0 * x * x - 1.0)


def build_serialized_capacity_targets(
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

    for channel in (0, 1):
        for delay in (1, 5):
            columns.append(_legendre_two(_delayed(values, channel, delay)))
            rows.append(
                {
                    "task": f"pointwise_p2_u{channel + 1}_d{delay}",
                    "group": "pointwise",
                    "channel": int(channel + 1),
                    "delay_a": int(delay),
                    "delay_b": int(delay),
                }
            )

    for channel in (0, 1):
        for delay_a, delay_b in ((1, 2), (2, 5)):
            columns.append(
                _delayed(values, channel, delay_a)
                * _delayed(values, channel, delay_b)
            )
            rows.append(
                {
                    "task": f"within_u{channel + 1}_d{delay_a}_d{delay_b}",
                    "group": "within_channel",
                    "channel": int(channel + 1),
                    "delay_a": int(delay_a),
                    "delay_b": int(delay_b),
                }
            )

    for name, delay_a, delay_b in (
        ("cross_same_d1", 1, 1),
        ("cross_u1d1_u2d2", 1, 2),
        ("cross_u1d2_u2d1", 2, 1),
        ("cross_u1d2_u2d5", 2, 5),
        ("cross_u1d5_u2d2", 5, 2),
    ):
        columns.append(
            _delayed(values, 0, delay_a) * _delayed(values, 1, delay_b)
        )
        rows.append(
            {
                "task": name,
                "group": "cross_channel",
                "channel": np.nan,
                "delay_a": int(delay_a),
                "delay_b": int(delay_b),
            }
        )

    columns.append(
        _delayed(values, 0, 1) * _delayed(values, 1, 2)
        - _delayed(values, 1, 1) * _delayed(values, 0, 2)
    )
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
        raise RuntimeError("serialized capacity target construction failed")
    return SyntheticTargets(values=target, metadata=metadata)


def _amplitude_drive(
    windows: np.ndarray,
    step: int,
    channel: int,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(windows, dtype=float)
    if channel not in (0, 1):
        raise ValueError("amplitude channel must be zero or one")
    omega = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * values[:, step, channel]
    )
    if np.any(omega <= 0):
        raise ValueError("serialized amplitude encoding became nonpositive")
    delta = np.full(len(values), reservoir.delta_center_rad_us, dtype=float)
    phase = np.full(len(values), float(phase_rad), dtype=float)
    return omega, delta, phase


def evolve_amplitude_program_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    channel_order: tuple[int, ...],
    *,
    program_name: str,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("serialized assay requires an exact six-atom reservoir")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if not channel_order or any(channel not in (0, 1) for channel in channel_order):
        raise ValueError("channel_order must contain only zero and one")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, steps, _ = values.shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []
    interactions = interaction_scale > 0
    segment_duration = reservoir.step_duration_us / len(channel_order)

    for step in range(steps):
        for channel in channel_order:
            omega, delta, phase = _amplitude_drive(
                values,
                step,
                int(channel),
                reservoir,
                drive_phase_rad,
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                segment_duration,
                reservoir,
                precomputed,
                interactions=interactions,
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1), n_atoms=precomputed.n_atoms
    )
    return stacked, {
        "encoding": "serialized_single_amplitude_channel",
        "program_name": str(program_name),
        "channel_order": [int(value) + 1 for value in channel_order],
        "substeps_per_financial_step": int(len(channel_order)),
        "segment_duration_us": float(segment_duration),
        "step_duration_us": float(reservoir.step_duration_us),
        "fixed_detuning_rad_us": float(reservoir.delta_center_rad_us),
        "interaction_scale": float(interaction_scale),
        "interactions_enabled": bool(interactions),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def build_serialized_representations(
    raw_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = set(PROGRAMS).difference(raw_features)
    if missing:
        raise ValueError(f"missing serialized feature banks: {sorted(missing)}")
    matrices = {name: np.asarray(raw_features[name], dtype=float) for name in PROGRAMS}
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise ValueError("serialized feature banks must be aligned")
    if next(iter(matrices.values())).shape[1] != 63:
        raise ValueError("each serialized source bank must contain 63 features")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("serialized feature banks must be finite")
    return {
        "single_u1": matrices["single_u1"],
        "single_u2": matrices["single_u2"],
        "single_channels_concat": np.concatenate(
            [matrices["single_u1"], matrices["single_u2"]], axis=1
        ),
        "serialized_forward": matrices["serialized_u1_u2"],
        "serialized_reverse": matrices["serialized_u2_u1"],
        "serialized_mirrors_concat": np.concatenate(
            [matrices["serialized_u1_u2"], matrices["serialized_u2_u1"]], axis=1
        ),
    }


def _capacity_sum(
    task_metrics: pd.DataFrame,
    group: str,
    *,
    channel: int | None = None,
    exclude_task: str | None = None,
) -> float:
    local = task_metrics.loc[task_metrics["group"].eq(group)]
    if channel is not None:
        local = local.loc[local["channel"].eq(channel)]
    if exclude_task is not None:
        local = local.loc[~local["task"].eq(exclude_task)]
    return float(local["capacity"].sum())


def _metric_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    payload: dict[str, float] = {}
    for channel in (1, 2):
        local = memory.loc[memory["channel"].eq(channel)]
        early = local.loc[local["delay_a"].le(5)]
        delay_five = local.loc[local["delay_a"].eq(5)]
        payload[f"channel{channel}_early"] = float(early["capacity"].sum())
        payload[f"channel{channel}_delay5"] = float(delay_five["capacity"].mean())
        payload[f"pointwise_u{channel}"] = _capacity_sum(
            task_metrics, "pointwise", channel=channel
        )
        payload[f"within_u{channel}"] = _capacity_sum(
            task_metrics, "within_channel", channel=channel
        )
    first = payload["channel1_early"]
    second = payload["channel2_early"]
    cross_same = float(
        task_metrics.loc[task_metrics["task"].eq("cross_same_d1"), "capacity"].sum()
    )
    cross_delayed = _capacity_sum(
        task_metrics,
        "cross_channel",
        exclude_task="cross_same_d1",
    )
    order = _capacity_sum(task_metrics, "order")
    nonlinear_total = float(
        payload["pointwise_u1"]
        + payload["pointwise_u2"]
        + payload["within_u1"]
        + payload["within_u2"]
        + cross_same
        + cross_delayed
    )
    payload.update(
        {
            "minimum_early": float(min(first, second)),
            "early_balance_ratio": float(min(first, second) / (max(first, second) + 1e-12)),
            "minimum_delay5": float(
                min(payload["channel1_delay5"], payload["channel2_delay5"])
            ),
            "minimum_within": float(min(payload["within_u1"], payload["within_u2"])),
            "cross_same": cross_same,
            "cross_delayed": cross_delayed,
            "order": order,
            "nonlinear_total": nonlinear_total,
        }
    )
    return payload


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "channel1_early",
        "channel2_early",
        "channel1_delay5",
        "channel2_delay5",
        "minimum_early",
        "minimum_delay5",
        "pointwise_u1",
        "pointwise_u2",
        "within_u1",
        "within_u2",
        "minimum_within",
        "cross_same",
        "cross_delayed",
        "order",
        "nonlinear_total",
    )
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & null_frame["representation"].eq(current["representation"])
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for serialized case")
        payload = current.to_dict()
        for metric in metrics:
            values = local[metric].to_numpy(dtype=float)
            observed_value = float(current[metric])
            payload[f"{metric}_null_q95"] = float(np.quantile(values, 0.95))
            payload[f"{metric}_permutation_p_ge"] = float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            )
            payload[f"{metric}_above_null_q95"] = bool(
                observed_value > payload[f"{metric}_null_q95"]
            )
        rows.append(payload)
    result = pd.DataFrame(rows)
    result["both_channels_early_above_null"] = (
        result["channel1_early_above_null_q95"]
        & result["channel2_early_above_null_q95"]
    )
    for metric in (
        "within_u1",
        "within_u2",
        "cross_same",
        "cross_delayed",
        "nonlinear_total",
    ):
        result[f"{metric}_margin"] = result[metric] - result[f"{metric}_null_q95"]
    return result


def _paired_on_off(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "channel1_delay5",
        "channel2_delay5",
        "within_u1",
        "within_u2",
        "cross_same",
        "cross_delayed",
        "nonlinear_total",
        "order",
    ]
    pivot = seed_summary.pivot_table(
        index=["seed", "representation"],
        columns="interaction",
        values=metrics,
    ).reset_index()
    pivot.columns = [
        "_".join(str(value) for value in column if str(value) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    for metric in metrics:
        pivot[f"{metric}_on_minus_off"] = pivot[f"{metric}_on"] - pivot[f"{metric}_off"]
    return pivot


def _aggregate(seed_summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    aggregate = seed_summary.groupby(
        ["interaction", "representation"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        feature_count=("feature_count", "first"),
        channel1_early_mean=("channel1_early", "mean"),
        channel2_early_mean=("channel2_early", "mean"),
        minimum_early_mean=("minimum_early", "mean"),
        early_balance_ratio_min=("early_balance_ratio", "min"),
        channel1_delay5_mean=("channel1_delay5", "mean"),
        channel2_delay5_mean=("channel2_delay5", "mean"),
        minimum_delay5_mean=("minimum_delay5", "mean"),
        pointwise_u1_mean=("pointwise_u1", "mean"),
        pointwise_u2_mean=("pointwise_u2", "mean"),
        within_u1_mean=("within_u1", "mean"),
        within_u2_mean=("within_u2", "mean"),
        minimum_within_mean=("minimum_within", "mean"),
        cross_same_mean=("cross_same", "mean"),
        cross_same_margin_mean=("cross_same_margin", "mean"),
        cross_same_margin_min=("cross_same_margin", "min"),
        cross_delayed_mean=("cross_delayed", "mean"),
        cross_delayed_margin_mean=("cross_delayed_margin", "mean"),
        cross_delayed_margin_min=("cross_delayed_margin", "min"),
        nonlinear_total_mean=("nonlinear_total", "mean"),
        nonlinear_total_margin_mean=("nonlinear_total_margin", "mean"),
        effective_rank_mean=("effective_rank_train_validation", "mean"),
        both_channels_early_above_null_fraction=(
            "both_channels_early_above_null",
            "mean",
        ),
        channel1_delay5_above_null_fraction=(
            "channel1_delay5_above_null_q95",
            "mean",
        ),
        channel2_delay5_above_null_fraction=(
            "channel2_delay5_above_null_q95",
            "mean",
        ),
        within_u1_above_null_fraction=("within_u1_above_null_q95", "mean"),
        within_u2_above_null_fraction=("within_u2_above_null_q95", "mean"),
        cross_same_above_null_fraction=("cross_same_above_null_q95", "mean"),
        cross_delayed_above_null_fraction=("cross_delayed_above_null_q95", "mean"),
    )
    paired_aggregate = paired.groupby("representation", as_index=False).agg(
        within_u1_on_minus_off_mean=("within_u1_on_minus_off", "mean"),
        within_u1_on_minus_off_min=("within_u1_on_minus_off", "min"),
        within_u2_on_minus_off_mean=("within_u2_on_minus_off", "mean"),
        within_u2_on_minus_off_min=("within_u2_on_minus_off", "min"),
        cross_delayed_on_minus_off_mean=("cross_delayed_on_minus_off", "mean"),
        cross_delayed_on_minus_off_min=("cross_delayed_on_minus_off", "min"),
        nonlinear_total_on_minus_off_mean=("nonlinear_total_on_minus_off", "mean"),
    )
    return aggregate.merge(
        paired_aggregate,
        on="representation",
        how="left",
        validate="many_to_one",
    )


def _equal_width_differences(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "minimum_early",
        "minimum_delay5",
        "within_u1",
        "within_u2",
        "cross_same",
        "cross_delayed",
        "nonlinear_total",
        "order",
    ]
    on = seed_summary.loc[seed_summary["interaction"].eq("on")]
    single = on.loc[
        on["representation"].eq("single_channels_concat"), ["seed", *metrics]
    ].copy()
    serialized = on.loc[
        on["representation"].eq("serialized_mirrors_concat"), ["seed", *metrics]
    ].copy()
    single = single.rename(columns={metric: f"{metric}_single" for metric in metrics})
    serialized = serialized.rename(
        columns={metric: f"{metric}_serialized" for metric in metrics}
    )
    merged = serialized.merge(single, on="seed", validate="one_to_one")
    for metric in metrics:
        merged[f"{metric}_serialized_minus_single"] = (
            merged[f"{metric}_serialized"] - merged[f"{metric}_single"]
        )
    return merged


def _render_plots(
    aggregate: pd.DataFrame,
    seed_summary: pd.DataFrame,
    differences: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    on = aggregate.loc[aggregate["interaction"].eq("on")].set_index("representation")
    order = list(REPRESENTATIONS)
    x = np.arange(len(order))
    width = 0.22

    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    axis.bar(x - width, on.loc[order, "minimum_within_mean"], width, label="within-channel")
    axis.bar(x, on.loc[order, "cross_same_mean"], width, label="cross same-lag")
    axis.bar(x + width, on.loc[order, "cross_delayed_mean"], width, label="cross delayed")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=22, ha="right")
    axis.set_ylabel("Mean capacity")
    axis.set_title("Single-channel and serialized nonlinear capacities")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "serialized_capacity_groups.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    local = seed_summary.loc[
        seed_summary["interaction"].eq("on")
        & seed_summary["representation"].isin(
            ["single_u1", "single_u2", "serialized_mirrors_concat"]
        )
    ]
    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    for representation, metric in (
        ("single_u1", "within_u1_margin"),
        ("single_u2", "within_u2_margin"),
        ("serialized_mirrors_concat", "cross_delayed_margin"),
    ):
        current = local.loc[local["representation"].eq(representation)].sort_values("seed")
        axis.plot(
            current["seed"].astype(str),
            current[metric],
            marker="o",
            label=representation,
        )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Capacity margin versus paired null")
    axis.set_title("Prerequisite and serialized temporal mixing by seed")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "serialized_temporal_margins_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    comparison = differences.sort_values("seed")
    x_seed = np.arange(len(comparison))
    figure, axis = plt.subplots(figsize=(9.8, 5.5))
    axis.bar(
        x_seed - 0.18,
        comparison["cross_delayed_serialized_minus_single"],
        0.36,
        label="delayed cross-channel",
    )
    axis.bar(
        x_seed + 0.18,
        comparison["nonlinear_total_serialized_minus_single"],
        0.36,
        label="total nonlinear",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_seed)
    axis.set_xticklabels(comparison["seed"].astype(str))
    axis.set_ylabel("Serialized mirrors minus separate single channels")
    axis.set_title("Equal-width serialization gain")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "serialized_equal_width_gain.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_serialized_amplitude_assay(
    *,
    results_root: Path,
    config: BivariateSerializedAmplitudeConfig = BivariateSerializedAmplitudeConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    config.validate()
    base_reservoir = reservoir or TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=config.step_duration_us,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seeds[0],
    )
    base_reservoir = replace(base_reservoir, step_duration_us=config.step_duration_us)
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    base_reservoir.validate()
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("serialized assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_interaction_control": True,
            "programs": {name: [channel + 1 for channel in order] for name, order in PROGRAMS.items()},
            "representations": list(REPRESENTATIONS),
            "scientific_question": (
                "Can the amplitude pathway perform off-diagonal temporal nonlinear processing, "
                "and does serializing two independent features through that same pathway produce "
                "robust delayed cross-channel products?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    stability = config.stability_config()
    task_frames: list[pd.DataFrame] = []
    observed_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        capacity_config = stability.capacity_config(int(seed))
        windows = generate_exchange_symmetric_windows(stability, seed=int(seed))
        targets = build_serialized_capacity_targets(windows, capacity_config)
        rng = np.random.default_rng(int(seed) + 2_718_281_828)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for interaction in ("off", "on"):
            scale = config.interaction_scale if interaction == "on" else 0.0
            raw_features: dict[str, np.ndarray] = {}
            for program_name, channel_order in PROGRAMS.items():
                probabilities, metadata = evolve_amplitude_program_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    channel_order,
                    program_name=program_name,
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
                raw_features[program_name] = build_crossover_feature_banks(probabilities)[
                    "occupation_pair_raw"
                ]
                np.savez_compressed(
                    probability_dir / f"seed_{seed}__{interaction}__{program_name}.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "program": program_name,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            representations = build_serialized_representations(raw_features)
            for representation, matrix in representations.items():
                metrics, diagnostics, candidates = fit_capacity_readout(
                    matrix,
                    targets,
                    capacity_config,
                )
                metrics.insert(0, "seed", int(seed))
                metrics.insert(1, "interaction", interaction)
                metrics.insert(2, "representation", representation)
                task_frames.append(metrics)
                candidates.to_csv(
                    selection_dir / f"seed_{seed}__{interaction}__{representation}.csv",
                    index=False,
                )
                observed_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "representation": representation,
                        "feature_count": int(matrix.shape[1]),
                        **diagnostics,
                        **_metric_payload(metrics),
                    }
                )

                for permutation, order in enumerate(null_orders):
                    null_targets = SyntheticTargets(
                        values=np.asarray(targets.values, dtype=float)[order],
                        metadata=targets.metadata.copy(),
                    )
                    null_metrics, _, _ = fit_capacity_readout(
                        matrix,
                        null_targets,
                        capacity_config,
                    )
                    null_rows.append(
                        {
                            "seed": int(seed),
                            "interaction": interaction,
                            "representation": representation,
                            "permutation": int(permutation),
                            **_metric_payload(null_metrics),
                        }
                    )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    null_frame = pd.DataFrame(null_rows)
    seed_summary = _annotate_nulls(observed, null_frame)
    paired = _paired_on_off(seed_summary)
    aggregate = _aggregate(seed_summary, paired)
    differences = _equal_width_differences(seed_summary)
    plots = _render_plots(aggregate, seed_summary, differences, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "serialized_seed_summary.csv", index=False)
    aggregate.to_csv(run_dir / "serialized_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
    differences.to_csv(run_dir / "serialized_vs_single_differences.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    on_seed = seed_summary.loc[seed_summary["interaction"].eq("on")]
    single_u1 = on_seed.loc[on_seed["representation"].eq("single_u1")]
    single_u2 = on_seed.loc[on_seed["representation"].eq("single_u2")]
    serialized = on_seed.loc[
        on_seed["representation"].eq("serialized_mirrors_concat")
    ]
    paired_serialized = paired.loc[
        paired["representation"].eq("serialized_mirrors_concat")
    ]

    single_u1_gate = bool(
        single_u1["channel1_delay5_above_null_q95"].all()
        and single_u1["within_u1_above_null_q95"].all()
    )
    single_u2_gate = bool(
        single_u2["channel2_delay5_above_null_q95"].all()
        and single_u2["within_u2_above_null_q95"].all()
    )
    serialized_gate = bool(
        serialized["both_channels_early_above_null"].all()
        and serialized["channel1_delay5_above_null_q95"].all()
        and serialized["channel2_delay5_above_null_q95"].all()
        and serialized["early_balance_ratio"].ge(0.80).all()
        and serialized["cross_delayed_above_null_q95"].all()
        and serialized["cross_delayed_margin"].gt(0.0).all()
    )
    serialized_interaction_gate = bool(
        serialized_gate
        and paired_serialized["cross_delayed_on_minus_off"].gt(0.0).all()
    )

    summary = {
        "status": "bivariate_serialized_amplitude_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "simulations_executed": int(len(config.seeds) * 2 * len(PROGRAMS)),
        "single_u1_temporal_gate_passed": single_u1_gate,
        "single_u2_temporal_gate_passed": single_u2_gate,
        "single_channel_prerequisite_passed": bool(single_u1_gate and single_u2_gate),
        "serialized_cross_temporal_gate_passed": serialized_gate,
        "serialized_interaction_assisted_gate_passed": serialized_interaction_gate,
        "serialized_cross_delayed_gain_over_single_all_seeds": bool(
            differences["cross_delayed_serialized_minus_single"].gt(0.0).all()
        ),
        "serialized_total_nonlinear_gain_over_single_all_seeds": bool(
            differences["nonlinear_total_serialized_minus_single"].gt(0.0).all()
        ),
        "decision_rule": (
            "First establish off-diagonal temporal nonlinear capacity for each scalar amplitude "
            "program while retaining delay-5 memory. Promote serialization only if the mirrored "
            "126-feature map retains balanced delay-5 memory and delayed cross-channel capacity "
            "exceeds paired nulls in every seed. Interaction assistance is reported separately."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
