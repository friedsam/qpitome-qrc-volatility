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
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    _fresh_states,
    _resolve_probe_steps,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)

PROGRAMS = (
    "amplitude_only",
    "detuning_only",
    "simultaneous_area_matched",
    "ordered_dx",
    "ordered_xd",
    "palindrome_dxd",
    "palindrome_xdx",
)
REPRESENTATIONS = (
    *PROGRAMS,
    "single_routes_concat",
    "ordered_mirrors_concat",
    "palindrome_mirrors_concat",
)
TEMPORAL_PRODUCTS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 5),
    (4, 5),
    (5, 8),
)
PRODUCT_FIELDS = tuple(f"product_d{left}_d{right}" for left, right in TEMPORAL_PRODUCTS)


@dataclass(frozen=True)
class BivariateCommonModeTemporalMixingConfig:
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
        if self.samples < 160:
            raise ValueError("samples must be at least 160")
        if self.sequence_length < 10:
            raise ValueError("sequence_length must be at least ten")
        if not self.memory_delays or any(delay < 1 for delay in self.memory_delays):
            raise ValueError("memory_delays must be positive")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the sequence")
        if any(
            max(left, right) >= self.sequence_length - 1
            for left, right in TEMPORAL_PRODUCTS
        ):
            raise ValueError("temporal product delay lies outside the sequence")
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def capacity_config(self, seed: int) -> BivariateCapacityConfig:
        return BivariateCapacityConfig(
            samples=self.samples,
            sequence_length=self.sequence_length,
            memory_delays=self.memory_delays,
            alphas=self.alphas,
            interaction_scale=self.interaction_scale,
            seed=int(seed),
        )


def generate_common_mode_windows(
    config: BivariateCommonModeTemporalMixingConfig,
    *,
    seed: int,
) -> np.ndarray:
    config.validate()
    rng = np.random.default_rng(int(seed))
    scalar = rng.uniform(
        -1.0,
        1.0,
        size=(config.samples, config.sequence_length, 1),
    )
    return np.repeat(scalar, 2, axis=2).astype(float)


def _common_scalar(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if not np.allclose(values[:, :, 0], values[:, :, 1], atol=0.0, rtol=0.0):
        raise ValueError("common-mode assay requires exactly duplicated input channels")
    return values[:, :, 0]


def _delayed(scalar: np.ndarray, delay: int) -> np.ndarray:
    values = np.asarray(scalar, dtype=float)
    if delay < 0 or delay >= values.shape[1]:
        raise ValueError("delay lies outside the scalar window")
    return values[:, -(delay + 1)]


def _legendre_two(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 0.5 * (3.0 * x * x - 1.0)


def build_common_mode_capacity_targets(
    windows: np.ndarray,
    config: BivariateCapacityConfig,
    temporal_products: tuple[tuple[int, int], ...] = TEMPORAL_PRODUCTS,
) -> SyntheticTargets:
    scalar = _common_scalar(windows)
    if scalar.shape[1] != config.sequence_length:
        raise ValueError("window length differs from capacity configuration")

    columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for delay in config.memory_delays:
        columns.append(_delayed(scalar, int(delay)))
        rows.append(
            {
                "task": f"memory_d{int(delay)}",
                "group": "memory",
                "channel": 1,
                "delay_a": int(delay),
                "delay_b": np.nan,
            }
        )

    for delay in (1, 5):
        columns.append(_legendre_two(_delayed(scalar, delay)))
        rows.append(
            {
                "task": f"pointwise_p2_d{delay}",
                "group": "pointwise",
                "channel": 1,
                "delay_a": int(delay),
                "delay_b": int(delay),
            }
        )

    for left, right in temporal_products:
        first, second = sorted((int(left), int(right)))
        columns.append(_delayed(scalar, first) * _delayed(scalar, second))
        rows.append(
            {
                "task": f"product_d{first}_d{second}",
                "group": "temporal_product",
                "channel": 1,
                "delay_a": first,
                "delay_b": second,
                "lag_separation": second - first,
            }
        )

    target = np.column_stack(columns)
    metadata = pd.DataFrame(rows)
    if target.shape != (len(scalar), len(metadata)) or not np.isfinite(target).all():
        raise RuntimeError("common-mode target construction failed")
    return SyntheticTargets(values=target, metadata=metadata)


def _segment_payloads(
    windows: np.ndarray,
    step: int,
    program: str,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    scalar = _common_scalar(windows)[:, int(step)]
    samples = len(scalar)
    omega_full = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * scalar
    )
    delta_full = reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * scalar
    omega_carrier = np.full(samples, reservoir.omega_base_rad_us, dtype=float)
    delta_carrier = np.full(samples, reservoir.delta_center_rad_us, dtype=float)
    zero = np.zeros(samples, dtype=float)
    phase = np.full(samples, float(phase_rad), dtype=float)
    duration = float(reservoir.step_duration_us)

    if np.any(omega_full <= 0):
        raise ValueError("common-mode amplitude encoding became nonpositive")
    if program == "amplitude_only":
        return [(omega_full, delta_carrier, phase, duration)]
    if program == "detuning_only":
        return [(omega_carrier, delta_full, phase, duration)]
    if program == "simultaneous_area_matched":
        return [(0.5 * omega_full, 0.5 * delta_full, phase, duration)]
    if program == "ordered_dx":
        return [
            (zero, delta_full, phase, 0.5 * duration),
            (omega_full, zero, phase, 0.5 * duration),
        ]
    if program == "ordered_xd":
        return [
            (omega_full, zero, phase, 0.5 * duration),
            (zero, delta_full, phase, 0.5 * duration),
        ]
    if program == "palindrome_dxd":
        return [
            (zero, delta_full, phase, 0.25 * duration),
            (omega_full, zero, phase, 0.5 * duration),
            (zero, delta_full, phase, 0.25 * duration),
        ]
    if program == "palindrome_xdx":
        return [
            (omega_full, zero, phase, 0.25 * duration),
            (zero, delta_full, phase, 0.5 * duration),
            (omega_full, zero, phase, 0.25 * duration),
        ]
    raise ValueError(f"unsupported common-mode program: {program}")


def evolve_common_mode_program_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    program: str,
    *,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    _common_scalar(windows)
    reservoir.validate()
    geometry.validate()
    if program not in PROGRAMS:
        raise ValueError(f"unsupported common-mode program: {program}")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("common-mode assay requires an exact six-atom reservoir")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, steps, _ = np.asarray(windows).shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []
    segment_counts: list[int] = []
    interactions = interaction_scale > 0

    for step in range(steps):
        segments = _segment_payloads(
            windows,
            step,
            program,
            reservoir,
            drive_phase_rad,
        )
        segment_counts.append(len(segments))
        for omega, delta, phase, duration in segments:
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                duration,
                reservoir,
                precomputed,
                interactions=interactions,
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = np.stack(blocks, axis=1)
    return stacked, {
        "encoding": "duplicated_common_mode_temporal_mixing",
        "program": program,
        "segments_per_step": int(segment_counts[0]),
        "step_duration_us": float(reservoir.step_duration_us),
        "interaction_scale": float(interaction_scale),
        "interactions_enabled": bool(interactions),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def build_common_mode_representations(
    raw_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = set(PROGRAMS).difference(raw_features)
    if missing:
        raise ValueError(f"missing common-mode feature banks: {sorted(missing)}")
    matrices = {name: np.asarray(raw_features[name], dtype=float) for name in PROGRAMS}
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise ValueError("common-mode feature banks must be aligned")
    if next(iter(matrices.values())).shape[1] != 63:
        raise ValueError("each source bank must contain 63 features")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("common-mode feature banks must be finite")
    return {
        **matrices,
        "single_routes_concat": np.concatenate(
            [matrices["amplitude_only"], matrices["detuning_only"]], axis=1
        ),
        "ordered_mirrors_concat": np.concatenate(
            [matrices["ordered_dx"], matrices["ordered_xd"]], axis=1
        ),
        "palindrome_mirrors_concat": np.concatenate(
            [matrices["palindrome_dxd"], matrices["palindrome_xdx"]], axis=1
        ),
    }


def _metric_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    early = memory.loc[memory["delay_a"].le(5)]
    delay_five = memory.loc[memory["delay_a"].eq(5)]
    pointwise = task_metrics.loc[task_metrics["group"].eq("pointwise")]
    temporal = task_metrics.loc[task_metrics["group"].eq("temporal_product")]
    payload: dict[str, float] = {
        "memory_total": float(memory["capacity"].sum()),
        "early_memory": float(early["capacity"].sum()),
        "delay5": float(delay_five["capacity"].mean()),
        "pointwise_total": float(pointwise["capacity"].sum()),
        "temporal_total": float(temporal["capacity"].sum()),
    }
    for field in PRODUCT_FIELDS:
        payload[field] = float(
            task_metrics.loc[task_metrics["task"].eq(field), "capacity"].sum()
        )
    return payload


def _permutation_orders(
    samples: int,
    permutations: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    return tuple(rng.permutation(samples) for _ in range(permutations))


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "memory_total",
        "early_memory",
        "delay5",
        "pointwise_total",
        "temporal_total",
        *PRODUCT_FIELDS,
    )
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & null_frame["representation"].eq(current["representation"])
        ]
        if local.empty:
            raise RuntimeError("missing permutation rows for common-mode case")
        payload = current.to_dict()
        for metric in metrics:
            values = local[metric].to_numpy(dtype=float)
            observed_value = float(current[metric])
            q95 = float(np.quantile(values, 0.95))
            payload[f"{metric}_null_q95"] = q95
            payload[f"{metric}_margin"] = observed_value - q95
            payload[f"{metric}_above_null_q95"] = bool(observed_value > q95)
            payload[f"{metric}_permutation_p_ge"] = float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _paired_on_off(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["delay5", "pointwise_total", "temporal_total", *PRODUCT_FIELDS]
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
    aggregations: dict[str, tuple[str, str]] = {
        "seeds": ("seed", "nunique"),
        "feature_count": ("feature_count", "first"),
        "early_memory_mean": ("early_memory", "mean"),
        "delay5_mean": ("delay5", "mean"),
        "delay5_pass_fraction": ("delay5_above_null_q95", "mean"),
        "pointwise_total_mean": ("pointwise_total", "mean"),
        "pointwise_margin_mean": ("pointwise_total_margin", "mean"),
        "temporal_total_mean": ("temporal_total", "mean"),
        "temporal_margin_mean": ("temporal_total_margin", "mean"),
        "temporal_margin_min": ("temporal_total_margin", "min"),
        "temporal_pass_fraction": ("temporal_total_above_null_q95", "mean"),
        "effective_rank_mean": ("effective_rank_train_validation", "mean"),
    }
    for field in PRODUCT_FIELDS:
        aggregations[f"{field}_mean"] = (field, "mean")
        aggregations[f"{field}_margin_mean"] = (f"{field}_margin", "mean")
        aggregations[f"{field}_margin_min"] = (f"{field}_margin", "min")
        aggregations[f"{field}_pass_fraction"] = (
            f"{field}_above_null_q95",
            "mean",
        )
    aggregate = seed_summary.groupby(
        ["interaction", "representation"], as_index=False
    ).agg(**aggregations)

    paired_aggregations: dict[str, tuple[str, str]] = {
        "temporal_total_on_minus_off_mean": ("temporal_total_on_minus_off", "mean"),
        "temporal_total_on_minus_off_min": ("temporal_total_on_minus_off", "min"),
    }
    for field in PRODUCT_FIELDS:
        paired_aggregations[f"{field}_on_minus_off_mean"] = (
            f"{field}_on_minus_off",
            "mean",
        )
        paired_aggregations[f"{field}_on_minus_off_min"] = (
            f"{field}_on_minus_off",
            "min",
        )
    paired_aggregate = paired.groupby("representation", as_index=False).agg(
        **paired_aggregations
    )
    return aggregate.merge(
        paired_aggregate,
        on="representation",
        how="left",
        validate="many_to_one",
    )


def _switching_differences(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "early_memory",
        "delay5",
        "pointwise_total",
        "temporal_total",
        "effective_rank_train_validation",
        *PRODUCT_FIELDS,
    ]
    on = seed_summary.loc[seed_summary["interaction"].eq("on")]
    baseline = on.loc[
        on["representation"].eq("single_routes_concat"), ["seed", *metrics]
    ].copy()
    baseline = baseline.rename(
        columns={metric: f"{metric}_single_routes" for metric in metrics}
    )
    rows: list[pd.DataFrame] = []
    for representation in ("ordered_mirrors_concat", "palindrome_mirrors_concat"):
        current = on.loc[
            on["representation"].eq(representation), ["seed", *metrics]
        ].copy()
        current.insert(1, "representation", representation)
        current = current.rename(columns={metric: f"{metric}_switched" for metric in metrics})
        merged = current.merge(baseline, on="seed", validate="one_to_one")
        for metric in metrics:
            merged[f"{metric}_switched_minus_single"] = (
                merged[f"{metric}_switched"] - merged[f"{metric}_single_routes"]
            )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _task_summary(task_metrics: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    temporal = task_metrics.loc[task_metrics["group"].eq("temporal_product")].copy()
    null_long = null_frame.melt(
        id_vars=["seed", "interaction", "representation", "permutation"],
        value_vars=list(PRODUCT_FIELDS),
        var_name="task",
        value_name="null_capacity",
    )
    null_summary = null_long.groupby(
        ["seed", "interaction", "representation", "task"], as_index=False
    ).agg(null_q95=("null_capacity", lambda values: float(np.quantile(values, 0.95))))
    temporal = temporal.merge(
        null_summary,
        on=["seed", "interaction", "representation", "task"],
        validate="many_to_one",
    )
    temporal["margin"] = temporal["capacity"] - temporal["null_q95"]
    temporal["above_null_q95"] = temporal["margin"] > 0
    return temporal


def _render_plots(
    aggregate: pd.DataFrame,
    seed_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    switching: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    on = aggregate.loc[aggregate["interaction"].eq("on")].set_index("representation")
    order = list(REPRESENTATIONS)
    x = np.arange(len(order))
    width = 0.26

    figure, axis = plt.subplots(figsize=(13.0, 6.2))
    axis.bar(x - width, on.loc[order, "delay5_mean"], width, label="delay-5 memory")
    axis.bar(x, on.loc[order, "pointwise_total_mean"], width, label="pointwise P2")
    axis.bar(x + width, on.loc[order, "temporal_total_mean"], width, label="temporal products")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=25, ha="right")
    axis.set_ylabel("Mean normalized capacity")
    axis.set_title("Common-mode memory and nonlinear capacity")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "common_mode_capacity_groups.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    key = seed_summary.loc[
        seed_summary["interaction"].eq("on")
        & seed_summary["representation"].isin(
            [
                "single_routes_concat",
                "ordered_mirrors_concat",
                "palindrome_mirrors_concat",
            ]
        )
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for representation, current in key.groupby("representation", sort=False):
        current = current.sort_values("seed")
        axis.plot(
            current["seed"].astype(str),
            current["temporal_total_margin"],
            marker="o",
            label=representation,
        )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Temporal-product margin versus paired null")
    axis.set_title("Effect of operator switching on temporal mixing")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "common_mode_temporal_margins_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    tasks = task_summary.loc[
        task_summary["interaction"].eq("on")
        & task_summary["representation"].isin(
            ["ordered_mirrors_concat", "palindrome_mirrors_concat"]
        )
    ]
    product_order = list(PRODUCT_FIELDS)
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    for representation, current in tasks.groupby("representation", sort=False):
        means = current.groupby("task")["margin"].mean().reindex(product_order)
        axis.plot(product_order, means, marker="o", label=representation)
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Temporal product target")
    axis.set_ylabel("Mean capacity margin versus null")
    axis.set_title("Temporal reach of common-mode mixing")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "common_mode_temporal_reach.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    comparison = switching.groupby("representation", as_index=False)[
        "temporal_total_switched_minus_single"
    ].mean()
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.bar(comparison["representation"], comparison["temporal_total_switched_minus_single"])
    axis.axhline(0.0, linewidth=1)
    axis.set_ylabel("Mean temporal capacity gain over separate routes")
    axis.set_title("Does within-trajectory switching add temporal mixing?")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "common_mode_switching_gain.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_common_mode_temporal_mixing_assay(
    *,
    results_root: Path,
    config: BivariateCommonModeTemporalMixingConfig = BivariateCommonModeTemporalMixingConfig(),
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
        raise ValueError("common-mode assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "duplicated_common_mode_input": True,
            "paired_interaction_control": True,
            "programs": list(PROGRAMS),
            "representations": list(REPRESENTATIONS),
            "scientific_question": (
                "Can the unchanged two-operator Rydberg architecture produce robust temporal "
                "nonlinear memory when both nominal channels carry the same scalar history, "
                "and how do simultaneous, ordered, and palindromic switching affect it?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    task_frames: list[pd.DataFrame] = []
    observed_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        capacity_config = config.capacity_config(int(seed))
        windows = generate_common_mode_windows(config, seed=int(seed))
        targets = build_common_mode_capacity_targets(
            windows,
            capacity_config,
        )
        rng = np.random.default_rng(int(seed) + 3_141_592_653)
        null_orders = _permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for interaction in ("off", "on"):
            scale = config.interaction_scale if interaction == "on" else 0.0
            raw_features: dict[str, np.ndarray] = {}
            for program in PROGRAMS:
                probabilities, metadata = evolve_common_mode_program_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    program,
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
                raw_features[program] = build_crossover_feature_banks(probabilities)[
                    "occupation_pair_raw"
                ]
                np.savez_compressed(
                    probability_dir / f"seed_{seed}__{interaction}__{program}.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "program": program,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            representations = build_common_mode_representations(raw_features)
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
    switching = _switching_differences(seed_summary)
    task_summary = _task_summary(task_metrics, null_frame)
    plots = _render_plots(
        aggregate,
        seed_summary,
        task_summary,
        switching,
        run_dir / "plots",
    )

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "common_mode_seed_summary.csv", index=False)
    aggregate.to_csv(run_dir / "common_mode_summary.csv", index=False)
    task_summary.to_csv(run_dir / "common_mode_temporal_task_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
    switching.to_csv(run_dir / "switching_vs_separate_differences.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    on = seed_summary.loc[seed_summary["interaction"].eq("on")]
    passing: list[str] = []
    interaction_assisted: list[str] = []
    product_passes: dict[str, list[str]] = {}
    for representation in REPRESENTATIONS:
        current = on.loc[on["representation"].eq(representation)]
        passed_products = [
            field for field in PRODUCT_FIELDS if current[f"{field}_above_null_q95"].all()
        ]
        product_passes[representation] = passed_products
        if current["delay5_above_null_q95"].all() and passed_products:
            passing.append(representation)
            paired_current = paired.loc[paired["representation"].eq(representation)]
            if any(
                paired_current[f"{field}_on_minus_off"].gt(0.0).all()
                for field in passed_products
            ):
                interaction_assisted.append(representation)

    summary = {
        "status": "bivariate_common_mode_temporal_mixing_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "programs_completed": len(PROGRAMS),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "simulations_executed": int(len(config.seeds) * 2 * len(PROGRAMS)),
        "temporal_mixing_gate_passed": bool(passing),
        "passing_representations": passing,
        "interaction_assisted_temporal_gate_passed": bool(interaction_assisted),
        "interaction_assisted_representations": interaction_assisted,
        "product_tasks_passing_all_seeds": product_passes,
        "decision_rule": (
            "A common-mode representation demonstrates temporal nonlinear memory only if delay-5 "
            "memory and at least one predeclared off-diagonal scalar product exceed their full "
            "readout-selection permutation floors in every seed. Interaction assistance is "
            "reported separately. Equal-width separate-route, ordered-switching, and palindromic "
            "representations isolate whether within-trajectory switching helps or suppresses mixing."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
