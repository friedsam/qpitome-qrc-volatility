from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.bivariate_capacity_assay import (
    BivariateCapacityConfig,
    SyntheticTargets,
    build_capacity_targets,
    capacity_summary,
    fit_capacity_readout,
    generate_bivariate_windows,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import (
    FeatureBankName,
    _evolve_segment_batch,
    build_feature_banks,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
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

ControlName = Literal["detuning", "amplitude"]
SlotOrder = tuple[ControlName, ControlName]


@dataclass(frozen=True)
class EncodingCase:
    name: str
    detuning_channel: int
    amplitude_channel: int
    simultaneous: bool
    slot_order: SlotOrder = ("detuning", "amplitude")

    def validate(self) -> None:
        if self.detuning_channel not in (0, 1) or self.amplitude_channel not in (0, 1):
            raise ValueError("encoding channels must be zero or one")
        if self.detuning_channel == self.amplitude_channel:
            raise ValueError("detuning and amplitude must receive different channels")
        if set(self.slot_order) != {"detuning", "amplitude"}:
            raise ValueError("slot_order must contain detuning and amplitude once")

    def route_for_channel(self, channel: int) -> str:
        if int(channel) == self.detuning_channel:
            return "detuning"
        if int(channel) == self.amplitude_channel:
            return "amplitude"
        raise ValueError("channel must be zero or one")

    def slot_for_channel(self, channel: int) -> str:
        if self.simultaneous:
            return "simultaneous"
        route = self.route_for_channel(channel)
        return "first" if self.slot_order[0] == route else "second"


ENCODING_CASES: tuple[EncodingCase, ...] = (
    EncodingCase(
        name="simultaneous_original",
        detuning_channel=0,
        amplitude_channel=1,
        simultaneous=True,
    ),
    EncodingCase(
        name="simultaneous_swapped",
        detuning_channel=1,
        amplitude_channel=0,
        simultaneous=True,
    ),
    EncodingCase(
        name="sequential_det_first_original",
        detuning_channel=0,
        amplitude_channel=1,
        simultaneous=False,
        slot_order=("detuning", "amplitude"),
    ),
    EncodingCase(
        name="sequential_det_first_swapped",
        detuning_channel=1,
        amplitude_channel=0,
        simultaneous=False,
        slot_order=("detuning", "amplitude"),
    ),
    EncodingCase(
        name="sequential_amp_first_original",
        detuning_channel=0,
        amplitude_channel=1,
        simultaneous=False,
        slot_order=("amplitude", "detuning"),
    ),
    EncodingCase(
        name="sequential_amp_first_swapped",
        detuning_channel=1,
        amplitude_channel=0,
        simultaneous=False,
        slot_order=("amplitude", "detuning"),
    ),
)


@dataclass(frozen=True)
class BivariateEncodingSymmetryConfig:
    samples: int = 320
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    interaction_scale: float = 1.25
    permutations: int = 32
    feature_banks: tuple[FeatureBankName, ...] = (
        "six_mode_density_curvature",
        "full_one_two_body",
    )

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("samples must be at least 100")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.memory_delays or any(value < 1 for value in self.memory_delays):
            raise ValueError("memory_delays must be positive")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the sequence")
        if not self.alphas or any(value <= 0 for value in self.alphas):
            raise ValueError("alphas must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if self.permutations < 0:
            raise ValueError("permutations cannot be negative")
        supported = {"six_mode_density_curvature", "full_one_two_body"}
        if not self.feature_banks or set(self.feature_banks).difference(supported):
            raise ValueError("unsupported feature bank in symmetry assay")
        for case in ENCODING_CASES:
            case.validate()

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


def _drive_payload(
    windows: np.ndarray,
    step: int,
    control: ControlName,
    case: EncodingCase,
    reservoir: TemporalRydbergChainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = len(windows)
    if control == "detuning":
        values = windows[:, step, case.detuning_channel]
        return (
            np.full(samples, reservoir.omega_base_rad_us, dtype=float),
            reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * values,
            np.zeros(samples, dtype=float),
        )
    values = windows[:, step, case.amplitude_channel]
    omega = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * values
    )
    if np.any(omega <= 0):
        raise ValueError("amplitude encoding became nonpositive")
    return (
        omega,
        np.full(samples, reservoir.delta_center_rad_us, dtype=float),
        np.full(samples, np.pi / 2.0, dtype=float),
    )


def evolve_encoding_case_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    case: EncodingCase,
    *,
    interaction_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve one paired encoding case while preserving total time per observation."""

    case.validate()
    reservoir.validate()
    geometry.validate()
    values = np.asarray(windows, dtype=float)
    if (
        values.ndim != 3
        or values.shape[2] != 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("symmetry assay requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("symmetry assay requires positive interactions")

    if case.simultaneous:
        ordered = values[:, :, [case.detuning_channel, case.amplitude_channel]]
        probabilities, metadata = evolve_ladder_probe_probabilities(
            ordered,
            reservoir,
            geometry,
            interaction_scale=float(interaction_scale),
            condition="ordered",
        )
        return probabilities, {
            **metadata,
            "encoding_case": case.name,
            "detuning_channel": int(case.detuning_channel + 1),
            "amplitude_channel": int(case.amplitude_channel + 1),
            "slot_order": "simultaneous",
        }

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    samples, steps, _ = values.shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    slot_duration = 0.5 * reservoir.step_duration_us
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []

    for step in range(steps):
        for control in case.slot_order:
            omega, delta, phase = _drive_payload(
                values, step, control, case, reservoir
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                slot_duration,
                reservoir,
                precomputed,
                interactions=True,
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1), n_atoms=precomputed.n_atoms
    )
    return stacked, {
        "encoding_case": case.name,
        "detuning_channel": int(case.detuning_channel + 1),
        "amplitude_channel": int(case.amplitude_channel + 1),
        "slot_order": list(case.slot_order),
        "slot_duration_us": float(slot_duration),
        "probe_steps": [int(value) for value in probe_steps],
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
        "interaction_scale": float(interaction_scale),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def _channel_memory_rows(
    task_metrics: pd.DataFrame,
    case: EncodingCase,
    *,
    seed: int,
    feature_bank: str,
) -> list[dict[str, object]]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    rows: list[dict[str, object]] = []
    for zero_channel in (0, 1):
        channel = zero_channel + 1
        local = memory.loc[memory["channel"].eq(channel)]
        early = local.loc[local["delay_a"].le(5)]
        delay_five = local.loc[local["delay_a"].eq(5)]
        rows.append(
            {
                "seed": int(seed),
                "encoding_case": case.name,
                "feature_bank": str(feature_bank),
                "channel": int(channel),
                "control_route": case.route_for_channel(zero_channel),
                "slot": case.slot_for_channel(zero_channel),
                "memory_total": float(local["capacity"].sum()),
                "memory_early_d1_d5": float(early["capacity"].sum()),
                "memory_delay5": (
                    float(delay_five["capacity"].mean())
                    if not delay_five.empty
                    else float("nan")
                ),
                "memory_mean": float(local["capacity"].mean()),
            }
        )
    return rows


def _null_summary(
    matrix: np.ndarray,
    targets: SyntheticTargets,
    capacity_config: BivariateCapacityConfig,
    permutations: tuple[np.ndarray, ...],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, order in enumerate(permutations):
        null_targets = SyntheticTargets(
            values=np.asarray(targets.values, dtype=float)[order],
            metadata=targets.metadata.copy(),
        )
        task_metrics, _, _ = fit_capacity_readout(
            matrix, null_targets, capacity_config
        )
        payload = capacity_summary(task_metrics)
        rows.append(
            {
                "permutation": int(index),
                "memory_total": float(
                    payload["memory_capacity_u1"] + payload["memory_capacity_u2"]
                ),
                "delay5_mean": float(payload["mean_delay5_memory_capacity"]),
                "mixing_sum": float(payload["mixing_capacity"]),
                "order": float(payload["order_capacity"]),
            }
        )
    return rows


def _aggregate_observed(
    case_summary: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> pd.DataFrame:
    case_aggregate = case_summary.groupby(
        ["encoding_case", "feature_bank"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        mixing_mean=("mixing_capacity", "mean"),
        mixing_std=("mixing_capacity", "std"),
        order_mean=("order_capacity", "mean"),
        order_std=("order_capacity", "std"),
        effective_rank_mean=("effective_rank_train_validation", "mean"),
    )
    channel_pivot = channel_summary.pivot_table(
        index=["seed", "encoding_case", "feature_bank"],
        columns="channel",
        values=["memory_total", "memory_early_d1_d5", "memory_delay5"],
    ).reset_index()
    channel_pivot.columns = [
        "_".join(str(value) for value in column if str(value) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in channel_pivot.columns
    ]
    for metric in ("memory_total", "memory_early_d1_d5", "memory_delay5"):
        left = channel_pivot[f"{metric}_1"]
        right = channel_pivot[f"{metric}_2"]
        channel_pivot[f"{metric}_minimum"] = np.minimum(left, right)
        channel_pivot[f"{metric}_imbalance"] = (
            np.abs(left - right) / (np.abs(left) + np.abs(right) + 1e-12)
        )
    channel_aggregate = channel_pivot.groupby(
        ["encoding_case", "feature_bank"], as_index=False
    ).agg(
        channel1_memory_total=("memory_total_1", "mean"),
        channel2_memory_total=("memory_total_2", "mean"),
        minimum_total_memory=("memory_total_minimum", "mean"),
        total_memory_imbalance=("memory_total_imbalance", "mean"),
        channel1_early_memory=("memory_early_d1_d5_1", "mean"),
        channel2_early_memory=("memory_early_d1_d5_2", "mean"),
        minimum_early_memory=("memory_early_d1_d5_minimum", "mean"),
        early_memory_imbalance=("memory_early_d1_d5_imbalance", "mean"),
        channel1_delay5=("memory_delay5_1", "mean"),
        channel2_delay5=("memory_delay5_2", "mean"),
        minimum_delay5=("memory_delay5_minimum", "mean"),
        delay5_imbalance=("memory_delay5_imbalance", "mean"),
    )
    return case_aggregate.merge(
        channel_aggregate,
        on=["encoding_case", "feature_bank"],
        how="inner",
        validate="one_to_one",
    )


def _aggregate_null(
    null_frame: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    if null_frame.empty:
        return observed.copy()
    averaged = null_frame.groupby(
        ["encoding_case", "feature_bank", "permutation"], as_index=False
    )[["memory_total", "delay5_mean", "mixing_sum", "order"]].mean()
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = averaged.loc[
            averaged["encoding_case"].eq(current["encoding_case"])
            & averaged["feature_bank"].eq(current["feature_bank"])
        ]
        payload = current.to_dict()
        for observed_column, null_column, prefix in (
            ("mixing_mean", "mixing_sum", "mixing"),
            ("order_mean", "order", "order"),
        ):
            values = local[null_column].to_numpy(dtype=float)
            observed_value = float(current[observed_column])
            payload[f"{prefix}_null_q95"] = float(np.quantile(values, 0.95))
            payload[f"{prefix}_permutation_p_ge"] = float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            )
            payload[f"{prefix}_above_null_q95"] = bool(
                observed_value > payload[f"{prefix}_null_q95"]
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _factor_effects(channel_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_bank, bank in channel_summary.groupby("feature_bank", sort=True):
        for family, local in (
            ("simultaneous", bank.loc[bank["slot"].eq("simultaneous")]),
            ("sequential", bank.loc[~bank["slot"].eq("simultaneous")]),
        ):
            route = local.groupby("control_route", as_index=False)[
                ["memory_total", "memory_early_d1_d5", "memory_delay5"]
            ].mean()
            for _, row in route.iterrows():
                rows.append(
                    {
                        "feature_bank": str(feature_bank),
                        "family": family,
                        "factor": "control_route",
                        "level": str(row["control_route"]),
                        "memory_total": float(row["memory_total"]),
                        "memory_early_d1_d5": float(row["memory_early_d1_d5"]),
                        "memory_delay5": float(row["memory_delay5"]),
                    }
                )
        sequential = bank.loc[~bank["slot"].eq("simultaneous")]
        slot = sequential.groupby("slot", as_index=False)[
            ["memory_total", "memory_early_d1_d5", "memory_delay5"]
        ].mean()
        for _, row in slot.iterrows():
            rows.append(
                {
                    "feature_bank": str(feature_bank),
                    "family": "sequential",
                    "factor": "slot",
                    "level": str(row["slot"]),
                    "memory_total": float(row["memory_total"]),
                    "memory_early_d1_d5": float(row["memory_early_d1_d5"]),
                    "memory_delay5": float(row["memory_delay5"]),
                }
            )
    return pd.DataFrame(rows)


def _render_plots(
    aggregate: pd.DataFrame,
    factor_effects: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    full = aggregate.loc[aggregate["feature_bank"].eq("full_one_two_body")].copy()
    full = full.sort_values("encoding_case").reset_index(drop=True)
    x = np.arange(len(full))
    width = 0.36

    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    axis.bar(x - width / 2, full["channel1_early_memory"], width, label="channel 1")
    axis.bar(x + width / 2, full["channel2_early_memory"], width, label="channel 2")
    axis.set_xticks(x)
    axis.set_xticklabels(full["encoding_case"], rotation=28, ha="right")
    axis.set_ylabel("Sum of delay-1 through delay-5 capacities")
    axis.set_title("Channel memory follows encoding route and slot")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "channel_early_memory_by_encoding.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(12.0, 5.7))
    for bank, group in aggregate.groupby("feature_bank", sort=True):
        ordered = group.set_index("encoding_case").reindex(full["encoding_case"])
        axis.plot(
            x,
            ordered["early_memory_imbalance"],
            marker="o",
            label=str(bank),
        )
    axis.set_xticks(x)
    axis.set_xticklabels(full["encoding_case"], rotation=28, ha="right")
    axis.set_ylabel("|C1-C2| / (|C1|+|C2|)")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Encoding-induced channel imbalance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "channel_imbalance_by_encoding.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    axis.bar(x - width / 2, full["mixing_mean"], width, label="observed mixing")
    if "mixing_null_q95" in full:
        axis.bar(x + width / 2, full["mixing_null_q95"], width, label="95% permutation floor")
    axis.set_xticks(x)
    axis.set_xticklabels(full["encoding_case"], rotation=28, ha="right")
    axis.set_ylabel("Sum of normalized mixing capacities")
    axis.set_title("Cross-channel mixing versus finite-sample null")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "mixing_vs_permutation_floor.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    factors = factor_effects.loc[
        factor_effects["feature_bank"].eq("full_one_two_body")
        & factor_effects["family"].eq("sequential")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    for axis, factor in zip(axes, ("control_route", "slot"), strict=True):
        local = factors.loc[factors["factor"].eq(factor)]
        axis.bar(local["level"], local["memory_early_d1_d5"])
        axis.set_title(factor.replace("_", " "))
        axis.set_xlabel("level")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean delay-1 through delay-5 capacity")
    figure.suptitle("Operator and recency effects on channel memory")
    figure.tight_layout()
    filename = "operator_slot_effects.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_encoding_symmetry_assay(
    *,
    results_root: Path,
    config: BivariateEncodingSymmetryConfig = BivariateEncodingSymmetryConfig(),
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
        shot_seed=config.seeds[0],
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    reservoir_config.validate()
    geometry_config.validate()

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": reservoir_config.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "paired_design": True,
            "scientific_question": (
                "Does channel-memory imbalance follow the amplitude/detuning route, "
                "the first/second slot, or an input label?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    task_frames: list[pd.DataFrame] = []
    case_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        capacity_config = config.capacity_config(seed)
        windows = generate_bivariate_windows(capacity_config)
        targets = build_capacity_targets(windows, capacity_config)
        rng = np.random.default_rng(int(seed) + 9_001_171)
        permutations = tuple(
            rng.permutation(config.samples) for _ in range(config.permutations)
        )

        for case in ENCODING_CASES:
            probabilities, metadata = evolve_encoding_case_probabilities(
                windows,
                reservoir_config,
                geometry_config,
                case,
                interaction_scale=config.interaction_scale,
            )
            np.savez_compressed(
                probability_dir / f"seed_{seed}__{case.name}.npz",
                probabilities=probabilities,
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "encoding_case": case.name,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )
            banks = build_feature_banks(probabilities)
            for feature_bank in config.feature_banks:
                matrix = banks[feature_bank]
                task_metrics, diagnostics, candidates = fit_capacity_readout(
                    matrix, targets, capacity_config
                )
                task_metrics.insert(0, "seed", int(seed))
                task_metrics.insert(1, "encoding_case", case.name)
                task_metrics.insert(2, "feature_bank", str(feature_bank))
                task_frames.append(task_metrics)
                candidates.to_csv(
                    selection_dir
                    / f"seed_{seed}__{case.name}__{feature_bank}.csv",
                    index=False,
                )
                summary = capacity_summary(task_metrics)
                case_rows.append(
                    {
                        "seed": int(seed),
                        "encoding_case": case.name,
                        "feature_bank": str(feature_bank),
                        **diagnostics,
                        **summary,
                    }
                )
                channel_rows.extend(
                    _channel_memory_rows(
                        task_metrics,
                        case,
                        seed=int(seed),
                        feature_bank=str(feature_bank),
                    )
                )
                for null in _null_summary(
                    matrix, targets, capacity_config, permutations
                ):
                    null_rows.append(
                        {
                            "seed": int(seed),
                            "encoding_case": case.name,
                            "feature_bank": str(feature_bank),
                            **null,
                        }
                    )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    case_summary = pd.DataFrame(case_rows)
    channel_summary = pd.DataFrame(channel_rows)
    null_frame = pd.DataFrame(null_rows)
    observed = _aggregate_observed(case_summary, channel_summary)
    aggregate = _aggregate_null(null_frame, observed)
    factor_effects = _factor_effects(channel_summary)
    plots = _render_plots(aggregate, factor_effects, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    case_summary.to_csv(run_dir / "case_seed_summary.csv", index=False)
    channel_summary.to_csv(run_dir / "channel_memory_seed_summary.csv", index=False)
    null_frame.to_csv(run_dir / "permutation_null.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(run_dir / "encoding_symmetry_summary.csv", index=False)
    factor_effects.to_csv(run_dir / "operator_slot_effects.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    full = aggregate.loc[aggregate["feature_bank"].eq("full_one_two_body")]
    most_balanced = full.loc[full["early_memory_imbalance"].idxmin()]
    strongest_minimum = full.loc[full["minimum_early_memory"].idxmax()]
    strongest_mixing_margin = full.assign(
        mixing_margin=lambda frame: frame["mixing_mean"]
        - frame.get("mixing_null_q95", 0.0)
    ).sort_values("mixing_margin", ascending=False).iloc[0]
    summary = {
        "status": "bivariate_encoding_symmetry_complete",
        "seeds_completed": int(case_summary["seed"].nunique()),
        "encoding_cases_completed": int(case_summary["encoding_case"].nunique()),
        "most_balanced_full_observable_encoding": str(most_balanced["encoding_case"]),
        "most_balanced_early_memory_imbalance": float(
            most_balanced["early_memory_imbalance"]
        ),
        "strongest_minimum_early_memory_encoding": str(
            strongest_minimum["encoding_case"]
        ),
        "strongest_minimum_early_memory": float(
            strongest_minimum["minimum_early_memory"]
        ),
        "strongest_mixing_margin_encoding": str(
            strongest_mixing_margin["encoding_case"]
        ),
        "strongest_mixing_margin": float(strongest_mixing_margin["mixing_margin"]),
        "interpretation_rule": (
            "An advantage that follows amplitude identifies operator sensitivity; "
            "an advantage that follows the second slot identifies recency; persistence "
            "with a channel label after paired swaps indicates an implementation or "
            "target-alignment defect. Timescale tuning should use the encoding with the "
            "largest balanced early memory, but only mixing above its permutation floor "
            "counts as nonlinear integration."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
