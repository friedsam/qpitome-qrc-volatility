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
    _evolve_segment_batch,
    build_feature_banks,
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

BranchName = Literal["A", "B"]
CrossoverFeatureBank = Literal[
    "six_mode_density_curvature",
    "occupation_pair_raw",
    "full_one_two_body",
]


@dataclass(frozen=True)
class CrossoverSchedule:
    name: str
    segments: tuple[tuple[BranchName, float], ...]

    def validate(self) -> None:
        if not self.name or not self.segments:
            raise ValueError("crossover schedule must be named and nonempty")
        if any(branch not in ("A", "B") for branch, _ in self.segments):
            raise ValueError("crossover branches must be A or B")
        if any(fraction <= 0 for _, fraction in self.segments):
            raise ValueError("crossover segment fractions must be positive")
        if not np.isclose(sum(fraction for _, fraction in self.segments), 1.0):
            raise ValueError("crossover segment fractions must sum to one")


CROSSOVER_SCHEDULES: tuple[CrossoverSchedule, ...] = (
    CrossoverSchedule("crossover_A_B", (("A", 0.5), ("B", 0.5))),
    CrossoverSchedule("crossover_B_A", (("B", 0.5), ("A", 0.5))),
    CrossoverSchedule(
        "crossover_Ahalf_B_Ahalf",
        (("A", 0.25), ("B", 0.5), ("A", 0.25)),
    ),
    CrossoverSchedule(
        "crossover_Bhalf_A_Bhalf",
        (("B", 0.25), ("A", 0.5), ("B", 0.25)),
    ),
)


@dataclass(frozen=True)
class BivariateCrossoverConfig:
    samples: int = 320
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    interaction_scale: float = 1.25
    drive_phase_rad: float = 0.0
    permutations: int = 32
    feature_banks: tuple[CrossoverFeatureBank, ...] = (
        "six_mode_density_curvature",
        "occupation_pair_raw",
        "full_one_two_body",
    )

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("samples must be at least 100")
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
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not np.isfinite(self.drive_phase_rad):
            raise ValueError("drive_phase_rad must be finite")
        if self.permutations < 0:
            raise ValueError("permutations cannot be negative")
        supported = {
            "six_mode_density_curvature",
            "occupation_pair_raw",
            "full_one_two_body",
        }
        if not self.feature_banks or set(self.feature_banks).difference(supported):
            raise ValueError("unsupported crossover feature bank")
        for schedule in CROSSOVER_SCHEDULES:
            schedule.validate()

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


def _branch_drive(
    windows: np.ndarray,
    step: int,
    branch: BranchName,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if branch == "A":
        detuning_channel, amplitude_channel = 0, 1
    elif branch == "B":
        detuning_channel, amplitude_channel = 1, 0
    else:
        raise ValueError(f"unsupported crossover branch: {branch}")

    detuning_input = windows[:, step, detuning_channel]
    amplitude_input = windows[:, step, amplitude_channel]
    omega = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * amplitude_input
    )
    if np.any(omega <= 0):
        raise ValueError("crossover amplitude encoding became nonpositive")
    delta = reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * detuning_input
    phase = np.full(len(windows), float(phase_rad), dtype=float)
    return omega, delta, phase


def evolve_crossover_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    schedule: CrossoverSchedule,
    *,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply mirrored composite controls with equal operator exposure per channel."""

    schedule.validate()
    reservoir.validate()
    geometry.validate()
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("crossover assay requires an exact six-atom reservoir")
    if interaction_scale <= 0:
        raise ValueError("crossover assay requires positive interactions")
    if not np.isfinite(drive_phase_rad):
        raise ValueError("drive_phase_rad must be finite")

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
                drive_phase_rad,
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                reservoir.step_duration_us * float(fraction),
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
    metadata = {
        "encoding": "symmetric_crossover",
        "schedule": schedule.name,
        "segments": [[branch, float(fraction)] for branch, fraction in schedule.segments],
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "interaction_scale": float(interaction_scale),
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
        "equal_amplitude_exposure_per_channel": True,
        "equal_detuning_exposure_per_channel": True,
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }
    return stacked, metadata


def build_crossover_feature_banks(
    probabilities: np.ndarray,
) -> dict[CrossoverFeatureBank, np.ndarray]:
    """Return historical, hardware-natural, and diagnostic readout banks."""

    base = build_feature_banks(probabilities)
    full = np.asarray(base["full_one_two_body"], dtype=float)
    probes = validate_probe_probabilities(probabilities).shape[1]
    full_block_width = 36
    raw_block_width = 21
    raw_indices = np.asarray(
        [
            probe * full_block_width + offset
            for probe in range(probes)
            for offset in range(raw_block_width)
        ],
        dtype=int,
    )
    banks: dict[CrossoverFeatureBank, np.ndarray] = {
        "six_mode_density_curvature": np.asarray(
            base["six_mode_density_curvature"], dtype=float
        ),
        "occupation_pair_raw": full[:, raw_indices],
        "full_one_two_body": full,
    }
    expected = {
        "six_mode_density_curvature": probes * 2,
        "occupation_pair_raw": probes * 21,
        "full_one_two_body": probes * 36,
    }
    for name, matrix in banks.items():
        if matrix.shape != (len(full), expected[name]):
            raise RuntimeError(f"unexpected feature width for {name}: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite crossover features in {name}")
    return banks


def _channel_memory_rows(
    task_metrics: pd.DataFrame,
    *,
    seed: int,
    schedule: str,
    feature_bank: str,
) -> list[dict[str, object]]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    rows: list[dict[str, object]] = []
    for channel in (1, 2):
        local = memory.loc[memory["channel"].eq(channel)]
        early = local.loc[local["delay_a"].le(5)]
        delay_five = local.loc[local["delay_a"].eq(5)]
        rows.append(
            {
                "seed": int(seed),
                "schedule": str(schedule),
                "feature_bank": str(feature_bank),
                "channel": int(channel),
                "memory_total": float(local["capacity"].sum()),
                "memory_early_d1_d5": float(early["capacity"].sum()),
                "memory_delay5": (
                    float(delay_five["capacity"].mean())
                    if not delay_five.empty
                    else float("nan")
                ),
            }
        )
    return rows


def _null_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    channel_early = []
    channel_delay5 = []
    for channel in (1, 2):
        local = memory.loc[memory["channel"].eq(channel)]
        channel_early.append(float(local.loc[local["delay_a"].le(5), "capacity"].sum()))
        channel_delay5.append(
            float(local.loc[local["delay_a"].eq(5), "capacity"].mean())
        )
    summary = capacity_summary(task_metrics)
    return {
        "channel1_early": channel_early[0],
        "channel2_early": channel_early[1],
        "minimum_early": float(min(channel_early)),
        "channel1_delay5": channel_delay5[0],
        "channel2_delay5": channel_delay5[1],
        "minimum_delay5": float(min(channel_delay5)),
        "mixing_sum": float(summary["mixing_capacity"]),
        "order": float(summary["order_capacity"]),
    }


def _aggregate_observed(
    case_summary: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> pd.DataFrame:
    pivot = channel_summary.pivot_table(
        index=["seed", "schedule", "feature_bank"],
        columns="channel",
        values=["memory_total", "memory_early_d1_d5", "memory_delay5"],
    ).reset_index()
    pivot.columns = [
        "_".join(str(value) for value in column if str(value) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    for metric in ("memory_total", "memory_early_d1_d5", "memory_delay5"):
        left = pivot[f"{metric}_1"]
        right = pivot[f"{metric}_2"]
        pivot[f"{metric}_minimum"] = np.minimum(left, right)
        pivot[f"{metric}_balance_ratio"] = np.minimum(left, right) / (
            np.maximum(left, right) + 1e-12
        )
        pivot[f"{metric}_imbalance"] = np.abs(left - right) / (
            np.abs(left) + np.abs(right) + 1e-12
        )

    channel_aggregate = pivot.groupby(
        ["schedule", "feature_bank"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        channel1_early_mean=("memory_early_d1_d5_1", "mean"),
        channel2_early_mean=("memory_early_d1_d5_2", "mean"),
        minimum_early_mean=("memory_early_d1_d5_minimum", "mean"),
        early_balance_ratio_mean=("memory_early_d1_d5_balance_ratio", "mean"),
        early_balance_ratio_min=("memory_early_d1_d5_balance_ratio", "min"),
        early_imbalance_mean=("memory_early_d1_d5_imbalance", "mean"),
        channel1_delay5_mean=("memory_delay5_1", "mean"),
        channel2_delay5_mean=("memory_delay5_2", "mean"),
        minimum_delay5_mean=("memory_delay5_minimum", "mean"),
        delay5_balance_ratio_min=("memory_delay5_balance_ratio", "min"),
        total_memory_minimum_mean=("memory_total_minimum", "mean"),
    )
    case_aggregate = case_summary.groupby(
        ["schedule", "feature_bank"], as_index=False
    ).agg(
        mixing_mean=("mixing_capacity", "mean"),
        mixing_std=("mixing_capacity", "std"),
        order_mean=("order_capacity", "mean"),
        order_std=("order_capacity", "std"),
        effective_rank_mean=("effective_rank_train_validation", "mean"),
    )
    return channel_aggregate.merge(
        case_aggregate,
        on=["schedule", "feature_bank"],
        validate="one_to_one",
    )


def _aggregate_null(
    null_frame: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    if null_frame.empty:
        return observed.copy()
    averaged = null_frame.groupby(
        ["schedule", "feature_bank", "permutation"], as_index=False
    )[
        [
            "channel1_early",
            "channel2_early",
            "minimum_early",
            "channel1_delay5",
            "channel2_delay5",
            "minimum_delay5",
            "mixing_sum",
            "order",
        ]
    ].mean()
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = averaged.loc[
            averaged["schedule"].eq(current["schedule"])
            & averaged["feature_bank"].eq(current["feature_bank"])
        ]
        payload = current.to_dict()
        for observed_column, null_column, prefix in (
            ("minimum_early_mean", "minimum_early", "minimum_early"),
            ("minimum_delay5_mean", "minimum_delay5", "minimum_delay5"),
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


def _render_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = [schedule.name for schedule in CROSSOVER_SCHEDULES]

    raw = summary.loc[summary["feature_bank"].eq("occupation_pair_raw")].copy()
    raw = raw.set_index("schedule").reindex(order).reset_index()
    x = np.arange(len(raw))
    width = 0.36

    figure, axis = plt.subplots(figsize=(11.5, 6.0))
    axis.bar(x - width / 2, raw["channel1_early_mean"], width, label="channel 1")
    axis.bar(x + width / 2, raw["channel2_early_mean"], width, label="channel 2")
    axis.set_xticks(x)
    axis.set_xticklabels(raw["schedule"], rotation=25, ha="right")
    axis.set_ylabel("Mean summed capacity, delays 1-5")
    axis.set_title("Symmetric crossover channel memory")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "crossover_channel_memory.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    for bank, group in summary.groupby("feature_bank", sort=True):
        local = group.set_index("schedule").reindex(order)
        axis.plot(
            x,
            local["early_balance_ratio_min"],
            marker="o",
            label=str(bank),
        )
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=25, ha="right")
    axis.set_ylabel("Worst-seed min(C1,C2)/max(C1,C2)")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Within-seed crossover balance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "crossover_within_seed_balance.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.5, 6.0))
    axis.bar(x - width / 2, raw["mixing_mean"], width, label="observed mixing")
    if "mixing_null_q95" in raw:
        axis.bar(x + width / 2, raw["mixing_null_q95"], width, label="95% null")
    axis.set_xticks(x)
    axis.set_xticklabels(raw["schedule"], rotation=25, ha="right")
    axis.set_ylabel("Sum of normalized mixing capacities")
    axis.set_title("Crossover mixing versus permutation floor")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "crossover_mixing_vs_null.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    comparable = summary.loc[
        summary["feature_bank"].isin(["occupation_pair_raw", "full_one_two_body"])
    ].pivot(
        index="schedule",
        columns="feature_bank",
        values=["minimum_early_mean", "mixing_mean"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for axis, metric, title in (
        (axes[0], "minimum_early_mean", "Balanced early memory"),
        (axes[1], "mixing_mean", "Mixing capacity"),
    ):
        x_values = comparable[(metric, "occupation_pair_raw")]
        y_values = comparable[(metric, "full_one_two_body")]
        axis.scatter(x_values, y_values)
        low = float(min(x_values.min(), y_values.min()))
        high = float(max(x_values.max(), y_values.max()))
        axis.plot([low, high], [low, high], linestyle="--")
        axis.set_xlabel("63 raw occupation/pair features")
        axis.set_ylabel("108 diagnostic features")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle("Hardware-natural versus derived diagnostic bank")
    figure.tight_layout()
    filename = "raw63_vs_full108.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_crossover_assay(
    *,
    results_root: Path,
    config: BivariateCrossoverConfig = BivariateCrossoverConfig(),
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
                "Does equal amplitude/detuning exposure restore within-seed bivariate "
                "memory balance, and does any crossover schedule produce mixing above null?"
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
        rng = np.random.default_rng(int(seed) + 31_415_927)
        permutations = tuple(
            rng.permutation(config.samples) for _ in range(config.permutations)
        )

        for schedule in CROSSOVER_SCHEDULES:
            probabilities, metadata = evolve_crossover_probabilities(
                windows,
                reservoir_config,
                geometry_config,
                schedule,
                interaction_scale=config.interaction_scale,
                drive_phase_rad=config.drive_phase_rad,
            )
            np.savez_compressed(
                probability_dir / f"seed_{seed}__{schedule.name}.npz",
                probabilities=probabilities,
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "schedule": schedule.name,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )
            banks = build_crossover_feature_banks(probabilities)
            for feature_bank in config.feature_banks:
                matrix = banks[feature_bank]
                task_metrics, diagnostics, candidates = fit_capacity_readout(
                    matrix, targets, capacity_config
                )
                task_metrics.insert(0, "seed", int(seed))
                task_metrics.insert(1, "schedule", schedule.name)
                task_metrics.insert(2, "feature_bank", str(feature_bank))
                task_frames.append(task_metrics)
                candidates.to_csv(
                    selection_dir
                    / f"seed_{seed}__{schedule.name}__{feature_bank}.csv",
                    index=False,
                )
                summary = capacity_summary(task_metrics)
                case_rows.append(
                    {
                        "seed": int(seed),
                        "schedule": schedule.name,
                        "feature_bank": str(feature_bank),
                        **diagnostics,
                        **summary,
                    }
                )
                channel_rows.extend(
                    _channel_memory_rows(
                        task_metrics,
                        seed=int(seed),
                        schedule=schedule.name,
                        feature_bank=str(feature_bank),
                    )
                )
                for permutation, order in enumerate(permutations):
                    null_targets = SyntheticTargets(
                        values=np.asarray(targets.values, dtype=float)[order],
                        metadata=targets.metadata.copy(),
                    )
                    null_metrics, _, _ = fit_capacity_readout(
                        matrix, null_targets, capacity_config
                    )
                    null_rows.append(
                        {
                            "seed": int(seed),
                            "schedule": schedule.name,
                            "feature_bank": str(feature_bank),
                            "permutation": int(permutation),
                            **_null_payload(null_metrics),
                        }
                    )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    case_summary = pd.DataFrame(case_rows)
    channel_summary = pd.DataFrame(channel_rows)
    null_frame = pd.DataFrame(null_rows)
    observed = _aggregate_observed(case_summary, channel_summary)
    aggregate = _aggregate_null(null_frame, observed)
    plots = _render_plots(aggregate, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    case_summary.to_csv(run_dir / "case_seed_summary.csv", index=False)
    channel_summary.to_csv(run_dir / "channel_memory_seed_summary.csv", index=False)
    null_frame.to_csv(run_dir / "permutation_null.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(run_dir / "crossover_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    primary = aggregate.loc[aggregate["feature_bank"].eq("occupation_pair_raw")].copy()
    most_balanced = primary.sort_values(
        ["early_balance_ratio_min", "minimum_early_mean"], ascending=[False, False]
    ).iloc[0]
    strongest_memory = primary.sort_values(
        ["minimum_early_mean", "early_balance_ratio_min"], ascending=[False, False]
    ).iloc[0]
    strongest_mixing = primary.assign(
        mixing_margin=lambda frame: frame["mixing_mean"]
        - frame.get("mixing_null_q95", 0.0)
    ).sort_values("mixing_margin", ascending=False).iloc[0]
    raw_full = aggregate.loc[
        aggregate["feature_bank"].isin(["occupation_pair_raw", "full_one_two_body"])
    ]
    comparison = raw_full.pivot(
        index="schedule",
        columns="feature_bank",
        values=["minimum_early_mean", "mixing_mean"],
    )
    agreement = {
        metric: float(
            np.corrcoef(
                comparison[(metric, "occupation_pair_raw")],
                comparison[(metric, "full_one_two_body")],
            )[0, 1]
        )
        for metric in ("minimum_early_mean", "mixing_mean")
    }
    summary_payload = {
        "status": "bivariate_crossover_complete",
        "seeds_completed": int(case_summary["seed"].nunique()),
        "schedules_completed": int(case_summary["schedule"].nunique()),
        "primary_feature_bank": "occupation_pair_raw",
        "most_balanced_schedule": str(most_balanced["schedule"]),
        "most_balanced_worst_seed_ratio": float(
            most_balanced["early_balance_ratio_min"]
        ),
        "strongest_balanced_memory_schedule": str(strongest_memory["schedule"]),
        "strongest_minimum_early_memory": float(
            strongest_memory["minimum_early_mean"]
        ),
        "strongest_mixing_margin_schedule": str(strongest_mixing["schedule"]),
        "strongest_mixing_margin": float(strongest_mixing["mixing_margin"]),
        "raw63_full108_agreement": agreement,
        "duration_sweep_rule": (
            "Proceed to a small step-duration sweep only when a crossover schedule "
            "shows materially improved within-seed balance and both-channel early memory. "
            "Mixing above its permutation floor is desirable but not required before the "
            "first bounded duration sweep."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
