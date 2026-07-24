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
    SyntheticTargets,
    build_capacity_targets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import _evolve_segment_batch
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    _branch_drive,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    BivariateCrossoverStabilityConfig,
    _metric_payload,
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
    staggered_ladder_positions,
)

PALINDROMIC_SCHEDULE_NAME = "crossover_Ahalf_B_Ahalf"


@dataclass(frozen=True)
class BivariateCrossoverInteractionConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_duration_us: float = 0.02
    interaction_scales: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        3.00,
    )
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
        if (
            not self.interaction_scales
            or len(set(self.interaction_scales)) != len(self.interaction_scales)
            or any(scale < 0 for scale in self.interaction_scales)
            or not any(scale > 0 for scale in self.interaction_scales)
        ):
            raise ValueError(
                "interaction scales must be unique, nonnegative, and include a positive scale"
            )
        if not np.isfinite(self.drive_phase_rad):
            raise ValueError("drive_phase_rad must be finite")
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        train_end = int(np.floor(self.samples * 0.60))
        validation_end = train_end + int(np.floor(self.samples * 0.20))
        if train_end % 2 or validation_end % 2:
            raise ValueError("split boundaries must preserve complete exchange pairs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def stability_config(self) -> BivariateCrossoverStabilityConfig:
        return BivariateCrossoverStabilityConfig(
            samples=self.samples,
            sequence_length=self.sequence_length,
            memory_delays=self.memory_delays,
            alphas=self.alphas,
            seeds=self.seeds,
            interaction_scale=1.0,
            drive_phase_rad=self.drive_phase_rad,
            permutations=self.permutations,
        )


def _schedule():
    return next(
        schedule
        for schedule in CROSSOVER_SCHEDULES
        if schedule.name == PALINDROMIC_SCHEDULE_NAME
    )


def equivalent_geometry_factor(interaction_scale: float) -> float:
    """Return the uniform coordinate multiplier reproducing a C6 scale on hardware."""

    scale = float(interaction_scale)
    if scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if scale == 0:
        return float("nan")
    return float(scale ** (-1.0 / 6.0))


def minimum_pair_distance_um(geometry: StaggeredLadderGeometryConfig) -> float:
    positions = staggered_ladder_positions(geometry)
    distances = np.sqrt(
        np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
    )
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _evolve_interaction_off_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the same palindrome with the pair-interaction Hamiltonian disabled."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("interaction sweep requires an exact six-atom reservoir")

    schedule = _schedule()
    precomputed = precompute_ladder(reservoir, geometry, interaction_scale=0.0)
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
                interactions=False,
            )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1), n_atoms=precomputed.n_atoms
    )
    return stacked, {
        "encoding": "symmetric_crossover",
        "schedule": schedule.name,
        "interaction_scale": 0.0,
        "interactions_enabled": False,
        "probe_steps": [int(value) for value in probe_steps],
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & np.isclose(
                null_frame["interaction_scale"].to_numpy(dtype=float),
                float(current["interaction_scale"]),
            )
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for interaction case")
        payload = current.to_dict()
        for metric in (
            "channel1_early",
            "channel2_early",
            "minimum_early",
            "minimum_delay5",
            "mixing_sum",
            "order",
        ):
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
    result["mixing_margin"] = result["mixing_sum"] - result["mixing_sum_null_q95"]
    return result


def _aggregate(seed_summary: pd.DataFrame) -> pd.DataFrame:
    return seed_summary.groupby("interaction_scale", as_index=False).agg(
        seeds=("seed", "nunique"),
        equivalent_geometry_factor=("equivalent_geometry_factor", "first"),
        equivalent_min_spacing_um=("equivalent_min_spacing_um", "first"),
        channel1_early_mean=("channel1_early", "mean"),
        channel2_early_mean=("channel2_early", "mean"),
        minimum_early_mean=("minimum_early", "mean"),
        minimum_early_min=("minimum_early", "min"),
        early_balance_ratio_mean=("early_balance_ratio", "mean"),
        early_balance_ratio_min=("early_balance_ratio", "min"),
        channel1_delay5_mean=("channel1_delay5", "mean"),
        channel2_delay5_mean=("channel2_delay5", "mean"),
        minimum_delay5_mean=("minimum_delay5", "mean"),
        minimum_delay5_min=("minimum_delay5", "min"),
        mixing_mean=("mixing_sum", "mean"),
        mixing_max=("mixing_sum", "max"),
        mixing_margin_mean=("mixing_margin", "mean"),
        mixing_margin_min=("mixing_margin", "min"),
        order_mean=("order", "mean"),
        effective_rank_mean=("effective_rank_train_validation", "mean"),
        both_channels_early_above_null_fraction=(
            "both_channels_early_above_null",
            "mean",
        ),
        minimum_delay5_above_null_fraction=(
            "minimum_delay5_above_null_q95",
            "mean",
        ),
        mixing_above_null_fraction=("mixing_sum_above_null_q95", "mean"),
    )


def _memory_curves(task_metrics: pd.DataFrame) -> pd.DataFrame:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")].copy()
    return memory.groupby(["interaction_scale", "channel", "delay_a"], as_index=False).agg(
        capacity_mean=("capacity", "mean"),
        capacity_min=("capacity", "min"),
    )


def _render_plots(
    aggregate: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    ordered = aggregate.sort_values("interaction_scale").reset_index(drop=True)
    x = ordered["interaction_scale"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(x, ordered["channel1_delay5_mean"], marker="o", label="channel 1")
    axis.plot(x, ordered["channel2_delay5_mean"], marker="o", label="channel 2")
    axis.plot(x, ordered["minimum_delay5_mean"], marker="o", label="minimum")
    axis.set_xlabel("Interaction scale")
    axis.set_ylabel("Mean delay-5 normalized capacity")
    axis.set_title("Delay-5 memory versus interaction strength")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "delay5_capacity_vs_interaction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(x, ordered["mixing_mean"], marker="o", label="observed mixing")
    axis.plot(
        x,
        ordered["mixing_mean"] - ordered["mixing_margin_mean"],
        marker="o",
        label="mean 95% null",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Interaction scale")
    axis.set_ylabel("Summed normalized mixing capacity")
    axis.set_title("Mixing versus paired-permutation floor")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "mixing_vs_interaction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(x, ordered["minimum_early_mean"], marker="o", label="minimum early memory")
    axis.plot(x, ordered["early_balance_ratio_min"], marker="o", label="worst-seed balance")
    axis.set_xlabel("Interaction scale")
    axis.set_ylabel("Capacity / ratio")
    axis.set_title("Balanced memory versus interaction strength")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "balanced_memory_vs_interaction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(10.2, 6.0))
    for scale, group in curves.groupby("interaction_scale", sort=True):
        channels = group.pivot(index="delay_a", columns="channel", values="capacity_mean")
        axis.plot(
            channels.index,
            channels.min(axis=1),
            marker="o",
            label=f"{scale:g}",
        )
    axis.set_xlabel("Delay")
    axis.set_ylabel("Minimum mean channel capacity")
    axis.set_title("Worst-channel memory curve by interaction scale")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, title="scale")
    figure.tight_layout()
    filename = "minimum_memory_curve_by_interaction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_crossover_interaction_assay(
    *,
    results_root: Path,
    config: BivariateCrossoverInteractionConfig = BivariateCrossoverInteractionConfig(),
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
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    base_reservoir = replace(base_reservoir, step_duration_us=config.step_duration_us)
    base_reservoir.validate()
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("interaction assay requires the exact six-atom reservoir")

    base_min_spacing = minimum_pair_distance_um(geometry_config)
    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "base_minimum_pair_distance_um": base_min_spacing,
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_across_interaction_scales": True,
            "schedule": PALINDROMIC_SCHEDULE_NAME,
            "feature_bank": "occupation_pair_raw",
            "hardware_translation": "uniform coordinate factor = interaction_scale^(-1/6)",
            "scientific_question": (
                "At the selected 0.02 us input duration, can interaction strength create "
                "permutation-significant cross-channel mixing while preserving balanced "
                "delay-5 memory?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    stability_config = config.stability_config()
    task_frames: list[pd.DataFrame] = []
    observed_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        capacity_config = stability_config.capacity_config(seed)
        windows = generate_exchange_symmetric_windows(stability_config, seed=int(seed))
        targets = build_capacity_targets(windows, capacity_config)
        rng = np.random.default_rng(int(seed) + 577_215_664)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for scale in config.interaction_scales:
            scale = float(scale)
            if scale == 0.0:
                probabilities, metadata = _evolve_interaction_off_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    drive_phase_rad=config.drive_phase_rad,
                )
            else:
                probabilities, metadata = evolve_crossover_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    _schedule(),
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
            scale_tag = str(scale).replace(".", "p")
            np.savez_compressed(
                probability_dir / f"seed_{seed}__interaction_{scale_tag}.npz",
                probabilities=probabilities,
            )
            factor = equivalent_geometry_factor(scale)
            effective_spacing = (
                float(base_min_spacing * factor) if np.isfinite(factor) else float("nan")
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "interaction_scale": scale,
                    "equivalent_geometry_factor": factor,
                    "equivalent_min_spacing_um": effective_spacing,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )
            matrix = build_crossover_feature_banks(probabilities)["occupation_pair_raw"]
            task_metrics, diagnostics, candidates = fit_capacity_readout(
                matrix,
                targets,
                capacity_config,
            )
            task_metrics.insert(0, "seed", int(seed))
            task_metrics.insert(1, "interaction_scale", scale)
            task_frames.append(task_metrics)
            candidates.to_csv(
                selection_dir / f"seed_{seed}__interaction_{scale_tag}.csv",
                index=False,
            )
            observed_rows.append(
                {
                    "seed": int(seed),
                    "interaction_scale": scale,
                    "equivalent_geometry_factor": factor,
                    "equivalent_min_spacing_um": effective_spacing,
                    **diagnostics,
                    **_metric_payload(task_metrics),
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
                        "interaction_scale": scale,
                        "permutation": int(permutation),
                        **_metric_payload(null_metrics),
                    }
                )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    null_frame = pd.DataFrame(null_rows)
    seed_summary = _annotate_nulls(observed, null_frame)
    aggregate = _aggregate(seed_summary)
    curves = _memory_curves(task_metrics)
    plots = _render_plots(aggregate, curves, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "interaction_seed_summary.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(run_dir / "interaction_summary.csv", index=False)
    curves.to_csv(run_dir / "memory_curves.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    positive = aggregate.loc[aggregate["interaction_scale"].gt(0.0)].copy()
    memory_eligible = positive.loc[
        positive["both_channels_early_above_null_fraction"].eq(1.0)
        & positive["minimum_delay5_above_null_fraction"].eq(1.0)
        & positive["early_balance_ratio_min"].ge(0.80)
    ].copy()
    mixing_pass = memory_eligible.loc[
        memory_eligible["mixing_above_null_fraction"].eq(1.0)
        & memory_eligible["mixing_margin_min"].gt(0.0)
    ].copy()
    preserving_pool = memory_eligible if not memory_eligible.empty else positive
    strongest_preserving = preserving_pool.sort_values(
        ["mixing_margin_mean", "minimum_delay5_mean", "minimum_early_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    promoted = None
    if not mixing_pass.empty:
        promoted = mixing_pass.sort_values(
            ["mixing_margin_mean", "minimum_delay5_mean"],
            ascending=[False, False],
        ).iloc[0]
    interaction_off = aggregate.loc[aggregate["interaction_scale"].eq(0.0)].iloc[0]

    summary = {
        "status": "bivariate_crossover_interaction_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "interaction_scales_completed": int(seed_summary["interaction_scale"].nunique()),
        "step_duration_us": float(config.step_duration_us),
        "memory_eligible_scale_count": int(len(memory_eligible)),
        "mixing_gate_passed": bool(promoted is not None),
        "promoted_interaction_scale": (
            float(promoted["interaction_scale"]) if promoted is not None else None
        ),
        "promoted_equivalent_geometry_factor": (
            float(promoted["equivalent_geometry_factor"]) if promoted is not None else None
        ),
        "strongest_memory_preserving_scale": float(
            strongest_preserving["interaction_scale"]
        ),
        "strongest_memory_preserving_mixing_margin_mean": float(
            strongest_preserving["mixing_margin_mean"]
        ),
        "strongest_memory_preserving_minimum_delay5_mean": float(
            strongest_preserving["minimum_delay5_mean"]
        ),
        "interaction_off_minimum_delay5_mean": float(
            interaction_off["minimum_delay5_mean"]
        ),
        "interaction_off_mixing_mean": float(interaction_off["mixing_mean"]),
        "decision_rule": (
            "Promote an interaction scale only if balanced early and delay-5 memory survive "
            "in every seed and mixing exceeds its paired null in every seed. Otherwise, "
            "interaction strength alone has not repaired the mixing bottleneck."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
