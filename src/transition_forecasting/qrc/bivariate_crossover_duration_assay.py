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
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    BivariateCrossoverStabilityConfig,
    _metric_payload,
    generate_exchange_symmetric_windows,
    paired_permutation_orders,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

PALINDROMIC_SCHEDULE_NAME = "crossover_Ahalf_B_Ahalf"


@dataclass(frozen=True)
class BivariateCrossoverDurationConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_durations_us: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
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
        if (
            not self.step_durations_us
            or len(set(self.step_durations_us)) != len(self.step_durations_us)
            or any(duration <= 0 for duration in self.step_durations_us)
        ):
            raise ValueError("step durations must be positive and unique")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not np.isfinite(self.drive_phase_rad):
            raise ValueError("drive_phase_rad must be finite")
        if self.permutations < 0:
            raise ValueError("permutations cannot be negative")
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
            interaction_scale=self.interaction_scale,
            drive_phase_rad=self.drive_phase_rad,
            permutations=self.permutations,
        )


def _schedule():
    return next(
        schedule
        for schedule in CROSSOVER_SCHEDULES
        if schedule.name == PALINDROMIC_SCHEDULE_NAME
    )


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    if null_frame.empty:
        return observed.copy()
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & np.isclose(
                null_frame["step_duration_us"].to_numpy(dtype=float),
                float(current["step_duration_us"]),
            )
        ]
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
    return seed_summary.groupby("step_duration_us", as_index=False).agg(
        seeds=("seed", "nunique"),
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
    return memory.groupby(["step_duration_us", "channel", "delay_a"], as_index=False).agg(
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
    ordered = aggregate.sort_values("step_duration_us").reset_index(drop=True)
    x = ordered["step_duration_us"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    axis.plot(x, ordered["channel1_delay5_mean"], marker="o", label="channel 1")
    axis.plot(x, ordered["channel2_delay5_mean"], marker="o", label="channel 2")
    axis.plot(x, ordered["minimum_delay5_mean"], marker="o", label="minimum")
    axis.set_xscale("log")
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Mean delay-5 normalized capacity")
    axis.set_title("Palindromic crossover delay-5 memory")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "delay5_capacity_vs_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    axis.plot(x, ordered["minimum_early_mean"], marker="o", label="minimum early memory")
    axis.plot(x, ordered["early_balance_ratio_min"], marker="o", label="worst-seed balance")
    axis.set_xscale("log")
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Capacity / ratio")
    axis.set_title("Balanced early memory versus duration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "balanced_early_memory_vs_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    axis.plot(x, ordered["mixing_mean"], marker="o", label="observed mixing")
    axis.plot(
        x,
        ordered["mixing_mean"] - ordered["mixing_margin_mean"],
        marker="o",
        label="mean 95% null",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Summed normalized mixing capacity")
    axis.set_title("Mixing versus paired-permutation floor")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "mixing_vs_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(10.0, 6.0))
    for duration, group in curves.groupby("step_duration_us", sort=True):
        minimum = group.pivot(index="delay_a", columns="channel", values="capacity_mean")
        axis.plot(
            minimum.index,
            minimum.min(axis=1),
            marker="o",
            label=f"{duration:g} us",
        )
    axis.set_xlabel("Delay")
    axis.set_ylabel("Minimum mean channel capacity")
    axis.set_title("Worst-channel memory curve by duration")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    filename = "minimum_memory_curve_by_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_crossover_duration_assay(
    *,
    results_root: Path,
    config: BivariateCrossoverDurationConfig = BivariateCrossoverDurationConfig(),
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
        step_duration_us=0.03,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seeds[0],
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    base_reservoir.validate()
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("duration assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "base_reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_across_durations": True,
            "schedule": PALINDROMIC_SCHEDULE_NAME,
            "feature_bank": "occupation_pair_raw",
            "scientific_question": (
                "Which input duration gives both channels useful delay-1 through delay-5 "
                "memory, stable balance, and nonlinear mixing above a paired null?"
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
        rng = np.random.default_rng(int(seed) + 161_803_399)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )

        for duration in config.step_durations_us:
            reservoir_config = replace(
                base_reservoir,
                step_duration_us=float(duration),
                shot_seed=int(seed),
            )
            reservoir_config.validate()
            probabilities, metadata = evolve_crossover_probabilities(
                windows,
                reservoir_config,
                geometry_config,
                _schedule(),
                interaction_scale=config.interaction_scale,
                drive_phase_rad=config.drive_phase_rad,
            )
            duration_tag = str(float(duration)).replace(".", "p")
            np.savez_compressed(
                probability_dir / f"seed_{seed}__duration_{duration_tag}.npz",
                probabilities=probabilities,
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "step_duration_us": float(duration),
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
            task_metrics.insert(1, "step_duration_us", float(duration))
            task_frames.append(task_metrics)
            candidates.to_csv(
                selection_dir / f"seed_{seed}__duration_{duration_tag}.csv",
                index=False,
            )
            observed_rows.append(
                {
                    "seed": int(seed),
                    "step_duration_us": float(duration),
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
                        "step_duration_us": float(duration),
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
    seed_summary.to_csv(run_dir / "duration_seed_summary.csv", index=False)
    null_frame.to_csv(run_dir / "paired_permutation_null.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(run_dir / "duration_summary.csv", index=False)
    curves.to_csv(run_dir / "memory_curves.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    eligible = aggregate.loc[
        aggregate["both_channels_early_above_null_fraction"].eq(1.0)
        & aggregate["early_balance_ratio_min"].ge(0.80)
    ].copy()
    selection_pool = eligible if not eligible.empty else aggregate.copy()
    best_delay5 = selection_pool.sort_values(
        ["minimum_delay5_mean", "minimum_early_mean", "mixing_margin_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    best_mixing = aggregate.sort_values(
        ["mixing_margin_mean", "minimum_delay5_mean"],
        ascending=[False, False],
    ).iloc[0]
    summary = {
        "status": "bivariate_crossover_duration_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "durations_completed": int(seed_summary["step_duration_us"].nunique()),
        "eligible_duration_count": int(len(eligible)),
        "recommended_delay5_duration_us": float(best_delay5["step_duration_us"]),
        "recommended_minimum_delay5_mean": float(best_delay5["minimum_delay5_mean"]),
        "recommended_worst_seed_balance_ratio": float(best_delay5["early_balance_ratio_min"]),
        "strongest_mixing_duration_us": float(best_mixing["step_duration_us"]),
        "strongest_mixing_margin_mean": float(best_mixing["mixing_margin_mean"]),
        "interaction_sweep_rule": (
            "Promote a duration for interaction-scale testing only if both channels exceed "
            "their paired memory nulls across seeds, worst-seed balance remains at least "
            "0.8, and delay-5 memory improves. Mixing above null is preferred; otherwise "
            "the interaction sweep must target mixing explicitly."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
