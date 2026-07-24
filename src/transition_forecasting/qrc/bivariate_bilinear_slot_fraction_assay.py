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
from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
    Route,
    _route_drive,
)
from transition_forecasting.qrc.bivariate_capacity_assay import (
    SyntheticTargets,
    build_capacity_targets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_capacity_dynamics import _evolve_segment_batch
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
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
)

ALL_ORDERS = tuple(BILINEAR_SCHEDULES)
SAME_LAG_TASK = "mix_same_d1"


@dataclass(frozen=True)
class BivariateBilinearSlotFractionConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_duration_us: float = 0.015
    detuning_fractions: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70, 0.80)
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
        if (
            not self.detuning_fractions
            or len(set(self.detuning_fractions)) != len(self.detuning_fractions)
            or any(not 0.0 < fraction < 1.0 for fraction in self.detuning_fractions)
        ):
            raise ValueError("detuning fractions must be unique and lie strictly between zero and one")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
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
            interaction_scale=self.interaction_scale,
            drive_phase_rad=self.drive_phase_rad,
            permutations=self.permutations,
        )


def _route_fraction(route: Route, detuning_fraction: float) -> float:
    if route.startswith("D"):
        return float(detuning_fraction)
    if route.startswith("X"):
        return float(1.0 - detuning_fraction)
    raise ValueError(f"unsupported route: {route}")


def evolve_fractional_bilinear_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    routes: tuple[Route, Route],
    *,
    detuning_fraction: float,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve one D/X schedule with a fixed total time and unequal slot durations."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("slot-fraction assay requires the exact six-atom reservoir")
    if not 0.0 < detuning_fraction < 1.0:
        raise ValueError("detuning_fraction must lie strictly between zero and one")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if len(routes) != 2 or sum(route.startswith("D") for route in routes) != 1:
        raise ValueError("each bilinear schedule must contain exactly one detuning route")
    if sum(route.startswith("X") for route in routes) != 1:
        raise ValueError("each bilinear schedule must contain exactly one amplitude route")

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

    for step in range(steps):
        for route in routes:
            omega, delta, phase = _route_drive(
                values,
                step,
                route,
                reservoir,
                drive_phase_rad,
            )
            states = _evolve_segment_batch(
                states,
                omega,
                delta,
                phase,
                reservoir.step_duration_us * _route_fraction(route, detuning_fraction),
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
        "encoding": "pure_bilinear_noncommuting_slot_fraction",
        "routes": list(routes),
        "interaction_scale": float(interaction_scale),
        "interactions_enabled": bool(interactions),
        "step_duration_us": float(reservoir.step_duration_us),
        "detuning_fraction": float(detuning_fraction),
        "amplitude_fraction": float(1.0 - detuning_fraction),
        "detuning_duration_us": float(reservoir.step_duration_us * detuning_fraction),
        "amplitude_duration_us": float(
            reservoir.step_duration_us * (1.0 - detuning_fraction)
        ),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def build_all_orders_joint(raw_features: dict[str, np.ndarray]) -> np.ndarray:
    missing = set(ALL_ORDERS).difference(raw_features)
    if missing:
        raise ValueError(f"missing ordered feature matrices: {sorted(missing)}")
    matrices = [np.asarray(raw_features[name], dtype=float) for name in ALL_ORDERS]
    if len({matrix.shape for matrix in matrices}) != 1 or matrices[0].shape[1] != 63:
        raise ValueError("all ordered matrices must be aligned 63-feature banks")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("ordered feature matrices must be finite")
    return np.concatenate(matrices, axis=1)


def _mixing_decomposition(task_metrics: pd.DataFrame) -> dict[str, float]:
    mixing = task_metrics.loc[task_metrics["group"].eq("mixing")]
    if mixing.empty or SAME_LAG_TASK not in set(mixing["task"]):
        raise RuntimeError("capacity output is missing the same-lag mixing task")
    same = float(mixing.loc[mixing["task"].eq(SAME_LAG_TASK), "capacity"].sum())
    delayed = float(mixing.loc[~mixing["task"].eq(SAME_LAG_TASK), "capacity"].sum())
    return {
        "same_lag_mixing": same,
        "delayed_mixing": delayed,
    }


def _task_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    return {**_metric_payload(task_metrics), **_mixing_decomposition(task_metrics)}


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "channel1_early",
        "channel2_early",
        "minimum_early",
        "minimum_delay5",
        "mixing_sum",
        "same_lag_mixing",
        "delayed_mixing",
        "order",
    )
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & np.isclose(
                null_frame["detuning_fraction"].to_numpy(dtype=float),
                float(current["detuning_fraction"]),
            )
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for slot-fraction case")
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
    for metric in ("mixing_sum", "same_lag_mixing", "delayed_mixing"):
        result[f"{metric}_margin"] = result[metric] - result[f"{metric}_null_q95"]
    return result


def _paired_on_off(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "minimum_early",
        "minimum_delay5",
        "mixing_sum",
        "same_lag_mixing",
        "delayed_mixing",
        "order",
    ]
    pivot = seed_summary.pivot_table(
        index=["seed", "detuning_fraction"],
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
        ["interaction", "detuning_fraction"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        channel1_early_mean=("channel1_early", "mean"),
        channel2_early_mean=("channel2_early", "mean"),
        minimum_early_mean=("minimum_early", "mean"),
        minimum_early_min=("minimum_early", "min"),
        early_balance_ratio_mean=("early_balance_ratio", "mean"),
        early_balance_ratio_min=("early_balance_ratio", "min"),
        minimum_delay5_mean=("minimum_delay5", "mean"),
        minimum_delay5_min=("minimum_delay5", "min"),
        mixing_mean=("mixing_sum", "mean"),
        mixing_margin_mean=("mixing_sum_margin", "mean"),
        mixing_margin_min=("mixing_sum_margin", "min"),
        same_lag_mixing_mean=("same_lag_mixing", "mean"),
        same_lag_margin_mean=("same_lag_mixing_margin", "mean"),
        same_lag_margin_min=("same_lag_mixing_margin", "min"),
        delayed_mixing_mean=("delayed_mixing", "mean"),
        delayed_margin_mean=("delayed_mixing_margin", "mean"),
        delayed_margin_min=("delayed_mixing_margin", "min"),
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
        same_lag_above_null_fraction=("same_lag_mixing_above_null_q95", "mean"),
        delayed_above_null_fraction=("delayed_mixing_above_null_q95", "mean"),
    )
    paired_aggregate = paired.groupby("detuning_fraction", as_index=False).agg(
        same_lag_on_minus_off_mean=("same_lag_mixing_on_minus_off", "mean"),
        same_lag_on_minus_off_min=("same_lag_mixing_on_minus_off", "min"),
        delayed_on_minus_off_mean=("delayed_mixing_on_minus_off", "mean"),
        delayed_on_minus_off_min=("delayed_mixing_on_minus_off", "min"),
        delay5_on_minus_off_mean=("minimum_delay5_on_minus_off", "mean"),
    )
    return aggregate.merge(
        paired_aggregate,
        on="detuning_fraction",
        how="left",
        validate="many_to_one",
    )


def _render_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    on = summary.loc[summary["interaction"].eq("on")].sort_values(
        "detuning_fraction"
    )
    x = on["detuning_fraction"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(x, on["same_lag_margin_mean"], marker="o", label="same-lag margin")
    axis.plot(x, on["delayed_margin_mean"], marker="o", label="delayed margin")
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Detuning share of each timestep")
    axis.set_ylabel("Mean capacity margin versus paired null")
    axis.set_title("Same-lag and delayed mixing versus D/X allocation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "mixing_margin_vs_detuning_fraction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(x, on["minimum_delay5_mean"], marker="o", label="delay-5 memory")
    axis.plot(x, on["minimum_early_mean"], marker="o", label="early memory")
    axis.set_xlabel("Detuning share of each timestep")
    axis.set_ylabel("Mean minimum-channel capacity")
    axis.set_title("Balanced memory versus D/X allocation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "memory_vs_detuning_fraction.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    axis.plot(
        x,
        on["same_lag_on_minus_off_mean"],
        marker="o",
        label="mean on-off advantage",
    )
    axis.plot(
        x,
        on["same_lag_on_minus_off_min"],
        marker="o",
        label="worst-seed on-off advantage",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Detuning share of each timestep")
    axis.set_ylabel("Same-lag mixing: interactions on minus off")
    axis.set_title("Interaction-assisted same-lag mixing")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "same_lag_on_off_advantage.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_bilinear_slot_fraction_assay(
    *,
    results_root: Path,
    config: BivariateBilinearSlotFractionConfig = BivariateBilinearSlotFractionConfig(),
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
        raise ValueError("slot-fraction assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_across_slot_fractions": True,
            "paired_interaction_control": True,
            "representation": "all_orders_joint",
            "feature_count": 252,
            "scientific_question": (
                "At the 0.015 us bilinear near-pass, can reallocating fixed evolution time "
                "between the weaker detuning route and stronger amplitude route make same-lag "
                "or delayed cross-channel mixing reproducible across all seeds without losing "
                "balanced delay-5 memory?"
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
        rng = np.random.default_rng(int(seed) + 3_141_592_654)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for detuning_fraction in config.detuning_fractions:
            fraction = float(detuning_fraction)
            fraction_tag = str(fraction).replace(".", "p")
            for interaction in ("off", "on"):
                scale = config.interaction_scale if interaction == "on" else 0.0
                raw_features: dict[str, np.ndarray] = {}
                for schedule_name, routes in BILINEAR_SCHEDULES.items():
                    probabilities, metadata = evolve_fractional_bilinear_probabilities(
                        windows,
                        reservoir_config,
                        geometry_config,
                        routes,
                        detuning_fraction=fraction,
                        interaction_scale=scale,
                        drive_phase_rad=config.drive_phase_rad,
                    )
                    raw_features[schedule_name] = build_crossover_feature_banks(
                        probabilities
                    )["occupation_pair_raw"]
                    np.savez_compressed(
                        probability_dir
                        / (
                            f"seed_{seed}__detuning_{fraction_tag}__{interaction}"
                            f"__{schedule_name}.npz"
                        ),
                        probabilities=probabilities,
                    )
                    metadata_rows.append(
                        {
                            "seed": int(seed),
                            "detuning_fraction": fraction,
                            "interaction": interaction,
                            "schedule": schedule_name,
                            "metadata_json": json.dumps(metadata, sort_keys=True),
                        }
                    )

                matrix = build_all_orders_joint(raw_features)
                task_metrics, diagnostics, candidates = fit_capacity_readout(
                    matrix,
                    targets,
                    capacity_config,
                )
                task_metrics.insert(0, "seed", int(seed))
                task_metrics.insert(1, "detuning_fraction", fraction)
                task_metrics.insert(2, "interaction", interaction)
                task_frames.append(task_metrics)
                candidates.to_csv(
                    selection_dir
                    / f"seed_{seed}__detuning_{fraction_tag}__{interaction}.csv",
                    index=False,
                )
                observed_rows.append(
                    {
                        "seed": int(seed),
                        "detuning_fraction": fraction,
                        "amplitude_fraction": float(1.0 - fraction),
                        "interaction": interaction,
                        "feature_count": int(matrix.shape[1]),
                        **diagnostics,
                        **_task_payload(task_metrics),
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
                            "detuning_fraction": fraction,
                            "interaction": interaction,
                            "permutation": int(permutation),
                            **_task_payload(null_metrics),
                        }
                    )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    null_frame = pd.DataFrame(null_rows)
    seed_summary = _annotate_nulls(observed, null_frame)
    paired = _paired_on_off(seed_summary)
    aggregate = _aggregate(seed_summary, paired)
    plots = _render_plots(aggregate, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "slot_fraction_seed_summary.csv", index=False)
    aggregate.to_csv(run_dir / "slot_fraction_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    on = aggregate.loc[aggregate["interaction"].eq("on")].copy()
    eligible = on.loc[
        on["both_channels_early_above_null_fraction"].eq(1.0)
        & on["minimum_delay5_above_null_fraction"].eq(1.0)
        & on["early_balance_ratio_min"].ge(0.80)
        & on["same_lag_above_null_fraction"].eq(1.0)
        & on["same_lag_margin_min"].gt(0.0)
        & on["same_lag_on_minus_off_min"].gt(0.0)
    ].copy()
    promoted = None
    if not eligible.empty:
        promoted = eligible.sort_values(
            ["same_lag_margin_mean", "minimum_delay5_mean", "delayed_margin_mean"],
            ascending=[False, False, False],
        ).iloc[0]
    strongest = on.sort_values(
        ["same_lag_margin_mean", "minimum_delay5_mean", "delayed_margin_mean"],
        ascending=[False, False, False],
    ).iloc[0]

    summary = {
        "status": "bivariate_bilinear_slot_fraction_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "slot_fractions_completed": int(seed_summary["detuning_fraction"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "step_duration_us": float(config.step_duration_us),
        "interaction_scale": float(config.interaction_scale),
        "same_lag_gate_passed": bool(promoted is not None),
        "promoted_detuning_fraction": (
            float(promoted["detuning_fraction"]) if promoted is not None else None
        ),
        "promoted_amplitude_fraction": (
            float(1.0 - promoted["detuning_fraction"]) if promoted is not None else None
        ),
        "promoted_delayed_mixing_above_null_fraction": (
            float(promoted["delayed_above_null_fraction"])
            if promoted is not None
            else None
        ),
        "strongest_candidate_detuning_fraction": float(
            strongest["detuning_fraction"]
        ),
        "strongest_candidate_same_lag_margin_mean": float(
            strongest["same_lag_margin_mean"]
        ),
        "strongest_candidate_same_lag_worst_seed_margin": float(
            strongest["same_lag_margin_min"]
        ),
        "strongest_candidate_minimum_delay5_mean": float(
            strongest["minimum_delay5_mean"]
        ),
        "delayed_mixing_gate_passed": bool(
            promoted is not None
            and float(promoted["delayed_above_null_fraction"]) == 1.0
            and float(promoted["delayed_margin_min"]) > 0.0
            and float(promoted["delayed_on_minus_off_min"]) > 0.0
        ),
        "decision_rule": (
            "Promote a D/X allocation only if balanced early and delay-5 memory, same-lag "
            "mixing above paired null, and positive interaction assistance all hold in every "
            "seed. Delayed cross-lag mixing is reported separately. If no allocation passes, "
            "close scalar duration/interaction/allocation tuning for this architecture."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
