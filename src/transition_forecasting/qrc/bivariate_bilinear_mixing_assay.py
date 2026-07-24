from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

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
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_interaction_assay import (
    _evolve_interaction_off_probabilities,
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

Route = Literal["D1", "D2", "X1", "X2"]
InteractionCondition = Literal["on", "off"]

PALINDROMIC_SCHEDULE_NAME = "crossover_Ahalf_B_Ahalf"
BILINEAR_SCHEDULES: dict[str, tuple[Route, Route]] = {
    "d1_then_x2": ("D1", "X2"),
    "d2_then_x1": ("D2", "X1"),
    "x2_then_d1": ("X2", "D1"),
    "x1_then_d2": ("X1", "D2"),
}
REPRESENTATIONS = (
    "palindrome_control",
    "forward_mirror_concat",
    "reverse_mirror_concat",
    "commutator_contrast_concat",
)


@dataclass(frozen=True)
class BivariateBilinearMixingConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_duration_us: float = 0.02
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


def _palindromic_schedule():
    return next(
        schedule
        for schedule in CROSSOVER_SCHEDULES
        if schedule.name == PALINDROMIC_SCHEDULE_NAME
    )


def _route_drive(
    windows: np.ndarray,
    step: int,
    route: Route,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(windows, dtype=float)
    samples = len(values)
    phase = np.full(samples, float(phase_rad), dtype=float)
    zeros = np.zeros(samples, dtype=float)

    if route in ("D1", "D2"):
        channel = 0 if route == "D1" else 1
        omega = zeros
        delta = (
            reservoir.delta_center_rad_us
            + reservoir.delta_span_rad_us * values[:, step, channel]
        )
    elif route in ("X1", "X2"):
        channel = 0 if route == "X1" else 1
        omega = reservoir.omega_base_rad_us * (
            1.0 + reservoir.omega_mod_fraction * values[:, step, channel]
        )
        if np.any(omega <= 0):
            raise ValueError("bilinear amplitude encoding became nonpositive")
        delta = zeros
    else:
        raise ValueError(f"unsupported bilinear route: {route}")
    return omega, delta, phase


def evolve_bilinear_schedule_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    routes: tuple[Route, Route],
    *,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve a pure longitudinal/transverse two-slot schedule."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("bilinear assay requires an exact six-atom reservoir")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")
    if len(routes) != 2:
        raise ValueError("bilinear schedule must contain two routes")

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
                reservoir.step_duration_us * 0.5,
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
        "encoding": "pure_bilinear_noncommuting",
        "routes": list(routes),
        "interaction_scale": float(interaction_scale),
        "interactions_enabled": bool(interactions),
        "step_duration_us": float(reservoir.step_duration_us),
        "slot_duration_us": float(reservoir.step_duration_us * 0.5),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def build_bilinear_representations(
    raw_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    required = {"palindrome", *BILINEAR_SCHEDULES}
    missing = required.difference(raw_features)
    if missing:
        raise ValueError(f"missing bilinear feature matrices: {sorted(missing)}")
    matrices = {name: np.asarray(value, dtype=float) for name, value in raw_features.items()}
    shapes = {matrix.shape for matrix in matrices.values()}
    if len(shapes) != 1 or next(iter(shapes))[1] != 63:
        raise ValueError("all bilinear inputs must be aligned 63-feature matrices")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("bilinear feature matrices must be finite")

    forward = np.concatenate(
        [matrices["d1_then_x2"], matrices["d2_then_x1"]], axis=1
    )
    reverse = np.concatenate(
        [matrices["x2_then_d1"], matrices["x1_then_d2"]], axis=1
    )
    contrast = np.concatenate(
        [
            matrices["d1_then_x2"] - matrices["x2_then_d1"],
            matrices["d2_then_x1"] - matrices["x1_then_d2"],
        ],
        axis=1,
    )
    return {
        "palindrome_control": matrices["palindrome"],
        "forward_mirror_concat": forward,
        "reverse_mirror_concat": reverse,
        "commutator_contrast_concat": contrast,
    }


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & null_frame["representation"].eq(current["representation"])
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for bilinear case")
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
    return seed_summary.groupby(["interaction", "representation"], as_index=False).agg(
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
        mixing_margin_min=("mixing_margin", "min"),
        order_mean=("order", "mean"),
        order_max=("order", "max"),
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
        order_above_null_fraction=("order_above_null_q95", "mean"),
    )


def _render_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = list(REPRESENTATIONS)
    x = np.arange(len(order))
    width = 0.36

    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    for offset, condition in ((-width / 2, "off"), (width / 2, "on")):
        local = summary.loc[summary["interaction"].eq(condition)].set_index(
            "representation"
        ).reindex(order)
        axis.bar(x + offset, local["mixing_mean"], width, label=f"interactions {condition}")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=22, ha="right")
    axis.set_ylabel("Mean summed mixing capacity")
    axis.set_title("Bilinear schedule mixing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "bilinear_mixing_on_off.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    on = summary.loc[summary["interaction"].eq("on")].set_index("representation").reindex(order)
    axis.bar(x - width / 2, on["minimum_delay5_mean"], width, label="delay-5 memory")
    axis.bar(x + width / 2, on["mixing_margin_mean"], width, label="mixing margin")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=22, ha="right")
    axis.set_title("Memory versus permutation-controlled mixing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "bilinear_memory_mixing_tradeoff.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.bar(x, on["early_balance_ratio_min"])
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=22, ha="right")
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Worst-seed channel balance")
    axis.set_title("Bilinear representation balance")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "bilinear_balance.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_bilinear_mixing_assay(
    *,
    results_root: Path,
    config: BivariateBilinearMixingConfig = BivariateBilinearMixingConfig(),
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
        raise ValueError("bilinear mixing assay requires the exact six-atom reservoir")

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
            "feature_bank": "occupation_pair_raw",
            "scientific_question": (
                "Can an order-reversed longitudinal/transverse schedule expose bilinear "
                "cross-channel terms that the balanced palindrome suppresses, and are any "
                "such terms interaction-dependent?"
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
        rng = np.random.default_rng(int(seed) + 1_414_213_562)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for interaction in ("off", "on"):
            scale = config.interaction_scale if interaction == "on" else 0.0
            raw_features: dict[str, np.ndarray] = {}

            if interaction == "on":
                probabilities, metadata = evolve_crossover_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    _palindromic_schedule(),
                    interaction_scale=config.interaction_scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
            else:
                probabilities, metadata = _evolve_interaction_off_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    drive_phase_rad=config.drive_phase_rad,
                )
            raw_features["palindrome"] = build_crossover_feature_banks(probabilities)[
                "occupation_pair_raw"
            ]
            np.savez_compressed(
                probability_dir / f"seed_{seed}__{interaction}__palindrome.npz",
                probabilities=probabilities,
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "interaction": interaction,
                    "schedule": "palindrome",
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )

            for schedule_name, routes in BILINEAR_SCHEDULES.items():
                probabilities, metadata = evolve_bilinear_schedule_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    routes,
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
                raw_features[schedule_name] = build_crossover_feature_banks(probabilities)[
                    "occupation_pair_raw"
                ]
                np.savez_compressed(
                    probability_dir / f"seed_{seed}__{interaction}__{schedule_name}.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "schedule": schedule_name,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            representations = build_bilinear_representations(raw_features)
            for representation, matrix in representations.items():
                task_metrics, diagnostics, candidates = fit_capacity_readout(
                    matrix,
                    targets,
                    capacity_config,
                )
                task_metrics.insert(0, "seed", int(seed))
                task_metrics.insert(1, "interaction", interaction)
                task_metrics.insert(2, "representation", representation)
                task_frames.append(task_metrics)
                candidates.to_csv(
                    selection_dir / f"seed_{seed}__{interaction}__{representation}.csv",
                    index=False,
                )
                observed_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "representation": representation,
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
    aggregate = _aggregate(seed_summary)

    off_mixing = aggregate.loc[aggregate["interaction"].eq("off")].set_index(
        "representation"
    )["mixing_mean"]
    aggregate["mixing_advantage_vs_off"] = aggregate.apply(
        lambda row: (
            float(row["mixing_mean"] - off_mixing.loc[row["representation"]])
            if row["interaction"] == "on"
            else 0.0
        ),
        axis=1,
    )
    plots = _render_plots(aggregate, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "bilinear_seed_summary.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(run_dir / "bilinear_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    on = aggregate.loc[
        aggregate["interaction"].eq("on")
        & ~aggregate["representation"].eq("palindrome_control")
    ].copy()
    mixing_pass = on.loc[
        on["both_channels_early_above_null_fraction"].eq(1.0)
        & on["minimum_delay5_above_null_fraction"].eq(1.0)
        & on["early_balance_ratio_min"].ge(0.80)
        & on["mixing_above_null_fraction"].eq(1.0)
        & on["mixing_margin_min"].gt(0.0)
        & on["mixing_advantage_vs_off"].gt(0.0)
    ].copy()
    strongest = on.sort_values(
        ["mixing_margin_mean", "minimum_delay5_mean", "minimum_early_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    promoted = None
    if not mixing_pass.empty:
        promoted = mixing_pass.sort_values(
            ["mixing_margin_mean", "minimum_delay5_mean"],
            ascending=[False, False],
        ).iloc[0]

    summary = {
        "status": "bivariate_bilinear_mixing_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "step_duration_us": float(config.step_duration_us),
        "interaction_scale": float(config.interaction_scale),
        "mixing_gate_passed": bool(promoted is not None),
        "promoted_representation": (
            str(promoted["representation"]) if promoted is not None else None
        ),
        "strongest_candidate_representation": str(strongest["representation"]),
        "strongest_candidate_mixing_margin_mean": float(strongest["mixing_margin_mean"]),
        "strongest_candidate_mixing_advantage_vs_off": float(
            strongest["mixing_advantage_vs_off"]
        ),
        "strongest_candidate_minimum_delay5_mean": float(
            strongest["minimum_delay5_mean"]
        ),
        "decision_rule": (
            "Promote only when a non-palindromic representation preserves balanced early "
            "and delay-5 memory in every seed, exceeds its paired mixing null in every seed, "
            "and produces more mixing with interactions on than off."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
