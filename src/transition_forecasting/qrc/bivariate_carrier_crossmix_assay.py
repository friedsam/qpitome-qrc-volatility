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
    evolve_bilinear_schedule_probabilities,
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

ROOT_TWO_INV = float(1.0 / np.sqrt(2.0))
CARRIER_MASKS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "identity": ((1.0, 0.0), (0.0, 1.0)),
    "identity_channel_swap": ((0.0, 1.0), (1.0, 0.0)),
    "hadamard": (
        (ROOT_TWO_INV, ROOT_TWO_INV),
        (ROOT_TWO_INV, -ROOT_TWO_INV),
    ),
    "hadamard_channel_swap": (
        (ROOT_TWO_INV, ROOT_TWO_INV),
        (-ROOT_TWO_INV, ROOT_TWO_INV),
    ),
}
REPRESENTATIONS = (
    "pure_slot_all_orders",
    "carrier_identity_pair",
    "carrier_hadamard_pair",
    "carrier_all_masks",
)
SAME_LAG_TASK = "mix_same_d1"


@dataclass(frozen=True)
class BivariateCarrierCrossmixConfig:
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


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mask, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("carrier mask must be a finite 2x2 matrix")
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("carrier mask must have full rank")
    return matrix


def carrier_drive(
    windows: np.ndarray,
    step: int,
    mask: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if not 0 <= step < values.shape[1]:
        raise IndexError("step lies outside the input window")
    matrix = _validate_mask(mask)
    mixed = values[:, step, :] @ matrix.T
    delta_mix = mixed[:, 0]
    amplitude_mix = mixed[:, 1]
    omega = reservoir.omega_base_rad_us * (
        1.0 + reservoir.omega_mod_fraction * amplitude_mix
    )
    if np.any(omega <= 0):
        raise ValueError("carrier-assisted amplitude encoding became nonpositive")
    delta = reservoir.delta_center_rad_us + reservoir.delta_span_rad_us * delta_mix
    phase = np.full(len(values), float(phase_rad), dtype=float)
    return omega, delta, phase


def evolve_carrier_mask_probabilities(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    mask: np.ndarray,
    *,
    mask_name: str,
    interaction_scale: float,
    drive_phase_rad: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve one full-step carrier waveform with simultaneous amplitude and detuning."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    reservoir.validate()
    geometry.validate()
    matrix = _validate_mask(mask)
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("carrier cross-mixing assay requires the exact six-atom reservoir")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")

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
        omega, delta, phase = carrier_drive(
            values,
            step,
            matrix,
            reservoir,
            drive_phase_rad,
        )
        states = _evolve_segment_batch(
            states,
            omega,
            delta,
            phase,
            reservoir.step_duration_us,
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
        "encoding": "carrier_assisted_cross_mixed_global",
        "mask_name": str(mask_name),
        "mask": matrix.tolist(),
        "interaction_scale": float(interaction_scale),
        "interactions_enabled": bool(interactions),
        "step_duration_us": float(reservoir.step_duration_us),
        "omega_carrier_rad_us": float(reservoir.omega_base_rad_us),
        "delta_carrier_rad_us": float(reservoir.delta_center_rad_us),
        "omega_mod_fraction": float(reservoir.omega_mod_fraction),
        "delta_span_rad_us": float(reservoir.delta_span_rad_us),
        "drive_phase_rad": float(drive_phase_rad),
        "probe_steps": [int(value) for value in probe_steps],
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
    }


def _validate_feature_banks(raw: dict[str, np.ndarray], required: set[str]) -> dict[str, np.ndarray]:
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"missing feature matrices: {sorted(missing)}")
    matrices = {name: np.asarray(raw[name], dtype=float) for name in required}
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise ValueError("feature matrices must be aligned")
    if next(iter(matrices.values())).shape[1] != 63:
        raise ValueError("each source bank must contain 63 occupation/pair features")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("feature matrices must be finite")
    return matrices


def build_architecture_representations(
    pure_slot_features: dict[str, np.ndarray],
    carrier_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pure = _validate_feature_banks(pure_slot_features, set(BILINEAR_SCHEDULES))
    carrier = _validate_feature_banks(carrier_features, set(CARRIER_MASKS))
    pure_all = np.concatenate([pure[name] for name in BILINEAR_SCHEDULES], axis=1)
    identity = np.concatenate(
        [carrier["identity"], carrier["identity_channel_swap"]], axis=1
    )
    hadamard = np.concatenate(
        [carrier["hadamard"], carrier["hadamard_channel_swap"]], axis=1
    )
    return {
        "pure_slot_all_orders": pure_all,
        "carrier_identity_pair": identity,
        "carrier_hadamard_pair": hadamard,
        "carrier_all_masks": np.concatenate([identity, hadamard], axis=1),
    }


def _mixing_decomposition(task_metrics: pd.DataFrame) -> dict[str, float]:
    mixing = task_metrics.loc[task_metrics["group"].eq("mixing")]
    if mixing.empty or SAME_LAG_TASK not in set(mixing["task"]):
        raise RuntimeError("capacity output is missing the same-lag mixing task")
    same = float(mixing.loc[mixing["task"].eq(SAME_LAG_TASK), "capacity"].sum())
    delayed = float(mixing.loc[~mixing["task"].eq(SAME_LAG_TASK), "capacity"].sum())
    return {"same_lag_mixing": same, "delayed_mixing": delayed}


def _task_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    return {**_metric_payload(task_metrics), **_mixing_decomposition(task_metrics)}


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
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
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & null_frame["representation"].eq(current["representation"])
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for carrier cross-mixing case")
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


def _carrier_vs_pure(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "minimum_early",
        "minimum_delay5",
        "mixing_sum",
        "same_lag_mixing",
        "delayed_mixing",
        "order",
    ]
    on = seed_summary.loc[seed_summary["interaction"].eq("on")]
    pure = on.loc[on["representation"].eq("pure_slot_all_orders"), ["seed", *metrics]].copy()
    carrier = on.loc[on["representation"].eq("carrier_all_masks"), ["seed", *metrics]].copy()
    pure = pure.rename(columns={metric: f"{metric}_pure" for metric in metrics})
    carrier = carrier.rename(columns={metric: f"{metric}_carrier" for metric in metrics})
    merged = carrier.merge(pure, on="seed", validate="one_to_one")
    for metric in metrics:
        merged[f"{metric}_carrier_minus_pure"] = (
            merged[f"{metric}_carrier"] - merged[f"{metric}_pure"]
        )
    return merged


def _aggregate(seed_summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    aggregate = seed_summary.groupby(["interaction", "representation"], as_index=False).agg(
        seeds=("seed", "nunique"),
        feature_count=("feature_count", "first"),
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
    paired_aggregate = paired.groupby("representation", as_index=False).agg(
        same_lag_on_minus_off_mean=("same_lag_mixing_on_minus_off", "mean"),
        same_lag_on_minus_off_min=("same_lag_mixing_on_minus_off", "min"),
        delayed_on_minus_off_mean=("delayed_mixing_on_minus_off", "mean"),
        delayed_on_minus_off_min=("delayed_mixing_on_minus_off", "min"),
        delay5_on_minus_off_mean=("minimum_delay5_on_minus_off", "mean"),
    )
    return aggregate.merge(
        paired_aggregate,
        on="representation",
        how="left",
        validate="many_to_one",
    )


def _render_plots(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    carrier_vs_pure: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = list(REPRESENTATIONS)
    on = aggregate.loc[aggregate["interaction"].eq("on")].set_index("representation").reindex(order)
    x = np.arange(len(order))
    width = 0.36

    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    axis.bar(x - width / 2, on["same_lag_margin_mean"], width, label="same-lag margin")
    axis.bar(x + width / 2, on["delayed_margin_mean"], width, label="delayed margin")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=20, ha="right")
    axis.set_ylabel("Mean capacity margin versus paired null")
    axis.set_title("Pure-slot and carrier-assisted mixing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "carrier_mixing_margins.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.8, 5.8))
    for representation in order:
        row = on.loc[representation]
        axis.scatter(
            float(row["minimum_delay5_mean"]),
            float(row["same_lag_margin_mean"]),
            s=75,
            label=representation,
        )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Mean minimum-channel delay-5 capacity")
    axis.set_ylabel("Mean same-lag mixing margin")
    axis.set_title("Memory–mixing operating points")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "carrier_memory_mixing_operating_points.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    pair = paired.loc[paired["representation"].eq("carrier_all_masks")].sort_values("seed")
    seeds = pair["seed"].astype(str).tolist()
    x_seed = np.arange(len(pair))
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.bar(x_seed - width / 2, pair["same_lag_mixing_on_minus_off"], width, label="same-lag")
    axis.bar(x_seed + width / 2, pair["delayed_mixing_on_minus_off"], width, label="delayed")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_seed)
    axis.set_xticklabels(seeds)
    axis.set_ylabel("Interactions on minus off")
    axis.set_title("Carrier interaction contribution by seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "carrier_interaction_advantage_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    comparison = carrier_vs_pure.sort_values("seed")
    x_seed = np.arange(len(comparison))
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.bar(
        x_seed - width / 2,
        comparison["same_lag_mixing_carrier_minus_pure"],
        width,
        label="same-lag",
    )
    axis.bar(
        x_seed + width / 2,
        comparison["delayed_mixing_carrier_minus_pure"],
        width,
        label="delayed",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_seed)
    axis.set_xticklabels(comparison["seed"].astype(str))
    axis.set_ylabel("Carrier all masks minus pure-slot all orders")
    axis.set_title("Paired carrier architecture gain by seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "carrier_vs_pure_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_carrier_crossmix_assay(
    *,
    results_root: Path,
    config: BivariateCarrierCrossmixConfig = BivariateCarrierCrossmixConfig(),
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
        raise ValueError("carrier cross-mixing assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_architecture_comparison": True,
            "paired_interaction_control": True,
            "pure_slot_programs": list(BILINEAR_SCHEDULES),
            "carrier_masks": {name: [list(row) for row in mask] for name, mask in CARRIER_MASKS.items()},
            "representations": list(REPRESENTATIONS),
            "scientific_question": (
                "Does continuous coexistence of global amplitude and detuning, combined with "
                "fixed multivariate input masks, convert the pure-slot near-pass into robust "
                "same-lag or delayed cross-channel mixing while retaining balanced delay-5 memory?"
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
        rng = np.random.default_rng(int(seed) + 1_618_033_989)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )
        reservoir_config = replace(base_reservoir, shot_seed=int(seed))

        for interaction in ("off", "on"):
            scale = config.interaction_scale if interaction == "on" else 0.0
            pure_features: dict[str, np.ndarray] = {}
            carrier_features: dict[str, np.ndarray] = {}

            for schedule_name, routes in BILINEAR_SCHEDULES.items():
                probabilities, metadata = evolve_bilinear_schedule_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    routes,
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
                pure_features[schedule_name] = build_crossover_feature_banks(probabilities)[
                    "occupation_pair_raw"
                ]
                np.savez_compressed(
                    probability_dir
                    / f"seed_{seed}__{interaction}__pure__{schedule_name}.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "architecture": "pure_slot",
                        "program": schedule_name,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            for mask_name, mask in CARRIER_MASKS.items():
                probabilities, metadata = evolve_carrier_mask_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    np.asarray(mask, dtype=float),
                    mask_name=mask_name,
                    interaction_scale=scale,
                    drive_phase_rad=config.drive_phase_rad,
                )
                carrier_features[mask_name] = build_crossover_feature_banks(probabilities)[
                    "occupation_pair_raw"
                ]
                np.savez_compressed(
                    probability_dir
                    / f"seed_{seed}__{interaction}__carrier__{mask_name}.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "interaction": interaction,
                        "architecture": "carrier_crossmix",
                        "program": mask_name,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )

            representations = build_architecture_representations(
                pure_features,
                carrier_features,
            )
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
                        **_task_payload(metrics),
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
                            **_task_payload(null_metrics),
                        }
                    )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    null_frame = pd.DataFrame(null_rows)
    seed_summary = _annotate_nulls(observed, null_frame)
    paired = _paired_on_off(seed_summary)
    carrier_vs_pure = _carrier_vs_pure(seed_summary)
    aggregate = _aggregate(seed_summary, paired)
    plots = _render_plots(aggregate, paired, carrier_vs_pure, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "carrier_crossmix_seed_summary.csv", index=False)
    aggregate.to_csv(run_dir / "carrier_crossmix_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
    carrier_vs_pure.to_csv(run_dir / "carrier_vs_pure_differences.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    on = aggregate.loc[
        aggregate["interaction"].eq("on")
        & aggregate["representation"].str.startswith("carrier_")
    ].copy()
    viable = on.loc[
        on["both_channels_early_above_null_fraction"].eq(1.0)
        & on["minimum_delay5_above_null_fraction"].eq(1.0)
        & on["early_balance_ratio_min"].ge(0.80)
        & on["same_lag_above_null_fraction"].eq(1.0)
        & on["same_lag_margin_min"].gt(0.0)
        & on["same_lag_on_minus_off_min"].gt(0.0)
    ].copy()
    delayed = viable.loc[
        viable["delayed_above_null_fraction"].eq(1.0)
        & viable["delayed_margin_min"].gt(0.0)
        & viable["delayed_on_minus_off_min"].gt(0.0)
    ].copy()

    promoted = None
    if not delayed.empty:
        promoted = delayed.sort_values(
            ["delayed_margin_mean", "same_lag_margin_mean", "minimum_delay5_mean"],
            ascending=[False, False, False],
        ).iloc[0]
    elif not viable.empty:
        promoted = viable.sort_values(
            ["same_lag_margin_mean", "minimum_delay5_mean", "delayed_margin_mean"],
            ascending=[False, False, False],
        ).iloc[0]
    strongest = on.sort_values(
        ["delayed_margin_mean", "same_lag_margin_mean", "minimum_delay5_mean"],
        ascending=[False, False, False],
    ).iloc[0]

    comparison = carrier_vs_pure
    summary = {
        "status": "bivariate_carrier_crossmix_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "simulations_executed": int(
            len(config.seeds) * 2 * (len(BILINEAR_SCHEDULES) + len(CARRIER_MASKS))
        ),
        "step_duration_us": float(config.step_duration_us),
        "interaction_scale": float(config.interaction_scale),
        "carrier_viability_gate_passed": bool(not viable.empty),
        "delayed_mixing_gate_passed": bool(not delayed.empty),
        "promoted_representation": (
            str(promoted["representation"]) if promoted is not None else None
        ),
        "strongest_carrier_representation": str(strongest["representation"]),
        "strongest_carrier_same_lag_margin_mean": float(strongest["same_lag_margin_mean"]),
        "strongest_carrier_delayed_margin_mean": float(strongest["delayed_margin_mean"]),
        "strongest_carrier_minimum_delay5_mean": float(strongest["minimum_delay5_mean"]),
        "carrier_all_masks_same_lag_gain_over_pure_all_seeds": bool(
            comparison["same_lag_mixing_carrier_minus_pure"].gt(0.0).all()
        ),
        "carrier_all_masks_delayed_gain_over_pure_all_seeds": bool(
            comparison["delayed_mixing_carrier_minus_pure"].gt(0.0).all()
        ),
        "decision_rule": (
            "A carrier representation is viable only if both-channel early and delay-5 memory, "
            "same-lag mixing above paired null, and positive interaction assistance hold in every "
            "seed. Delayed mixing must independently exceed its paired null and interaction-off "
            "control in every seed. The equal-width carrier-all-masks map is compared directly "
            "against the pure-slot all-orders baseline within each seed."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
