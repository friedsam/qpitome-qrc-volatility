from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    build_capacity_targets,
    capacity_summary,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

PALINDROMIC_SCHEDULE_NAMES = (
    "crossover_Ahalf_B_Ahalf",
    "crossover_Bhalf_A_Bhalf",
)
REPRESENTATIONS = (
    "palindrome_A",
    "palindrome_B",
    "concatenated_mirrors",
)


@dataclass(frozen=True)
class BivariateCrossoverStabilityConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
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
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not np.isfinite(self.drive_phase_rad):
            raise ValueError("drive_phase_rad must be finite")
        if self.permutations < 0:
            raise ValueError("permutations cannot be negative")

        train_end = int(np.floor(self.samples * 0.60))
        validation_end = train_end + int(np.floor(self.samples * 0.20))
        if train_end % 2 or validation_end % 2:
            raise ValueError(
                "train/validation/test boundaries must preserve complete exchange pairs"
            )

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


def generate_exchange_symmetric_windows(
    config: BivariateCrossoverStabilityConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Return interleaved ``(u1,u2)`` and ``(u2,u1)`` window pairs."""

    config.validate()
    rng = np.random.default_rng(int(seed))
    base = rng.uniform(
        -1.0,
        1.0,
        size=(config.samples // 2, config.sequence_length, 2),
    )
    windows = np.empty(
        (config.samples, config.sequence_length, 2), dtype=float
    )
    windows[0::2] = base
    windows[1::2] = base[:, :, ::-1]
    if not np.allclose(windows[1::2], windows[0::2, :, ::-1]):
        raise RuntimeError("exchange-paired synthetic construction failed")
    return windows


def paired_permutation_orders(
    samples: int,
    permutations: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Permute complete exchange pairs and optionally swap members within each pair."""

    if samples % 2:
        raise ValueError("paired permutations require an even sample count")
    pair_count = samples // 2
    outputs: list[np.ndarray] = []
    for _ in range(int(permutations)):
        pair_order = rng.permutation(pair_count)
        swap = rng.integers(0, 2, size=pair_count, dtype=np.int8)
        order = np.empty(samples, dtype=int)
        order[0::2] = 2 * pair_order + swap
        order[1::2] = 2 * pair_order + (1 - swap)
        if not np.array_equal(np.sort(order), np.arange(samples)):
            raise RuntimeError("paired null order is not a permutation")
        outputs.append(order)
    return tuple(outputs)


def _schedule(name: str):
    return next(schedule for schedule in CROSSOVER_SCHEDULES if schedule.name == name)


def build_stability_representations(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> dict[str, np.ndarray]:
    left = np.asarray(features_a, dtype=float)
    right = np.asarray(features_b, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("mirrored feature matrices must have equal two-dimensional shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("mirrored feature matrices must be finite")
    return {
        "palindrome_A": left,
        "palindrome_B": right,
        "concatenated_mirrors": np.concatenate([left, right], axis=1),
    }


def _metric_payload(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    payload: dict[str, float] = {}
    for channel in (1, 2):
        local = memory.loc[memory["channel"].eq(channel)]
        early = local.loc[local["delay_a"].le(5)]
        delay_five = local.loc[local["delay_a"].eq(5)]
        payload[f"channel{channel}_memory_total"] = float(local["capacity"].sum())
        payload[f"channel{channel}_early"] = float(early["capacity"].sum())
        payload[f"channel{channel}_delay5"] = float(delay_five["capacity"].mean())
    summary = capacity_summary(task_metrics)
    first = payload["channel1_early"]
    second = payload["channel2_early"]
    payload.update(
        {
            "minimum_early": float(min(first, second)),
            "early_balance_ratio": float(min(first, second) / (max(first, second) + 1e-12)),
            "early_imbalance": float(abs(first - second) / (abs(first) + abs(second) + 1e-12)),
            "minimum_delay5": float(
                min(payload["channel1_delay5"], payload["channel2_delay5"])
            ),
            "mixing_sum": float(summary["mixing_capacity"]),
            "order": float(summary["order_capacity"]),
        }
    )
    return payload


def _annotate_seed_nulls(
    observed_seed: pd.DataFrame,
    null_frame: pd.DataFrame,
) -> pd.DataFrame:
    if null_frame.empty:
        return observed_seed.copy()
    rows: list[dict[str, object]] = []
    for _, current in observed_seed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["representation"].eq(current["representation"])
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
            observed = float(current[metric])
            payload[f"{metric}_null_q95"] = float(np.quantile(values, 0.95))
            payload[f"{metric}_permutation_p_ge"] = float(
                (1 + np.sum(values >= observed)) / (1 + len(values))
            )
            payload[f"{metric}_above_null_q95"] = bool(
                observed > payload[f"{metric}_null_q95"]
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _aggregate(seed_summary: pd.DataFrame) -> pd.DataFrame:
    return seed_summary.groupby("representation", as_index=False).agg(
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
        mixing_mean=("mixing_sum", "mean"),
        mixing_max=("mixing_sum", "max"),
        order_mean=("order", "mean"),
        both_channels_early_above_null_fraction=(
            "both_channels_early_above_null",
            "mean",
        ),
        minimum_early_above_null_fraction=("minimum_early_above_null_q95", "mean"),
        mixing_above_null_fraction=("mixing_sum_above_null_q95", "mean"),
    )


def _render_plots(
    seed_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = list(REPRESENTATIONS)
    seeds = sorted(seed_summary["seed"].unique())

    figure, axes = plt.subplots(1, len(order), figsize=(15.0, 5.2), sharey=True)
    for axis, representation in zip(axes, order, strict=True):
        local = seed_summary.loc[
            seed_summary["representation"].eq(representation)
        ].set_index("seed").reindex(seeds)
        x = np.arange(len(seeds))
        width = 0.35
        axis.bar(x - width / 2, local["channel1_early"], width, label="channel 1")
        axis.bar(x + width / 2, local["channel2_early"], width, label="channel 2")
        axis.set_xticks(x)
        axis.set_xticklabels([str(seed) for seed in seeds], rotation=25)
        axis.set_title(representation)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Summed capacity, delays 1-5")
    axes[-1].legend()
    figure.suptitle("Exchange-paired crossover memory by seed")
    figure.tight_layout()
    filename = "paired_channel_memory_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    local = aggregate.set_index("representation").reindex(order).reset_index()
    x = np.arange(len(local))
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.bar(x, local["early_balance_ratio_min"])
    axis.set_xticks(x)
    axis.set_xticklabels(local["representation"], rotation=20, ha="right")
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Worst-seed min(C1,C2)/max(C1,C2)")
    axis.set_title("Exchange-paired within-seed balance")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "paired_worst_seed_balance.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.bar(x, local["both_channels_early_above_null_fraction"])
    axis.set_xticks(x)
    axis.set_xticklabels(local["representation"], rotation=20, ha="right")
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Fraction of seeds")
    axis.set_title("Both channels exceed their paired-permutation floors")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "paired_both_channels_above_null.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.bar(x, local["mixing_mean"])
    axis.set_xticks(x)
    axis.set_xticklabels(local["representation"], rotation=20, ha="right")
    axis.set_ylabel("Mean mixing capacity")
    axis.set_title("Exchange-paired mixing capacity")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "paired_mixing_capacity.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_crossover_stability_assay(
    *,
    results_root: Path,
    config: BivariateCrossoverStabilityConfig = BivariateCrossoverStabilityConfig(),
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
    if reservoir_config.n_atoms != 6 or reservoir_config.shots is not None:
        raise ValueError("stability assay requires an exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": reservoir_config.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "pair_preserving_splits": True,
            "primary_feature_bank": "occupation_pair_raw",
            "scientific_question": (
                "Does the palindromic crossover retain balanced two-channel memory when "
                "finite-sample asymmetry is removed, and does concatenating mirrored "
                "schedules stabilize memory or reveal mixing?"
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
        capacity_config = config.capacity_config(seed)
        windows = generate_exchange_symmetric_windows(config, seed=int(seed))
        targets = build_capacity_targets(windows, capacity_config)
        rng = np.random.default_rng(int(seed) + 271_828_183)
        null_orders = paired_permutation_orders(
            config.samples,
            config.permutations,
            rng=rng,
        )

        raw_features: dict[str, np.ndarray] = {}
        for schedule_name, representation in zip(
            PALINDROMIC_SCHEDULE_NAMES,
            ("palindrome_A", "palindrome_B"),
            strict=True,
        ):
            probabilities, metadata = evolve_crossover_probabilities(
                windows,
                reservoir_config,
                geometry_config,
                _schedule(schedule_name),
                interaction_scale=config.interaction_scale,
                drive_phase_rad=config.drive_phase_rad,
            )
            np.savez_compressed(
                probability_dir / f"seed_{seed}__{representation}.npz",
                probabilities=probabilities,
            )
            metadata_rows.append(
                {
                    "seed": int(seed),
                    "representation": representation,
                    "schedule": schedule_name,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )
            raw_features[representation] = build_crossover_feature_banks(
                probabilities
            )["occupation_pair_raw"]

        representations = build_stability_representations(
            raw_features["palindrome_A"],
            raw_features["palindrome_B"],
        )
        for representation, matrix in representations.items():
            task_metrics, diagnostics, candidates = fit_capacity_readout(
                matrix,
                targets,
                capacity_config,
            )
            task_metrics.insert(0, "seed", int(seed))
            task_metrics.insert(1, "representation", representation)
            task_frames.append(task_metrics)
            candidates.to_csv(
                selection_dir / f"seed_{seed}__{representation}.csv",
                index=False,
            )
            observed_rows.append(
                {
                    "seed": int(seed),
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
                        "representation": representation,
                        "permutation": int(permutation),
                        **_metric_payload(null_metrics),
                    }
                )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    observed_seed = pd.DataFrame(observed_rows)
    null_frame = pd.DataFrame(null_rows)
    seed_summary = _annotate_seed_nulls(observed_seed, null_frame)
    seed_summary["both_channels_early_above_null"] = (
        seed_summary["channel1_early_above_null_q95"]
        & seed_summary["channel2_early_above_null_q95"]
    )
    aggregate = _aggregate(seed_summary)
    plots = _render_plots(seed_summary, aggregate, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed_seed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "seed_stability_summary.csv", index=False)
    null_frame.to_csv(run_dir / "paired_permutation_null.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(run_dir / "stability_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    strongest = aggregate.sort_values(
        [
            "both_channels_early_above_null_fraction",
            "early_balance_ratio_min",
            "minimum_early_mean",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    concatenated = aggregate.loc[
        aggregate["representation"].eq("concatenated_mirrors")
    ].iloc[0]
    summary = {
        "status": "bivariate_crossover_stability_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "samples_per_seed": int(config.samples),
        "exchange_pairs_per_seed": int(config.samples // 2),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "strongest_stability_representation": str(strongest["representation"]),
        "strongest_worst_seed_balance_ratio": float(
            strongest["early_balance_ratio_min"]
        ),
        "strongest_both_channels_above_null_fraction": float(
            strongest["both_channels_early_above_null_fraction"]
        ),
        "concatenated_worst_seed_balance_ratio": float(
            concatenated["early_balance_ratio_min"]
        ),
        "concatenated_minimum_early_memory_mean": float(
            concatenated["minimum_early_mean"]
        ),
        "concatenated_mixing_mean": float(concatenated["mixing_mean"]),
        "concatenated_mixing_above_null_fraction": float(
            concatenated["mixing_above_null_fraction"]
        ),
        "duration_sweep_rule": (
            "Proceed with the palindromic or concatenated representation only when "
            "both channels exceed paired-permutation memory floors across seeds and "
            "the worst-seed balance ratio is materially improved. Mixing may remain "
            "below null before the first bounded duration sweep."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
