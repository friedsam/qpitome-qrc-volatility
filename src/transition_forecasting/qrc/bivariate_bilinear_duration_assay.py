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
    build_bilinear_representations,
    evolve_bilinear_schedule_probabilities,
)
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
from transition_forecasting.qrc.bivariate_crossover_interaction_assay import (
    _evolve_interaction_off_probabilities,
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
REPRESENTATIONS = (
    "palindrome_control",
    "reverse_mirror_concat",
    "commutator_contrast_concat",
)


@dataclass(frozen=True)
class BivariateBilinearDurationConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    seeds: tuple[int, ...] = (20260724, 20260725, 20260726)
    step_durations_us: tuple[float, ...] = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
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


def select_duration_representations(
    representations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = set(REPRESENTATIONS).difference(representations)
    if missing:
        raise ValueError(f"missing duration representations: {sorted(missing)}")
    return {name: np.asarray(representations[name], dtype=float) for name in REPRESENTATIONS}


def _annotate_nulls(observed: pd.DataFrame, null_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, current in observed.iterrows():
        local = null_frame.loc[
            null_frame["seed"].eq(current["seed"])
            & null_frame["interaction"].eq(current["interaction"])
            & null_frame["representation"].eq(current["representation"])
            & np.isclose(
                null_frame["step_duration_us"].to_numpy(dtype=float),
                float(current["step_duration_us"]),
            )
        ]
        if local.empty:
            raise RuntimeError("missing paired null rows for bilinear duration case")
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


def _paired_on_off(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mixing_sum", "minimum_delay5", "minimum_early", "order"]
    pivot = seed_summary.pivot_table(
        index=["seed", "step_duration_us", "representation"],
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
        ["interaction", "step_duration_us", "representation"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        minimum_early_mean=("minimum_early", "mean"),
        minimum_early_min=("minimum_early", "min"),
        early_balance_ratio_mean=("early_balance_ratio", "mean"),
        early_balance_ratio_min=("early_balance_ratio", "min"),
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
    paired_aggregate = paired.groupby(
        ["step_duration_us", "representation"], as_index=False
    ).agg(
        mixing_on_minus_off_mean=("mixing_sum_on_minus_off", "mean"),
        mixing_on_minus_off_min=("mixing_sum_on_minus_off", "min"),
        delay5_on_minus_off_mean=("minimum_delay5_on_minus_off", "mean"),
        order_on_minus_off_mean=("order_on_minus_off", "mean"),
    )
    return aggregate.merge(
        paired_aggregate,
        on=["step_duration_us", "representation"],
        how="left",
        validate="many_to_one",
    )


def _render_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    on = summary.loc[summary["interaction"].eq("on")].copy()

    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    for representation in REPRESENTATIONS:
        local = on.loc[on["representation"].eq(representation)].sort_values(
            "step_duration_us"
        )
        axis.plot(
            local["step_duration_us"],
            local["mixing_margin_mean"],
            marker="o",
            label=representation,
        )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Mean mixing margin versus paired null")
    axis.set_title("Bilinear mixing margin versus duration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "bilinear_mixing_margin_vs_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    for representation in REPRESENTATIONS:
        local = on.loc[on["representation"].eq(representation)].sort_values(
            "step_duration_us"
        )
        axis.plot(
            local["step_duration_us"],
            local["minimum_delay5_mean"],
            marker="o",
            label=representation,
        )
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Mean minimum-channel delay-5 capacity")
    axis.set_title("Bilinear delay-5 memory versus duration")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "bilinear_delay5_vs_duration.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    reverse = on.loc[on["representation"].eq("reverse_mirror_concat")].sort_values(
        "step_duration_us"
    )
    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    axis.plot(
        reverse["step_duration_us"],
        reverse["mixing_on_minus_off_mean"],
        marker="o",
        label="mean on-off mixing advantage",
    )
    axis.plot(
        reverse["step_duration_us"],
        reverse["mixing_on_minus_off_min"],
        marker="o",
        label="worst-seed on-off advantage",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Step duration (microseconds)")
    axis.set_ylabel("Interaction-on minus interaction-off mixing")
    axis.set_title("Reverse schedule interaction-assisted mixing")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "reverse_on_off_mixing_advantage.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_bilinear_duration_assay(
    *,
    results_root: Path,
    config: BivariateBilinearDurationConfig = BivariateBilinearDurationConfig(),
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
        step_duration_us=0.02,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.seeds[0],
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    base_reservoir.validate()
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("bilinear duration assay requires the exact six-atom reservoir")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "exchange_symmetric_pairs": True,
            "paired_across_durations": True,
            "paired_interaction_control": True,
            "feature_bank": "occupation_pair_raw",
            "scientific_question": (
                "Does a duration near the palindrome memory optimum make the reverse or "
                "commutator bilinear representation produce stable interaction-assisted "
                "mixing while retaining balanced delay-5 memory?"
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
        rng = np.random.default_rng(int(seed) + 2_718_281_828)
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
            duration_tag = str(float(duration)).replace(".", "p")

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
                    probability_dir
                    / f"seed_{seed}__duration_{duration_tag}__{interaction}__palindrome.npz",
                    probabilities=probabilities,
                )
                metadata_rows.append(
                    {
                        "seed": int(seed),
                        "step_duration_us": float(duration),
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
                    raw_features[schedule_name] = build_crossover_feature_banks(
                        probabilities
                    )["occupation_pair_raw"]
                    np.savez_compressed(
                        probability_dir
                        / f"seed_{seed}__duration_{duration_tag}__{interaction}__{schedule_name}.npz",
                        probabilities=probabilities,
                    )
                    metadata_rows.append(
                        {
                            "seed": int(seed),
                            "step_duration_us": float(duration),
                            "interaction": interaction,
                            "schedule": schedule_name,
                            "metadata_json": json.dumps(metadata, sort_keys=True),
                        }
                    )

                representations = select_duration_representations(
                    build_bilinear_representations(raw_features)
                )
                for representation, matrix in representations.items():
                    task_metrics, diagnostics, candidates = fit_capacity_readout(
                        matrix,
                        targets,
                        capacity_config,
                    )
                    task_metrics.insert(0, "seed", int(seed))
                    task_metrics.insert(1, "step_duration_us", float(duration))
                    task_metrics.insert(2, "interaction", interaction)
                    task_metrics.insert(3, "representation", representation)
                    task_frames.append(task_metrics)
                    candidates.to_csv(
                        selection_dir
                        / f"seed_{seed}__duration_{duration_tag}__{interaction}__{representation}.csv",
                        index=False,
                    )
                    observed_rows.append(
                        {
                            "seed": int(seed),
                            "step_duration_us": float(duration),
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
                                "step_duration_us": float(duration),
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
    plots = _render_plots(aggregate, run_dir / "plots")

    task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
    observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
    seed_summary.to_csv(run_dir / "bilinear_duration_seed_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_on_off_seed_differences.csv", index=False)
    null_frame.to_csv(
        run_dir / "paired_permutation_null.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(run_dir / "bilinear_duration_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "simulation_metadata.csv", index=False)

    candidates = aggregate.loc[
        aggregate["interaction"].eq("on")
        & ~aggregate["representation"].eq("palindrome_control")
    ].copy()
    mixing_pass = candidates.loc[
        candidates["both_channels_early_above_null_fraction"].eq(1.0)
        & candidates["minimum_delay5_above_null_fraction"].eq(1.0)
        & candidates["early_balance_ratio_min"].ge(0.80)
        & candidates["mixing_above_null_fraction"].eq(1.0)
        & candidates["mixing_margin_min"].gt(0.0)
        & candidates["mixing_on_minus_off_min"].gt(0.0)
    ].copy()
    strongest = candidates.sort_values(
        [
            "mixing_margin_mean",
            "mixing_on_minus_off_min",
            "minimum_delay5_mean",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    promoted = None
    if not mixing_pass.empty:
        promoted = mixing_pass.sort_values(
            ["mixing_margin_mean", "minimum_delay5_mean"],
            ascending=[False, False],
        ).iloc[0]

    summary = {
        "status": "bivariate_bilinear_duration_complete",
        "seeds_completed": int(seed_summary["seed"].nunique()),
        "durations_completed": int(seed_summary["step_duration_us"].nunique()),
        "representations_completed": int(seed_summary["representation"].nunique()),
        "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
        "interaction_scale": float(config.interaction_scale),
        "mixing_gate_passed": bool(promoted is not None),
        "promoted_duration_us": (
            float(promoted["step_duration_us"]) if promoted is not None else None
        ),
        "promoted_representation": (
            str(promoted["representation"]) if promoted is not None else None
        ),
        "strongest_candidate_duration_us": float(strongest["step_duration_us"]),
        "strongest_candidate_representation": str(strongest["representation"]),
        "strongest_candidate_mixing_margin_mean": float(
            strongest["mixing_margin_mean"]
        ),
        "strongest_candidate_worst_seed_on_off_advantage": float(
            strongest["mixing_on_minus_off_min"]
        ),
        "strongest_candidate_minimum_delay5_mean": float(
            strongest["minimum_delay5_mean"]
        ),
        "decision_rule": (
            "Promote only when a non-palindromic duration preserves balanced early and "
            "delay-5 memory in every seed, exceeds its paired mixing null in every seed, "
            "and has positive interaction-on minus interaction-off mixing in every seed."
        ),
        "plots": plots,
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
