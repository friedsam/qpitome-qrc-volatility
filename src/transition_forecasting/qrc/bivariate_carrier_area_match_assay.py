from __future__ import annotations

import json
import tempfile
import zipfile
from contextlib import contextmanager
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
)
from transition_forecasting.qrc.bivariate_capacity_assay import (
    SyntheticTargets,
    build_capacity_targets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    BivariateCarrierCrossmixConfig,
    _aggregate,
    _annotate_nulls,
    _paired_on_off,
    _task_payload,
    evolve_carrier_mask_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    generate_exchange_symmetric_windows,
    paired_permutation_orders,
)
from transition_forecasting.qrc.temporal_rydberg_chain import TemporalRydbergChainConfig
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

AREA_MATCH_SCALE = 0.5
REPRESENTATIONS = (
    "pure_slot_all_orders",
    "carrier_full_strength_all_masks",
    "carrier_area_identity_pair",
    "carrier_area_hadamard_pair",
    "carrier_area_all_masks",
)


@dataclass(frozen=True)
class BivariateCarrierAreaMatchConfig:
    seeds: tuple[int, ...] | None = None
    permutations: int | None = None
    field_scale: float = AREA_MATCH_SCALE

    def validate(self) -> None:
        if self.seeds is not None and (
            not self.seeds or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be nonempty and unique when supplied")
        if self.permutations is not None and self.permutations < 1:
            raise ValueError("permutations must be positive when supplied")
        if not np.isclose(self.field_scale, AREA_MATCH_SCALE, atol=0.0, rtol=0.0):
            raise ValueError(
                "the causal area-match assay fixes field_scale at exactly 0.5"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def materialize_carrier_source_run(source: Path):
    path = Path(source)
    if path.is_dir():
        yield _resolve_carrier_run_root(path)
        return
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"source run must be a directory or ZIP archive: {path}")
    with tempfile.TemporaryDirectory(prefix="carrier-area-match-") as temporary:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temporary)
        yield _resolve_carrier_run_root(Path(temporary))


def _resolve_carrier_run_root(root: Path) -> Path:
    matches: list[Path] = []
    for summary_path in sorted(Path(root).rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "bivariate_carrier_crossmix_complete":
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one completed carrier-crossmix run below {root}, found {len(matches)}"
        )
    return matches[0]


def _source_objects(
    run_root: Path,
) -> tuple[
    BivariateCarrierCrossmixConfig,
    TemporalRydbergChainConfig,
    StaggeredLadderGeometryConfig,
]:
    payload = json.loads((run_root / "params.json").read_text(encoding="utf-8"))
    parameters = payload["parameters"]

    config_values = dict(parameters["config"])
    for key in ("memory_delays", "alphas", "seeds"):
        config_values[key] = tuple(config_values[key])
    source_config = BivariateCarrierCrossmixConfig(**config_values)
    source_config.validate()

    reservoir_values = dict(parameters["reservoir"])
    reservoir_values["probe_fractions"] = tuple(reservoir_values["probe_fractions"])
    reservoir = TemporalRydbergChainConfig(**reservoir_values)
    reservoir.validate()

    geometry = StaggeredLadderGeometryConfig(**parameters["geometry"])
    geometry.validate()
    return source_config, reservoir, geometry


def area_matched_reservoir(
    reservoir: TemporalRydbergChainConfig,
    *,
    seed: int,
    field_scale: float = AREA_MATCH_SCALE,
) -> TemporalRydbergChainConfig:
    if not np.isclose(field_scale, AREA_MATCH_SCALE, atol=0.0, rtol=0.0):
        raise ValueError("area-matched reservoir requires field_scale=0.5")
    scaled = replace(
        reservoir,
        omega_base_rad_us=float(reservoir.omega_base_rad_us * field_scale),
        delta_center_rad_us=float(reservoir.delta_center_rad_us * field_scale),
        delta_span_rad_us=float(reservoir.delta_span_rad_us * field_scale),
        shot_seed=int(seed),
    )
    scaled.validate()
    return scaled


def _load_probabilities(
    run_root: Path,
    *,
    seed: int,
    interaction: str,
    architecture: str,
    program: str,
) -> np.ndarray:
    path = (
        run_root
        / "probabilities"
        / f"seed_{seed}__{interaction}__{architecture}__{program}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing saved source probabilities: {path}")
    with np.load(path) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"missing probabilities array in {path}")
        probabilities = np.asarray(archive["probabilities"], dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[2] != 64:
        raise ValueError(f"unexpected probability shape in {path}: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite probabilities in {path}")
    return probabilities


def _feature_bank(probabilities: np.ndarray) -> np.ndarray:
    return build_crossover_feature_banks(probabilities)["occupation_pair_raw"]


def build_area_match_representations(
    pure_slot_features: dict[str, np.ndarray],
    full_strength_carrier_features: dict[str, np.ndarray],
    area_matched_carrier_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    required_pure = set(BILINEAR_SCHEDULES)
    required_carrier = set(CARRIER_MASKS)
    for label, values, required in (
        ("pure", pure_slot_features, required_pure),
        ("full carrier", full_strength_carrier_features, required_carrier),
        ("area carrier", area_matched_carrier_features, required_carrier),
    ):
        missing = required.difference(values)
        if missing:
            raise ValueError(f"missing {label} feature banks: {sorted(missing)}")
        matrices = [np.asarray(values[name], dtype=float) for name in required]
        if len({matrix.shape for matrix in matrices}) != 1:
            raise ValueError(f"{label} feature banks must be aligned")
        if matrices[0].shape[1] != 63:
            raise ValueError(f"{label} source banks must each contain 63 features")
        if any(not np.isfinite(matrix).all() for matrix in matrices):
            raise ValueError(f"{label} feature banks must be finite")

    pure_all = np.concatenate(
        [pure_slot_features[name] for name in BILINEAR_SCHEDULES], axis=1
    )
    full_all = np.concatenate(
        [full_strength_carrier_features[name] for name in CARRIER_MASKS], axis=1
    )
    identity = np.concatenate(
        [
            area_matched_carrier_features["identity"],
            area_matched_carrier_features["identity_channel_swap"],
        ],
        axis=1,
    )
    hadamard = np.concatenate(
        [
            area_matched_carrier_features["hadamard"],
            area_matched_carrier_features["hadamard_channel_swap"],
        ],
        axis=1,
    )
    return {
        "pure_slot_all_orders": pure_all,
        "carrier_full_strength_all_masks": full_all,
        "carrier_area_identity_pair": identity,
        "carrier_area_hadamard_pair": hadamard,
        "carrier_area_all_masks": np.concatenate([identity, hadamard], axis=1),
    }


def _architecture_differences(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "minimum_early",
        "minimum_delay5",
        "mixing_sum",
        "same_lag_mixing",
        "delayed_mixing",
        "order",
    ]
    on = seed_summary.loc[seed_summary["interaction"].eq("on")]
    names = (
        "pure_slot_all_orders",
        "carrier_full_strength_all_masks",
        "carrier_area_all_masks",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        local = on.loc[on["representation"].eq(name), ["seed", *metrics]].copy()
        local = local.rename(columns={metric: f"{metric}__{name}" for metric in metrics})
        frames[name] = local
    merged = frames[names[0]]
    for name in names[1:]:
        merged = merged.merge(frames[name], on="seed", validate="one_to_one")
    for metric in metrics:
        merged[f"{metric}__area_minus_pure"] = (
            merged[f"{metric}__carrier_area_all_masks"]
            - merged[f"{metric}__pure_slot_all_orders"]
        )
        merged[f"{metric}__area_minus_full"] = (
            merged[f"{metric}__carrier_area_all_masks"]
            - merged[f"{metric}__carrier_full_strength_all_masks"]
        )
    return merged


def _render_plots(
    aggregate: pd.DataFrame,
    differences: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = list(REPRESENTATIONS)
    on = (
        aggregate.loc[aggregate["interaction"].eq("on")]
        .set_index("representation")
        .reindex(order)
    )
    x = np.arange(len(order))
    width = 0.36

    figure, axis = plt.subplots(figsize=(11.2, 5.8))
    axis.bar(x - width / 2, on["same_lag_margin_mean"], width, label="same-lag margin")
    axis.bar(x + width / 2, on["delayed_margin_mean"], width, label="delayed margin")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=22, ha="right")
    axis.set_ylabel("Mean capacity margin versus paired null")
    axis.set_title("Full-strength and area-matched carrier mixing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "area_match_mixing_margins.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(10.2, 5.8))
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
    axis.set_title("Area-matched carrier memory–mixing operating points")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "area_match_memory_mixing_operating_points.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    comparison = differences.sort_values("seed")
    x_seed = np.arange(len(comparison))
    figure, axis = plt.subplots(figsize=(9.8, 5.5))
    axis.bar(
        x_seed - width / 2,
        comparison["same_lag_mixing__area_minus_pure"],
        width,
        label="same-lag",
    )
    axis.bar(
        x_seed + width / 2,
        comparison["delayed_mixing__area_minus_pure"],
        width,
        label="delayed",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_seed)
    axis.set_xticklabels(comparison["seed"].astype(str))
    axis.set_ylabel("Area-matched carrier minus pure-slot capacity")
    axis.set_title("Equal-width architecture gain by seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "area_match_vs_pure_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    area_pair = paired.loc[
        paired["representation"].eq("carrier_area_all_masks")
    ].sort_values("seed")
    x_seed = np.arange(len(area_pair))
    figure, axis = plt.subplots(figsize=(9.8, 5.5))
    axis.bar(
        x_seed - width / 2,
        area_pair["same_lag_mixing_on_minus_off"],
        width,
        label="same-lag",
    )
    axis.bar(
        x_seed + width / 2,
        area_pair["delayed_mixing_on_minus_off"],
        width,
        label="delayed",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_seed)
    axis.set_xticklabels(area_pair["seed"].astype(str))
    axis.set_ylabel("Interactions on minus off")
    axis.set_title("Area-matched carrier interaction contribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "area_match_interaction_advantage_by_seed.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_carrier_area_match_assay(
    *,
    source_run: Path,
    results_root: Path,
    config: BivariateCarrierAreaMatchConfig = BivariateCarrierAreaMatchConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    with materialize_carrier_source_run(Path(source_run)) as source_root:
        source_config, source_reservoir, geometry = _source_objects(source_root)
        selected_seeds = (
            tuple(int(seed) for seed in config.seeds)
            if config.seeds is not None
            else tuple(int(seed) for seed in source_config.seeds)
        )
        unknown = set(selected_seeds).difference(source_config.seeds)
        if unknown:
            raise ValueError(f"requested seeds are absent from source run: {sorted(unknown)}")
        permutations = (
            int(config.permutations)
            if config.permutations is not None
            else int(source_config.permutations)
        )

        run_dir = begin_run(
            Path(results_root),
            {
                "config": config.to_dict(),
                "source_run": str(Path(source_run)),
                "source_config": source_config.to_dict(),
                "source_reservoir": source_reservoir.to_dict(),
                "geometry": geometry.to_dict(),
                "selected_seeds": list(selected_seeds),
                "resolved_permutations": permutations,
                "field_scale": AREA_MATCH_SCALE,
                "quantum_simulations_executed": int(
                    len(selected_seeds) * 2 * len(CARRIER_MASKS)
                ),
                "source_probabilities_reused": True,
                "first_order_area_match": {
                    "omega_base_rad_us": float(
                        source_reservoir.omega_base_rad_us * AREA_MATCH_SCALE
                    ),
                    "delta_center_rad_us": float(
                        source_reservoir.delta_center_rad_us * AREA_MATCH_SCALE
                    ),
                    "delta_span_rad_us": float(
                        source_reservoir.delta_span_rad_us * AREA_MATCH_SCALE
                    ),
                    "step_duration_us": float(source_reservoir.step_duration_us),
                    "interaction_scale_unchanged": float(source_config.interaction_scale),
                },
                "scientific_question": (
                    "When the simultaneous carrier fields are scaled to the first-order average "
                    "field area of the pure two-half-slot schedule, does continuous coexistence "
                    "and cross-masking produce robust same-lag or delayed mixing without losing "
                    "balanced delay-5 memory?"
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

        stability_config = source_config.stability_config()
        for seed in selected_seeds:
            capacity_config = stability_config.capacity_config(seed)
            windows = generate_exchange_symmetric_windows(stability_config, seed=seed)
            targets = build_capacity_targets(windows, capacity_config)
            rng = np.random.default_rng(int(seed) + 2_236_067_977)
            null_orders = paired_permutation_orders(
                source_config.samples,
                permutations,
                rng=rng,
            )
            scaled_reservoir = area_matched_reservoir(
                source_reservoir,
                seed=seed,
                field_scale=config.field_scale,
            )

            for interaction in ("off", "on"):
                scale = source_config.interaction_scale if interaction == "on" else 0.0
                pure_features: dict[str, np.ndarray] = {}
                full_features: dict[str, np.ndarray] = {}
                area_features: dict[str, np.ndarray] = {}

                for schedule_name in BILINEAR_SCHEDULES:
                    pure_features[schedule_name] = _feature_bank(
                        _load_probabilities(
                            source_root,
                            seed=seed,
                            interaction=interaction,
                            architecture="pure",
                            program=schedule_name,
                        )
                    )
                for mask_name in CARRIER_MASKS:
                    full_features[mask_name] = _feature_bank(
                        _load_probabilities(
                            source_root,
                            seed=seed,
                            interaction=interaction,
                            architecture="carrier",
                            program=mask_name,
                        )
                    )
                    probabilities, metadata = evolve_carrier_mask_probabilities(
                        windows,
                        scaled_reservoir,
                        geometry,
                        np.asarray(CARRIER_MASKS[mask_name], dtype=float),
                        mask_name=f"area_matched_{mask_name}",
                        interaction_scale=scale,
                        drive_phase_rad=source_config.drive_phase_rad,
                    )
                    area_features[mask_name] = _feature_bank(probabilities)
                    np.savez_compressed(
                        probability_dir
                        / f"seed_{seed}__{interaction}__area_carrier__{mask_name}.npz",
                        probabilities=probabilities,
                    )
                    metadata_rows.append(
                        {
                            "seed": int(seed),
                            "interaction": interaction,
                            "program": mask_name,
                            "metadata_json": json.dumps(metadata, sort_keys=True),
                        }
                    )

                representations = build_area_match_representations(
                    pure_features,
                    full_features,
                    area_features,
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
        aggregate = _aggregate(seed_summary, paired)
        differences = _architecture_differences(seed_summary)
        plots = _render_plots(aggregate, differences, paired, run_dir / "plots")

        task_metrics.to_csv(
            run_dir / "task_metrics.csv.gz", index=False, compression="gzip"
        )
        observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
        seed_summary.to_csv(run_dir / "carrier_area_match_seed_summary.csv", index=False)
        aggregate.to_csv(run_dir / "carrier_area_match_summary.csv", index=False)
        paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
        differences.to_csv(run_dir / "architecture_differences.csv", index=False)
        null_frame.to_csv(
            run_dir / "paired_permutation_null.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(metadata_rows).to_csv(
            run_dir / "simulation_metadata.csv", index=False
        )

        on = aggregate.loc[
            aggregate["interaction"].eq("on")
            & aggregate["representation"].str.startswith("carrier_area_")
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

        summary = {
            "status": "bivariate_carrier_area_match_complete",
            "source_status": "bivariate_carrier_crossmix_complete",
            "seeds_completed": int(seed_summary["seed"].nunique()),
            "representations_completed": int(seed_summary["representation"].nunique()),
            "interaction_conditions_completed": int(
                seed_summary["interaction"].nunique()
            ),
            "simulations_executed": int(
                len(selected_seeds) * 2 * len(CARRIER_MASKS)
            ),
            "field_scale": AREA_MATCH_SCALE,
            "carrier_viability_gate_passed": bool(not viable.empty),
            "delayed_mixing_gate_passed": bool(not delayed.empty),
            "promoted_representation": (
                str(promoted["representation"]) if promoted is not None else None
            ),
            "strongest_area_representation": str(strongest["representation"]),
            "strongest_area_same_lag_margin_mean": float(
                strongest["same_lag_margin_mean"]
            ),
            "strongest_area_delayed_margin_mean": float(
                strongest["delayed_margin_mean"]
            ),
            "strongest_area_minimum_delay5_mean": float(
                strongest["minimum_delay5_mean"]
            ),
            "decision_rule": (
                "The area-matched carrier is viable only if balanced early and delay-5 memory, "
                "same-lag mixing above paired null, and positive interaction assistance hold in "
                "every seed. Delayed mixing must independently exceed its paired null and "
                "interaction-off control in every seed."
            ),
            "plots": plots,
            "config": config.to_dict(),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return run_dir
