from __future__ import annotations

import json
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.bivariate_bilinear_mixing_assay import (
    BILINEAR_SCHEDULES,
    BivariateBilinearMixingConfig,
)
from transition_forecasting.qrc.bivariate_capacity_assay import (
    SyntheticTargets,
    build_capacity_targets,
    fit_capacity_readout,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.bivariate_crossover_stability_assay import (
    _metric_payload,
    generate_exchange_symmetric_windows,
    paired_permutation_orders,
)

REPRESENTATIONS = (
    "palindrome_control",
    "all_orders_joint",
    "palindrome_plus_all_orders",
)


@dataclass(frozen=True)
class BivariateAllOrdersReadoutConfig:
    permutations: int | None = None

    def validate(self) -> None:
        if self.permutations is not None and self.permutations < 1:
            raise ValueError("permutations must be positive when supplied")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def materialize_source_run(source: Path):
    path = Path(source)
    if path.is_dir():
        yield _resolve_run_root(path)
        return
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"source run must be a directory or ZIP archive: {path}")
    with tempfile.TemporaryDirectory(prefix="bilinear-all-orders-") as temporary:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temporary)
        yield _resolve_run_root(Path(temporary))


def _resolve_run_root(root: Path) -> Path:
    candidates = sorted(root.rglob("summary.json"))
    matches: list[Path] = []
    for summary_path in candidates:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "bivariate_bilinear_mixing_complete":
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one completed bilinear run below {root}, found {len(matches)}"
        )
    return matches[0]


def _source_config(run_root: Path) -> BivariateBilinearMixingConfig:
    payload = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    values = dict(payload["config"])
    for key in ("memory_delays", "alphas", "seeds"):
        values[key] = tuple(values[key])
    config = BivariateBilinearMixingConfig(**values)
    config.validate()
    return config


def _load_probabilities(run_root: Path, seed: int, interaction: str, schedule: str) -> np.ndarray:
    path = run_root / "probabilities" / f"seed_{seed}__{interaction}__{schedule}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing saved probabilities: {path}")
    with np.load(path) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"missing probabilities array in {path}")
        probabilities = np.asarray(archive["probabilities"], dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[2] != 64:
        raise ValueError(f"unexpected probability tensor shape in {path}: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite probabilities in {path}")
    return probabilities


def build_all_orders_representations(raw_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {"palindrome", *BILINEAR_SCHEDULES}
    missing = required.difference(raw_features)
    if missing:
        raise ValueError(f"missing feature matrices: {sorted(missing)}")
    matrices = {name: np.asarray(value, dtype=float) for name, value in raw_features.items()}
    shapes = {matrix.shape for matrix in matrices.values()}
    if len(shapes) != 1 or next(iter(shapes))[1] != 63:
        raise ValueError("all source matrices must be aligned 63-feature banks")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("source feature matrices must be finite")

    all_orders = np.concatenate(
        [
            matrices["d1_then_x2"],
            matrices["d2_then_x1"],
            matrices["x2_then_d1"],
            matrices["x1_then_d2"],
        ],
        axis=1,
    )
    return {
        "palindrome_control": matrices["palindrome"],
        "all_orders_joint": all_orders,
        "palindrome_plus_all_orders": np.concatenate(
            [matrices["palindrome"], all_orders], axis=1
        ),
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
            raise RuntimeError("missing paired null rows for all-orders readout")
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


def _paired_on_off(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["minimum_early", "minimum_delay5", "mixing_sum", "order"]
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


def _render_plots(aggregate: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(REPRESENTATIONS)
    x = np.arange(len(order))
    width = 0.36
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for offset, interaction in ((-width / 2, "off"), (width / 2, "on")):
        local = aggregate.loc[aggregate["interaction"].eq(interaction)].set_index(
            "representation"
        ).reindex(order)
        axis.bar(x + offset, local["mixing_mean"], width, label=f"interactions {interaction}")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=20, ha="right")
    axis.set_ylabel("Mean summed mixing capacity")
    axis.set_title("Joint readout mixing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "joint_readout_mixing_on_off.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    on = aggregate.loc[aggregate["interaction"].eq("on")].set_index(
        "representation"
    ).reindex(order)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.bar(x - width / 2, on["minimum_delay5_mean"], width, label="delay-5 memory")
    axis.bar(x + width / 2, on["mixing_margin_mean"], width, label="mixing margin")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=20, ha="right")
    axis.set_title("Joint readout memory–mixing tradeoff")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "joint_readout_memory_mixing_tradeoff.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    joint = paired.loc[paired["representation"].eq("palindrome_plus_all_orders")]
    seeds = sorted(joint["seed"].unique())
    local = joint.set_index("seed").reindex(seeds)
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    axis.bar(np.arange(len(seeds)), local["mixing_sum_on_minus_off"])
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(np.arange(len(seeds)))
    axis.set_xticklabels([str(seed) for seed in seeds])
    axis.set_ylabel("Mixing capacity: on minus off")
    axis.set_title("Paired interaction contribution by seed")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    filename = "joint_readout_paired_interaction_advantage.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_bivariate_all_orders_readout_assay(
    *,
    source_run: Path,
    results_root: Path,
    config: BivariateAllOrdersReadoutConfig = BivariateAllOrdersReadoutConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    with materialize_source_run(Path(source_run)) as source_root:
        source_config = _source_config(source_root)
        permutations = (
            int(config.permutations)
            if config.permutations is not None
            else int(source_config.permutations)
        )
        if permutations < 1:
            raise ValueError("resolved permutation count must be positive")

        run_dir = begin_run(
            Path(results_root),
            {
                "config": config.to_dict(),
                "source_run": str(Path(source_run)),
                "source_config": source_config.to_dict(),
                "quantum_simulations_executed": 0,
                "saved_probabilities_reused": True,
                "representations": list(REPRESENTATIONS),
                "scientific_question": (
                    "Can a single regularized linear readout recover both balanced memory and "
                    "cross-channel mixing when it receives the palindrome and all four ordered "
                    "longitudinal/transverse observable banks simultaneously?"
                ),
            },
            run_id=run_id,
        )
        selection_dir = run_dir / "selection_candidates"
        selection_dir.mkdir(parents=True, exist_ok=False)

        task_frames: list[pd.DataFrame] = []
        observed_rows: list[dict[str, object]] = []
        null_rows: list[dict[str, object]] = []

        for seed in source_config.seeds:
            capacity_config = source_config.stability_config().capacity_config(int(seed))
            windows = generate_exchange_symmetric_windows(
                source_config.stability_config(), seed=int(seed)
            )
            targets = build_capacity_targets(windows, capacity_config)
            rng = np.random.default_rng(int(seed) + 1_414_213_562)
            null_orders = paired_permutation_orders(
                source_config.samples,
                permutations,
                rng=rng,
            )

            for interaction in ("off", "on"):
                raw: dict[str, np.ndarray] = {}
                palindrome_probabilities = _load_probabilities(
                    source_root, int(seed), interaction, "palindrome"
                )
                raw["palindrome"] = build_crossover_feature_banks(
                    palindrome_probabilities
                )["occupation_pair_raw"]
                for schedule in BILINEAR_SCHEDULES:
                    probabilities = _load_probabilities(
                        source_root, int(seed), interaction, schedule
                    )
                    raw[schedule] = build_crossover_feature_banks(probabilities)[
                        "occupation_pair_raw"
                    ]

                representations = build_all_orders_representations(raw)
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
                        selection_dir
                        / f"seed_{seed}__{interaction}__{representation}.csv",
                        index=False,
                    )
                    observed_rows.append(
                        {
                            "seed": int(seed),
                            "interaction": interaction,
                            "representation": representation,
                            "feature_count": int(matrix.shape[1]),
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
        paired = _paired_on_off(seed_summary)
        plots = _render_plots(aggregate, paired, run_dir / "plots")

        task_metrics.to_csv(run_dir / "task_metrics.csv.gz", index=False, compression="gzip")
        observed.to_csv(run_dir / "observed_seed_metrics.csv", index=False)
        seed_summary.to_csv(run_dir / "joint_readout_seed_summary.csv", index=False)
        aggregate.to_csv(run_dir / "joint_readout_summary.csv", index=False)
        paired.to_csv(run_dir / "paired_on_off_differences.csv", index=False)
        null_frame.to_csv(
            run_dir / "paired_permutation_null.csv.gz",
            index=False,
            compression="gzip",
        )

        on = aggregate.loc[aggregate["interaction"].eq("on")].copy()
        paired_joint = paired.loc[
            paired["representation"].eq("palindrome_plus_all_orders")
        ]
        eligible = on.loc[
            ~on["representation"].eq("palindrome_control")
            & on["both_channels_early_above_null_fraction"].eq(1.0)
            & on["minimum_delay5_above_null_fraction"].eq(1.0)
            & on["early_balance_ratio_min"].ge(0.80)
            & on["mixing_above_null_fraction"].eq(1.0)
            & on["mixing_margin_min"].gt(0.0)
        ].copy()
        paired_advantage_all_seeds = bool(
            not paired_joint.empty
            and paired_joint["mixing_sum_on_minus_off"].gt(0.0).all()
        )
        promoted = None
        if not eligible.empty:
            eligible_names = set(eligible["representation"])
            for name in ("palindrome_plus_all_orders", "all_orders_joint"):
                if name in eligible_names:
                    promoted = eligible.loc[eligible["representation"].eq(name)].iloc[0]
                    break
        gate_passed = bool(promoted is not None and paired_advantage_all_seeds)
        strongest = on.sort_values(
            ["mixing_margin_mean", "minimum_delay5_mean", "minimum_early_mean"],
            ascending=[False, False, False],
        ).iloc[0]

        summary = {
            "status": "bivariate_all_orders_readout_complete",
            "quantum_simulations_executed": 0,
            "source_status": "bivariate_bilinear_mixing_complete",
            "seeds_completed": int(seed_summary["seed"].nunique()),
            "representations_completed": int(seed_summary["representation"].nunique()),
            "interaction_conditions_completed": int(seed_summary["interaction"].nunique()),
            "mixing_gate_passed": gate_passed,
            "paired_interaction_advantage_all_seeds": paired_advantage_all_seeds,
            "promoted_representation": (
                str(promoted["representation"]) if gate_passed else None
            ),
            "strongest_candidate_representation": str(strongest["representation"]),
            "strongest_candidate_mixing_margin_mean": float(
                strongest["mixing_margin_mean"]
            ),
            "strongest_candidate_minimum_delay5_mean": float(
                strongest["minimum_delay5_mean"]
            ),
            "decision_rule": (
                "Promote only when one joint readout preserves balanced early and delay-5 "
                "memory in every seed, exceeds its paired mixing null in every seed, and "
                "shows a positive interaction-on minus interaction-off mixing difference "
                "for every seed."
            ),
            "plots": plots,
            "config": config.to_dict(),
            "source_config": source_config.to_dict(),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return run_dir
