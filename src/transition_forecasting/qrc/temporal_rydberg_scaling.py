from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
    controlled_level_windows,
    effective_rank,
    fit_level_rate_scaler,
    transform_level_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)


SCALING_CONDITIONS = ("ordered", "shuffled", "interaction_off")


@dataclass(frozen=True)
class TemporalRydbergScalingConfig:
    """Bounded exact-state scaling assay for the frozen temporal reservoir.

    This assay deliberately measures representation and computational scaling,
    not financial forecasting skill. The same deterministic development panel,
    input sequence, physical spacings, drive parameters, and probe fractions are
    used at every atom count. Only ``n_atoms`` changes.
    """

    n_atoms: tuple[int, ...] = (5, 6, 8, 10)
    fold: int = 5
    lead: int = 5
    sequence_length: int = 40
    max_per_class: int = 2
    seed: int = 20260722
    batch_size: int = 2
    conditions: tuple[str, ...] = SCALING_CONDITIONS
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.n_atoms or len(set(self.n_atoms)) != len(self.n_atoms):
            raise ValueError("n_atoms must be nonempty and unique")
        if any(int(value) < 2 for value in self.n_atoms):
            raise ValueError("all atom counts must be at least two")
        if tuple(sorted(self.n_atoms)) != self.n_atoms:
            raise ValueError("n_atoms must be strictly increasing")
        if self.fold < 1 or self.lead < 1:
            raise ValueError("fold and lead must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.max_per_class < 1 or self.batch_size < 1:
            raise ValueError("max_per_class and batch_size must be positive")
        unknown = set(self.conditions).difference(SCALING_CONDITIONS)
        if unknown:
            raise ValueError(f"unsupported scaling conditions: {sorted(unknown)}")
        if "ordered" not in self.conditions:
            raise ValueError("ordered must be included in scaling conditions")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def exact_state_memory_mib(n_atoms: int, samples: int = 1) -> float:
    """Return complex128 state-vector storage, excluding temporary arrays."""

    if n_atoms < 1 or samples < 1:
        raise ValueError("n_atoms and samples must be positive")
    bytes_required = samples * (2**int(n_atoms)) * np.dtype(np.complex128).itemsize
    return float(bytes_required / (1024.0**2))


def _build_features_in_batches(
    windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    *,
    condition: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object], float]:
    """Build exact features in small batches and return measured wall time."""

    array = np.asarray(windows, dtype=float)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError("windows must have shape (samples, time, 2)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    blocks: list[np.ndarray] = []
    metadata: dict[str, object] | None = None
    started = time.perf_counter()
    for start in range(0, len(array), batch_size):
        stop = min(len(array), start + batch_size)
        block, block_metadata = build_temporal_rydberg_chain_features(
            array[start:stop],
            reservoir,
            condition=condition,
        )
        blocks.append(block)
        if metadata is None:
            metadata = block_metadata
        elif block_metadata["feature_count"] != metadata["feature_count"]:
            raise RuntimeError("feature count changed between batches")
    elapsed = float(time.perf_counter() - started)
    if metadata is None:
        raise ValueError("cannot build features for an empty sample panel")
    return np.concatenate(blocks, axis=0), metadata, elapsed


def _condition_row(
    *,
    n_atoms: int,
    condition: str,
    features: np.ndarray,
    train_mask: np.ndarray,
    elapsed_seconds: float,
    metadata: dict[str, object],
    batch_size: int,
) -> dict[str, object]:
    train = np.asarray(features[np.asarray(train_mask, dtype=bool)], dtype=float)
    centered = train - train.mean(axis=0, keepdims=True)
    standard_deviation = train.std(axis=0)
    return {
        "n_atoms": int(n_atoms),
        "condition": str(condition),
        "samples": int(len(features)),
        "train_samples": int(len(train)),
        "state_dimension": int(2**int(n_atoms)),
        "state_memory_mib_per_sample": exact_state_memory_mib(int(n_atoms)),
        "state_memory_mib_per_batch": exact_state_memory_mib(
            int(n_atoms), samples=int(batch_size)
        ),
        "feature_count": int(features.shape[1]),
        "effective_rank_train": effective_rank(train),
        "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
        "effective_rank_fraction": float(
            effective_rank(train) / max(1, min(train.shape))
        ),
        "near_constant_features": int((standard_deviation < 1e-8).sum()),
        "mean_feature_std": float(standard_deviation.mean()),
        "max_feature_std": float(standard_deviation.max(initial=0.0)),
        "wall_seconds": float(elapsed_seconds),
        "seconds_per_sample": float(elapsed_seconds / len(features)),
        "probe_steps": json.dumps(metadata["probe_steps"]),
        "total_evolution_time_us": float(metadata["total_evolution_time_us"]),
    }


def _comparison_rows(
    n_atoms: int,
    features_by_condition: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    ordered = np.asarray(features_by_condition["ordered"], dtype=float)
    scale = max(float(np.mean(np.abs(ordered))), 1e-12)
    rows: list[dict[str, object]] = []
    for condition, features in features_by_condition.items():
        if condition == "ordered":
            continue
        difference = ordered - np.asarray(features, dtype=float)
        rows.append(
            {
                "n_atoms": int(n_atoms),
                "comparison": f"ordered_minus_{condition}",
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "root_mean_square_difference": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "relative_mean_absolute_difference": float(
                    np.mean(np.abs(difference)) / scale
                ),
            }
        )
    return rows


def aggregate_scaling_summary(
    condition_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Return one compact row per atom count for reporting and plotting."""

    ordered = condition_metrics.loc[
        condition_metrics["condition"].eq("ordered")
    ].copy()
    columns = [
        "n_atoms",
        "state_dimension",
        "state_memory_mib_per_sample",
        "state_memory_mib_per_batch",
        "samples",
        "feature_count",
        "effective_rank_train",
        "numerical_rank_train",
        "effective_rank_fraction",
        "mean_feature_std",
        "wall_seconds",
        "seconds_per_sample",
    ]
    summary = ordered[columns].copy()

    comparison_map = {
        "ordered_minus_shuffled": "order_sensitivity_mad",
        "ordered_minus_interaction_off": "interaction_sensitivity_mad",
    }
    for comparison_name, output_name in comparison_map.items():
        local = comparisons.loc[
            comparisons["comparison"].eq(comparison_name),
            ["n_atoms", "mean_absolute_difference"],
        ].rename(columns={"mean_absolute_difference": output_name})
        summary = summary.merge(local, on="n_atoms", how="left", validate="one_to_one")
    return summary.sort_values("n_atoms").reset_index(drop=True)


def _render_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.plot(summary["n_atoms"], summary["seconds_per_sample"], marker="o")
    axis.set_yscale("log")
    axis.set(
        title="Exact temporal Rydberg runtime scaling",
        xlabel="Atom / qubit count",
        ylabel="Wall seconds per sample (log scale)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "runtime_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.plot(summary["n_atoms"], summary["effective_rank_train"], marker="o")
    axis.set(
        title="Ordered feature effective rank versus reservoir size",
        xlabel="Atom / qubit count",
        ylabel="Effective rank on training panel",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "effective_rank_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    sensitivity_columns = [
        ("order_sensitivity_mad", "Ordered − shuffled"),
        ("interaction_sensitivity_mad", "Ordered − interactions off"),
    ]
    available = [item for item in sensitivity_columns if item[0] in summary.columns]
    if available:
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        for column, label in available:
            axis.plot(summary["n_atoms"], summary[column], marker="o", label=label)
        axis.set(
            title="Mechanism sensitivity versus reservoir size",
            xlabel="Atom / qubit count",
            ylabel="Mean absolute feature difference",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        filename = "mechanism_sensitivity_scaling.png"
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    return outputs


def run_temporal_rydberg_scaling(
    *,
    fold_dir: Path,
    results_root: Path,
    scaling: TemporalRydbergScalingConfig = TemporalRydbergScalingConfig(),
    reservoir_template: TemporalRydbergChainConfig = TemporalRydbergChainConfig(),
    run_id: str | None = None,
) -> Path:
    """Run the bounded exact-state scaling assay on development rows only."""

    scaling.validate()
    reservoir_template.validate()
    if reservoir_template.shots is not None:
        raise ValueError("scaling assay requires exact features; set shots=None")

    dataset = load_rolling_fold_dataset(Path(fold_dir))
    level_channel = resolve_level_channel(
        dataset,
        name=scaling.level_channel_name,
        fallback=scaling.fallback_level_channel,
    )
    selected = select_balanced_episode_rows(
        dataset.manifest,
        fold=int(scaling.fold),
        lead=int(scaling.lead),
        max_per_class=int(scaling.max_per_class),
        splits=DEVELOPMENT_FOLD_SPLITS,
        seed=int(scaling.seed),
    )
    if selected["fold_split"].eq("test").any():
        raise RuntimeError("scaling assay must not receive test rows")

    tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
    level = extract_level_windows(
        dataset,
        tensor_rows,
        sequence_length=int(scaling.sequence_length),
        level_channel=level_channel,
    )
    usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
    frame = selected.loc[usable].reset_index(drop=True)
    level = level[usable]
    tensor_rows = tensor_rows[usable]
    if frame.empty:
        raise RuntimeError("no usable development samples remain")

    train_mask = frame["fold_split"].eq("train").to_numpy()
    if not train_mask.any() or not frame["fold_split"].eq("val").any():
        raise RuntimeError("scaling panel requires train and validation rows")
    scaler = fit_level_rate_scaler(level, train_mask)

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "scaling": scaling.to_dict(),
            "reservoir_template": reservoir_template.to_dict(),
            "scientific_scope": (
                "Exact-state representation and computational scaling only; "
                "no financial model selection and no test rows."
            ),
        },
        run_id=run_id,
    )
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    condition_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for n_atoms in scaling.n_atoms:
        defect_edge = min(int(reservoir_template.defect_edge), int(n_atoms) - 2)
        reservoir = replace(
            reservoir_template,
            n_atoms=int(n_atoms),
            defect_edge=defect_edge,
            shots=None,
        )
        features_by_condition: dict[str, np.ndarray] = {}
        for condition in scaling.conditions:
            controlled = controlled_level_windows(
                level,
                condition,
                seed=int(scaling.seed + 1009 * int(n_atoms)),
            )
            windows = transform_level_windows(controlled, scaler)
            features, metadata, elapsed = _build_features_in_batches(
                windows,
                reservoir,
                condition=condition,
                batch_size=int(scaling.batch_size),
            )
            features_by_condition[condition] = features
            np.savez_compressed(
                feature_dir / f"n{int(n_atoms)}_{condition}.npz",
                sample_id=frame["sample_id"].astype(str).to_numpy(),
                features=features,
            )
            condition_rows.append(
                _condition_row(
                    n_atoms=int(n_atoms),
                    condition=condition,
                    features=features,
                    train_mask=train_mask,
                    elapsed_seconds=elapsed,
                    metadata=metadata,
                    batch_size=int(scaling.batch_size),
                )
            )
            metadata_rows.append(
                {
                    "n_atoms": int(n_atoms),
                    "condition": condition,
                    "metadata": json.dumps(metadata, sort_keys=True),
                }
            )
        comparison_rows.extend(_comparison_rows(int(n_atoms), features_by_condition))

    condition_metrics = pd.DataFrame(condition_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summary = aggregate_scaling_summary(condition_metrics, comparisons)

    retained = frame[
        [
            "sample_id",
            "fold",
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
        ]
    ].copy()
    retained["tensor_row"] = tensor_rows

    condition_metrics.to_csv(run_dir / "condition_metrics.csv", index=False)
    comparisons.to_csv(run_dir / "mechanism_comparisons.csv", index=False)
    summary.to_csv(run_dir / "scaling_summary.csv", index=False)
    retained.to_csv(run_dir / "retained_samples.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(run_dir / "reservoir_metadata.csv", index=False)
    plots = _render_plots(summary, run_dir / "plots")

    payload = {
        "status": "temporal_rydberg_scaling_complete",
        "test_evaluated": False,
        "financial_forecasting_metrics_evaluated": False,
        "n_atoms": list(scaling.n_atoms),
        "sample_count": int(len(frame)),
        "train_sample_count": int(train_mask.sum()),
        "validation_sample_count": int((~train_mask).sum()),
        "fold": int(scaling.fold),
        "lead": int(scaling.lead),
        "sequence_length": int(scaling.sequence_length),
        "conditions": list(scaling.conditions),
        "scaler": scaler.to_dict(),
        "plots": plots,
        "known_limitations": [
            "Small deterministic development panel chosen to keep exact-state scaling bounded.",
            "This assay characterizes representation, mechanism sensitivity, runtime, and memory; it does not rank financial forecasts.",
            "Full Hilbert-space state-vector cost grows exponentially with atom count.",
            "MNIST will provide the separate task-performance scaling comparison.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return run_dir
