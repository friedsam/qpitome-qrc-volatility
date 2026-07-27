from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.qrc.bivariate_capacity_dynamics import (
    FeatureBankName,
    QRC_FEATURE_BANKS,
    _evolve_segment_batch,
    build_feature_banks,
    evolve_sequential_probe_probabilities,
    probabilities_to_full_low_order,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

EncodingName = Literal["simultaneous", "sequential_noncommuting"]
ENCODINGS: tuple[EncodingName, ...] = (
    "simultaneous",
    "sequential_noncommuting",
)


@dataclass(frozen=True)
class BivariateCapacityConfig:
    samples: int = 640
    sequence_length: int = 40
    memory_delays: tuple[int, ...] = (1, 2, 4, 5, 8, 12)
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    interaction_scale: float = 1.25
    sequential_first_slot_fraction: float = 0.50
    sequential_second_phase_rad: float = float(np.pi / 2.0)
    seed: int = 20260724

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("samples must be at least 100")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be at least eight")
        if not self.memory_delays or any(delay < 1 for delay in self.memory_delays):
            raise ValueError("memory_delays must be positive and nonempty")
        if max(self.memory_delays) >= self.sequence_length - 1:
            raise ValueError("memory delay lies outside the synthetic window")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive and nonempty")
        if not 0.4 <= self.train_fraction < 0.9:
            raise ValueError("train_fraction must lie in [0.4, 0.9)")
        if not 0.05 <= self.validation_fraction < 0.4:
            raise ValueError("validation_fraction must lie in [0.05, 0.4)")
        if self.train_fraction + self.validation_fraction >= 0.95:
            raise ValueError("train and validation fractions leave too little test data")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if not 0.1 <= self.sequential_first_slot_fraction <= 0.9:
            raise ValueError("sequential_first_slot_fraction must lie in [0.1, 0.9]")
        if not np.isfinite(self.sequential_second_phase_rad):
            raise ValueError("sequential_second_phase_rad must be finite")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticTargets:
    values: np.ndarray
    metadata: pd.DataFrame


def generate_bivariate_windows(config: BivariateCapacityConfig) -> np.ndarray:
    config.validate()
    rng = np.random.default_rng(config.seed)
    return rng.uniform(
        -1.0,
        1.0,
        size=(config.samples, config.sequence_length, 2),
    ).astype(float)


def _delayed(windows: np.ndarray, channel: int, delay: int) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    if delay < 0 or delay >= values.shape[1]:
        raise ValueError("delay lies outside the input window")
    return values[:, -(delay + 1), int(channel)]


def build_capacity_targets(
    windows: np.ndarray,
    config: BivariateCapacityConfig,
) -> SyntheticTargets:
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("windows must be finite with shape (samples, time, 2)")
    if values.shape[1] != config.sequence_length:
        raise ValueError("window length differs from capacity configuration")

    columns: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for channel in (0, 1):
        for delay in config.memory_delays:
            columns.append(_delayed(values, channel, int(delay)))
            rows.append(
                {
                    "task": f"memory_u{channel + 1}_d{int(delay)}",
                    "group": "memory",
                    "channel": int(channel + 1),
                    "delay_a": int(delay),
                    "delay_b": np.nan,
                }
            )

    for name, delay_a, delay_b in (
        ("mix_same_d1", 1, 1),
        ("mix_u1d1_u2d2", 1, 2),
        ("mix_u1d2_u2d1", 2, 1),
        ("mix_u1d2_u2d5", 2, 5),
        ("mix_u1d5_u2d2", 5, 2),
    ):
        columns.append(
            _delayed(values, 0, delay_a) * _delayed(values, 1, delay_b)
        )
        rows.append(
            {
                "task": name,
                "group": "mixing",
                "channel": np.nan,
                "delay_a": int(delay_a),
                "delay_b": int(delay_b),
            }
        )

    columns.append(
        _delayed(values, 0, 1) * _delayed(values, 1, 2)
        - _delayed(values, 1, 1) * _delayed(values, 0, 2)
    )
    rows.append(
        {
            "task": "antisymmetric_order_d1_d2",
            "group": "order",
            "channel": np.nan,
            "delay_a": 1,
            "delay_b": 2,
        }
    )

    target = np.column_stack(columns)
    metadata = pd.DataFrame(rows)
    if target.shape != (len(values), len(metadata)) or not np.isfinite(target).all():
        raise RuntimeError("capacity target construction failed")
    return SyntheticTargets(values=target, metadata=metadata)


def split_masks(config: BivariateCapacityConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(np.floor(config.samples * config.train_fraction))
    validation_end = train_end + int(
        np.floor(config.samples * config.validation_fraction)
    )
    indices = np.arange(config.samples)
    train = indices < train_end
    validation = (indices >= train_end) & (indices < validation_end)
    test = indices >= validation_end
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) < 20:
        raise ValueError("synthetic split is too small")
    return train, validation, test


def squared_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if np.std(observed) <= 1e-12 or np.std(predicted) <= 1e-12:
        return 0.0
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation * correlation, 0.0, 1.0))


def normalized_capacity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return max(0, 1 - MSE / mean(y^2)) for zero-mean synthetic targets."""

    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    denominator = float(np.mean(observed**2))
    if denominator <= 1e-12:
        return 0.0
    value = 1.0 - float(np.mean((observed - predicted) ** 2)) / denominator
    return float(np.clip(value, 0.0, 1.0))


def fit_capacity_readout(
    features: np.ndarray,
    targets: SyntheticTargets,
    config: BivariateCapacityConfig,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Fit one independently selected linear readout per capacity target."""

    matrix = np.asarray(features, dtype=float)
    y = np.asarray(targets.values, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(y) or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite matrix aligned with targets")
    train, validation, test = split_masks(config)

    train_scaler = StandardScaler().fit(matrix[train])
    train_design = train_scaler.transform(matrix[train])
    validation_design = train_scaler.transform(matrix[validation])
    validation_predictions: dict[float, np.ndarray] = {}
    candidate_rows: list[dict[str, object]] = []
    for alpha in config.alphas:
        model = Ridge(alpha=float(alpha)).fit(train_design, y[train])
        prediction = model.predict(validation_design)
        validation_predictions[float(alpha)] = prediction
        for task_index, task in targets.metadata.iterrows():
            observed = y[validation, task_index]
            estimated = prediction[:, task_index]
            candidate_rows.append(
                {
                    **task.to_dict(),
                    "task_index": int(task_index),
                    "alpha": float(alpha),
                    "validation_capacity": normalized_capacity(observed, estimated),
                    "validation_corr2": squared_correlation(observed, estimated),
                    "validation_r2": float(r2_score(observed, estimated)),
                }
            )
    candidates = pd.DataFrame(candidate_rows)

    selected_rows: list[pd.Series] = []
    for _, group in candidates.groupby("task_index", sort=True):
        selected_rows.append(
            min(
                (row for _, row in group.iterrows()),
                key=lambda row: (
                    -float(row["validation_capacity"]),
                    -float(row["validation_corr2"]),
                    -float(row["validation_r2"]),
                    float(row["alpha"]),
                ),
            )
        )
    selected = pd.DataFrame(selected_rows).sort_values("task_index").reset_index(drop=True)

    fit = train | validation
    fit_scaler = StandardScaler().fit(matrix[fit])
    fit_design = fit_scaler.transform(matrix[fit])
    test_design = fit_scaler.transform(matrix[test])
    test_predictions: dict[float, np.ndarray] = {}
    for alpha in sorted(set(selected["alpha"].astype(float))):
        model = Ridge(alpha=float(alpha)).fit(fit_design, y[fit])
        test_predictions[float(alpha)] = model.predict(test_design)

    metric_rows: list[dict[str, object]] = []
    for _, choice in selected.iterrows():
        task_index = int(choice["task_index"])
        alpha = float(choice["alpha"])
        observed = y[test, task_index]
        estimated = test_predictions[alpha][:, task_index]
        task = targets.metadata.iloc[task_index]
        metric_rows.append(
            {
                **task.to_dict(),
                "task_index": task_index,
                "selected_alpha": alpha,
                "validation_capacity": float(choice["validation_capacity"]),
                "validation_corr2": float(choice["validation_corr2"]),
                "test_samples": int(test.sum()),
                "capacity": normalized_capacity(observed, estimated),
                "corr2": squared_correlation(observed, estimated),
                "r2": float(r2_score(observed, estimated)),
                "rmse": float(np.sqrt(np.mean((observed - estimated) ** 2))),
            }
        )

    selected_alphas = selected["alpha"].to_numpy(dtype=float)
    diagnostics = {
        "feature_count": float(matrix.shape[1]),
        "effective_rank_train_validation": float(effective_rank(matrix[fit])),
        "numerical_rank_train_validation": float(
            np.linalg.matrix_rank(matrix[fit] - matrix[fit].mean(axis=0, keepdims=True))
        ),
        "selected_alpha_min": float(selected_alphas.min()),
        "selected_alpha_median": float(np.median(selected_alphas)),
        "selected_alpha_max": float(selected_alphas.max()),
    }
    return pd.DataFrame(metric_rows), diagnostics, candidates


def capacity_summary(task_metrics: pd.DataFrame) -> dict[str, float]:
    memory = task_metrics.loc[task_metrics["group"].eq("memory")]
    mixing = task_metrics.loc[task_metrics["group"].eq("mixing")]
    order = task_metrics.loc[task_metrics["group"].eq("order")]
    delay_five = memory.loc[memory["delay_a"].eq(5)]
    return {
        "memory_capacity_u1": float(memory.loc[memory["channel"].eq(1), "capacity"].sum()),
        "memory_capacity_u2": float(memory.loc[memory["channel"].eq(2), "capacity"].sum()),
        "mixing_capacity": float(mixing["capacity"].sum()),
        "order_capacity": float(order["capacity"].sum()),
        "mean_memory_capacity": float(memory["capacity"].mean()),
        "mean_mixing_capacity": float(mixing["capacity"].mean()),
        "mean_order_capacity": float(order["capacity"].mean()),
        "mean_delay5_memory_capacity": float(delay_five["capacity"].mean()),
        "mean_memory_corr2": float(memory["corr2"].mean()),
        "mean_mixing_corr2": float(mixing["corr2"].mean()),
        "mean_order_corr2": float(order["corr2"].mean()),
        "mean_task_r2": float(task_metrics["r2"].mean()),
    }


def _render_plots(
    task_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    interacting = task_metrics.loc[
        task_metrics["interaction"].eq("on")
        & task_metrics["feature_bank"].isin(
            ["six_mode_density_curvature", "all_occupations", "full_one_two_body"]
        )
        & task_metrics["group"].eq("memory")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True, sharey=True)
    for row, encoding in enumerate(ENCODINGS):
        for column, channel in enumerate((1, 2)):
            axis = axes[row, column]
            local = interacting.loc[
                interacting["encoding"].eq(encoding)
                & interacting["channel"].eq(channel)
            ]
            for bank, group in local.groupby("feature_bank", sort=True):
                group = group.sort_values("delay_a")
                axis.plot(group["delay_a"], group["capacity"], marker="o", label=bank)
            axis.set_title(f"{encoding}, channel {channel}")
            axis.set_xlabel("Delay")
            axis.set_ylabel("Normalized capacity")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    figure.suptitle("Bivariate memory curves with interactions")
    figure.tight_layout()
    filename = "memory_curves.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    qrc = summary.loc[~summary["encoding"].eq("raw_input")].reset_index(drop=True)
    labels = qrc["encoding"] + " / " + qrc["interaction"] + " / " + qrc["feature_bank"]
    for metric, title, filename in (
        ("mixing_capacity", "Cross-channel mixing capacity", "mixing_capacity.png"),
        ("order_capacity", "Order-sensitive bivariate capacity", "order_capacity.png"),
    ):
        figure, axis = plt.subplots(figsize=(11.5, max(6.0, 0.31 * len(qrc))))
        axis.barh(np.arange(len(qrc)), qrc[metric])
        axis.set_yticks(np.arange(len(qrc)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.set_xlabel("Sum of held-out normalized capacities")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
        outputs.append(filename)

    selected = qrc.loc[
        qrc["interaction"].eq("on")
        & qrc["feature_bank"].isin(
            ["six_mode_density_curvature", "full_one_two_body"]
        )
    ].copy()
    selected["case"] = selected["encoding"] + " / " + selected["feature_bank"]
    x = np.arange(len(selected))
    width = 0.26
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    axis.bar(
        x - width,
        selected["mean_delay5_memory_capacity"],
        width,
        label="delay-5 memory",
    )
    axis.bar(x, selected["mean_mixing_capacity"], width, label="mean mixing")
    axis.bar(x + width, selected["mean_order_capacity"], width, label="order")
    axis.set_xticks(x)
    axis.set_xticklabels(selected["case"], rotation=25, ha="right")
    axis.set_ylabel("Normalized capacity")
    axis.set_title("Compression and encoding diagnosis")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "compression_encoding_diagnosis.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for bank, group in qrc.groupby("feature_bank", sort=True):
        axis.scatter(
            group["effective_rank_train_validation"],
            group["mixing_capacity"],
            label=bank,
        )
    axis.set_xlabel("Effective feature rank")
    axis.set_ylabel("Mixing capacity")
    axis.set_title("Does additional observable rank create usable nonlinear mixing?")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    filename = "rank_vs_mixing.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def _one_summary(
    summary: pd.DataFrame,
    *,
    encoding: str,
    interaction: str,
    feature_bank: str,
) -> pd.Series:
    rows = summary.loc[
        summary["encoding"].eq(encoding)
        & summary["interaction"].eq(interaction)
        & summary["feature_bank"].eq(feature_bank)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one capacity row for {encoding}/{interaction}/{feature_bank}"
        )
    return rows.iloc[0]


def run_bivariate_capacity_assay(
    *,
    results_root: Path,
    config: BivariateCapacityConfig = BivariateCapacityConfig(),
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
        shot_seed=config.seed,
    )
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    reservoir_config.validate()
    geometry_config.validate()
    if reservoir_config.n_atoms != 6 or reservoir_config.shots is not None:
        raise ValueError("capacity assay requires the exact six-atom ladder")

    run_dir = begin_run(
        Path(results_root),
        {
            "config": config.to_dict(),
            "reservoir": reservoir_config.to_dict(),
            "geometry": geometry_config.to_dict(),
            "financial_data_used": False,
            "financial_test_rows_used": 0,
            "capacity_definition": "max(0, 1 - MSE / mean(target^2))",
            "readout_selection": "independent chronological synthetic validation per task",
            "scientific_questions": [
                "Does the incumbent reservoir retain each independent channel?",
                "Does it form linearly decodable nonlinear cross-channel products?",
                "Does it retain order-sensitive bivariate information?",
                "Does six-mode compression discard capacity present in atom/pair observables?",
                "Do sequential noncommuting slots improve mixing over simultaneous encoding?",
                "Are interactions required for observed mixing capacity?",
            ],
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    selection_dir = run_dir / "selection_candidates"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir.mkdir(parents=True, exist_ok=False)

    windows = generate_bivariate_windows(config)
    targets = build_capacity_targets(windows, config)
    np.savez_compressed(
        run_dir / "synthetic_inputs_targets.npz",
        windows=windows,
        targets=targets.values,
        task_names=targets.metadata["task"].astype(str).to_numpy(),
    )
    targets.metadata.to_csv(run_dir / "task_definitions.csv", index=False)

    task_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []

    def evaluate_case(
        matrix: np.ndarray,
        *,
        encoding: str,
        interaction: str,
        feature_bank: str,
        candidate_filename: str,
    ) -> None:
        task_table, diagnostics, candidates = fit_capacity_readout(matrix, targets, config)
        task_table.insert(0, "encoding", encoding)
        task_table.insert(1, "interaction", interaction)
        task_table.insert(2, "feature_bank", feature_bank)
        task_frames.append(task_table)
        summary_rows.append(
            {
                "encoding": encoding,
                "interaction": interaction,
                "feature_bank": feature_bank,
                **diagnostics,
                **capacity_summary(task_table),
            }
        )
        selection_frames.append(
            task_table[
                [
                    "encoding",
                    "interaction",
                    "feature_bank",
                    "task",
                    "group",
                    "selected_alpha",
                    "validation_capacity",
                    "validation_corr2",
                ]
            ].copy()
        )
        candidates.to_csv(selection_dir / candidate_filename, index=False)

    evaluate_case(
        windows.reshape(len(windows), -1),
        encoding="raw_input",
        interaction="none",
        feature_bank="raw_input_linear",
        candidate_filename="raw_input__none__raw_input_linear.csv",
    )

    simulation_metadata: dict[str, object] = {}
    for encoding in ENCODINGS:
        for interaction_name, interaction_scale in (
            ("on", config.interaction_scale),
            ("off", 0.0),
        ):
            if encoding == "simultaneous":
                probabilities, metadata = evolve_ladder_probe_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    interaction_scale=float(interaction_scale),
                    condition="ordered",
                )
                metadata = {**metadata, "encoding": encoding}
            else:
                probabilities, metadata = evolve_sequential_probe_probabilities(
                    windows,
                    reservoir_config,
                    geometry_config,
                    interaction_scale=float(interaction_scale),
                    first_slot_fraction=config.sequential_first_slot_fraction,
                    second_phase_rad=config.sequential_second_phase_rad,
                )
            case = f"{encoding}__interaction_{interaction_name}"
            simulation_metadata[case] = metadata
            np.savez_compressed(
                probability_dir / f"{case}.npz",
                probabilities=probabilities,
            )
            for bank_name, matrix in build_feature_banks(probabilities).items():
                evaluate_case(
                    matrix,
                    encoding=encoding,
                    interaction=interaction_name,
                    feature_bank=bank_name,
                    candidate_filename=f"{case}__{bank_name}.csv",
                )

    task_metrics = pd.concat(task_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    selections = pd.concat(selection_frames, ignore_index=True)
    task_metrics.to_csv(run_dir / "task_metrics.csv", index=False)
    summary.to_csv(run_dir / "capacity_summary.csv", index=False)
    selections.to_csv(run_dir / "selected_readouts.csv", index=False)
    plots = _render_plots(task_metrics, summary, run_dir / "plots")

    simultaneous_six = _one_summary(
        summary,
        encoding="simultaneous",
        interaction="on",
        feature_bank="six_mode_density_curvature",
    )
    simultaneous_full = _one_summary(
        summary,
        encoding="simultaneous",
        interaction="on",
        feature_bank="full_one_two_body",
    )
    sequential_full = _one_summary(
        summary,
        encoding="sequential_noncommuting",
        interaction="on",
        feature_bank="full_one_two_body",
    )
    sequential_off = _one_summary(
        summary,
        encoding="sequential_noncommuting",
        interaction="off",
        feature_bank="full_one_two_body",
    )
    diagnostic = {
        "status": "bivariate_capacity_assay_complete",
        "financial_data_used": False,
        "financial_test_rows_used": 0,
        "capacity_definition": "max(0, 1 - MSE / mean(target^2))",
        "compression_mixing_gain_full_minus_six": float(
            simultaneous_full["mixing_capacity"] - simultaneous_six["mixing_capacity"]
        ),
        "compression_order_gain_full_minus_six": float(
            simultaneous_full["order_capacity"] - simultaneous_six["order_capacity"]
        ),
        "encoding_mixing_gain_sequential_minus_simultaneous": float(
            sequential_full["mixing_capacity"] - simultaneous_full["mixing_capacity"]
        ),
        "encoding_order_gain_sequential_minus_simultaneous": float(
            sequential_full["order_capacity"] - simultaneous_full["order_capacity"]
        ),
        "interaction_mixing_gain_sequential_on_minus_off": float(
            sequential_full["mixing_capacity"] - sequential_off["mixing_capacity"]
        ),
        "simultaneous_six_delay5_memory": float(
            simultaneous_six["mean_delay5_memory_capacity"]
        ),
        "simultaneous_full_delay5_memory": float(
            simultaneous_full["mean_delay5_memory_capacity"]
        ),
        "sequential_full_delay5_memory": float(
            sequential_full["mean_delay5_memory_capacity"]
        ),
        "interpretation": {
            "compression": (
                "Positive full-minus-six gains indicate that six-mode compression "
                "discards decodable capacity present in the same probabilities."
            ),
            "encoding": (
                "Positive sequential-minus-simultaneous gains indicate that explicit "
                "noncommuting slots improve temporal mixing."
            ),
            "interactions": (
                "Positive interacting-minus-off gains indicate that atom interactions, "
                "rather than encoding alone, create usable mixing."
            ),
            "financial_failure": (
                "If delay-5 memory and mixing both succeed while finance fails, focus on "
                "financial inputs/targets. If capacity fails, repair encoding, timescale, "
                "or observable compression before further financial modeling."
            ),
        },
        "plots": plots,
        "simulation_metadata": simulation_metadata,
    }
    (run_dir / "summary.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return run_dir
