"""Deterministic MNIST benchmark for the position-encoded Rydberg reservoir.

Images use the official MNIST train/test split. Train-only PCA supplies N-1
coordinates to an N-atom chain, where the coordinates set nearest-neighbour
spacings. Because N and input dimension co-vary, every result reports both.
Classical controls use the identical PCA coordinates.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.data.mnist_acquisition import load_mnist
from transition_forecasting.qrc.position_encoded_rydberg import (
    PositionEncodedConfig,
    build_position_encoded_features,
    fit_unit_scaler,
)
from transition_forecasting.qrc.temporal_rydberg_chain import effective_rank


@dataclass(frozen=True)
class MnistPositionBenchmarkConfig:
    atom_counts: tuple[int, ...] = (5, 6, 7, 9)
    train_size: int = 1000
    test_size: int = 200
    subsample_seed: int = 20260722
    random_feature_seed: int = 20260722
    logistic_c: float = 1.0
    logistic_max_iter: int = 3000
    shots: int | None = None
    shot_seed: int = 20260722
    spacing_min_um: float = 8.0
    spacing_max_um: float = 12.0
    omega_rad_us: float = 6.0
    delta_rad_us: float = 6.0
    total_time_us: float = 1.6
    probe_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    max_phase_per_substep: float = 0.25
    max_substeps: int = 20000

    def validate(self) -> None:
        if not self.atom_counts or len(set(self.atom_counts)) != len(self.atom_counts):
            raise ValueError("atom_counts must be nonempty and unique")
        if tuple(sorted(self.atom_counts)) != self.atom_counts:
            raise ValueError("atom_counts must be strictly increasing")
        if any(value < 3 for value in self.atom_counts):
            raise ValueError("position encoding requires at least three atoms")
        if self.train_size < 100 or self.test_size < 20:
            raise ValueError("train_size/test_size are too small for ten-class MNIST")
        if self.train_size % 10 or self.test_size % 10:
            raise ValueError("train_size and test_size must be divisible by ten")
        if self.logistic_c <= 0 or self.logistic_max_iter < 100:
            raise ValueError("invalid logistic-regression configuration")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive when supplied")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stratified_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    target = np.asarray(labels, dtype=int).ravel()
    per_class = size // 10
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    for label in range(10):
        available = np.flatnonzero(target == label)
        if len(available) < per_class:
            raise ValueError(f"digit {label} has only {len(available)} rows")
        blocks.append(rng.choice(available, size=per_class, replace=False))
    selected = np.concatenate(blocks)
    rng.shuffle(selected)
    return selected


def prepare_fixed_split(
    arrays: dict[str, np.ndarray], config: MnistPositionBenchmarkConfig
) -> dict[str, np.ndarray]:
    """Return one balanced deterministic subset shared by every atom count."""
    config.validate()
    train_images = np.asarray(arrays["train_images"], dtype=np.float32)
    test_images = np.asarray(arrays["test_images"], dtype=np.float32)
    train_labels = np.asarray(arrays["train_labels"], dtype=int).ravel()
    test_labels = np.asarray(arrays["test_labels"], dtype=int).ravel()
    train_idx = _stratified_indices(
        train_labels, config.train_size, config.subsample_seed
    )
    test_idx = _stratified_indices(
        test_labels, config.test_size, config.subsample_seed + 1
    )

    def flatten(images: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return (
            images[indices]
            .reshape(len(indices), -1)
            .astype(np.float64)
            / 255.0
        )

    return {
        "x_train": flatten(train_images, train_idx),
        "y_train": train_labels[train_idx],
        "x_test": flatten(test_images, test_idx),
        "y_test": test_labels[test_idx],
        "train_indices": train_idx,
        "test_indices": test_idx,
    }


def _accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    config: MnistPositionBenchmarkConfig,
) -> float:
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=config.logistic_c,
        max_iter=config.logistic_max_iter,
        random_state=config.subsample_seed,
    )
    model.fit(scaler.transform(train_features), train_labels)
    predicted = model.predict(scaler.transform(test_features))
    return float(np.mean(predicted == test_labels))


def _random_tanh(
    train_features: np.ndarray,
    test_features: np.ndarray,
    feature_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_features)
    train_z = scaler.transform(train_features)
    test_z = scaler.transform(test_features)
    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0,
        1.0 / np.sqrt(train_z.shape[1]),
        size=(train_z.shape[1], feature_count),
    )
    bias = rng.uniform(-1.0, 1.0, size=feature_count)
    return np.tanh(train_z @ weights + bias), np.tanh(test_z @ weights + bias)


def benchmark_atom_count(
    split: dict[str, np.ndarray],
    n_atoms: int,
    config: MnistPositionBenchmarkConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run one position-encoded benchmark size on the fixed sample split."""
    n_components = int(n_atoms) - 1
    pca = PCA(n_components=n_components, random_state=config.subsample_seed)
    pca_train = pca.fit_transform(split["x_train"])
    pca_test = pca.transform(split["x_test"])
    unit_scaler = fit_unit_scaler(pca_train)
    unit_train = unit_scaler.transform(pca_train)
    unit_test = unit_scaler.transform(pca_test)
    stacked = np.vstack([unit_train, unit_test])

    reservoir = PositionEncodedConfig(
        n_atoms=int(n_atoms),
        spacing_min_um=config.spacing_min_um,
        spacing_max_um=config.spacing_max_um,
        omega_rad_us=config.omega_rad_us,
        delta_rad_us=config.delta_rad_us,
        total_time_us=config.total_time_us,
        probe_fractions=config.probe_fractions,
        max_phase_per_substep=config.max_phase_per_substep,
        max_substeps=config.max_substeps,
        shots=config.shots,
        shot_seed=config.shot_seed,
    )

    started = time.perf_counter()
    qrc_all, metadata = build_position_encoded_features(
        stacked, reservoir, interactions=True
    )
    qrc_seconds = float(time.perf_counter() - started)
    started = time.perf_counter()
    off_all, off_metadata = build_position_encoded_features(
        stacked, reservoir, interactions=False
    )
    off_seconds = float(time.perf_counter() - started)

    n_train = len(unit_train)
    qrc_train, qrc_test = qrc_all[:n_train], qrc_all[n_train:]
    off_train, off_test = off_all[:n_train], off_all[n_train:]
    tanh_train, tanh_test = _random_tanh(
        pca_train,
        pca_test,
        qrc_train.shape[1],
        seed=config.random_feature_seed + int(n_atoms),
    )
    y_train, y_test = split["y_train"], split["y_test"]

    row: dict[str, object] = {
        "n_atoms": int(n_atoms),
        "input_dimension": n_components,
        "state_dimension": 2**int(n_atoms),
        "reservoir_feature_count": int(qrc_train.shape[1]),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "reservoir_accuracy": _accuracy(
            qrc_train, y_train, qrc_test, y_test, config
        ),
        "interaction_off_accuracy": _accuracy(
            off_train, y_train, off_test, y_test, config
        ),
        "linear_pca_accuracy": _accuracy(
            pca_train, y_train, pca_test, y_test, config
        ),
        "random_tanh_accuracy": _accuracy(
            tanh_train, y_train, tanh_test, y_test, config
        ),
        "majority_accuracy": float(
            np.max(np.bincount(y_test.astype(int))) / len(y_test)
        ),
        "effective_rank_train": effective_rank(qrc_train),
        "interaction_off_effective_rank_train": effective_rank(off_train),
        "qrc_wall_seconds": qrc_seconds,
        "interaction_off_wall_seconds": off_seconds,
        "seconds_per_sample": qrc_seconds / len(stacked),
        "shots": None if config.shots is None else int(config.shots),
        "substeps": int(metadata["substeps"]),
        "probe_steps": json.dumps(metadata["probe_steps"]),
        "blockade_radius_um": float(metadata["blockade_radius_um"]),
        "interaction_off_substeps": int(off_metadata["substeps"]),
    }
    row["reservoir_minus_linear"] = (
        float(row["reservoir_accuracy"])
        - float(row["linear_pca_accuracy"])
    )
    row["reservoir_minus_random_tanh"] = (
        float(row["reservoir_accuracy"])
        - float(row["random_tanh_accuracy"])
    )
    artifacts = {
        "pca_train": pca_train,
        "pca_test": pca_test,
        "unit_train": unit_train,
        "unit_test": unit_test,
        "qrc_train": qrc_train,
        "qrc_test": qrc_test,
        "interaction_off_train": off_train,
        "interaction_off_test": off_test,
    }
    return row, artifacts


def _plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    for column, label in (
        ("reservoir_accuracy", "Position-encoded QRC"),
        ("linear_pca_accuracy", "Linear PCA"),
        ("random_tanh_accuracy", "Random tanh"),
        ("interaction_off_accuracy", "Interactions off"),
    ):
        axis.plot(metrics["n_atoms"], metrics[column], marker="o", label=label)
    axis.set(
        title="MNIST accuracy versus atom / qubit count",
        xlabel="Atom / qubit count (input dimension = N-1)",
        ylabel="Test accuracy",
    )
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    filename = "mnist_accuracy_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    axis.plot(metrics["n_atoms"], metrics["effective_rank_train"], marker="o")
    axis.set(
        title="MNIST reservoir effective-rank scaling",
        xlabel="Atom / qubit count",
        ylabel="Effective rank on training features",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "mnist_effective_rank_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    axis.plot(metrics["n_atoms"], metrics["seconds_per_sample"], marker="o")
    axis.set_yscale("log")
    axis.set(
        title="MNIST exact-simulation runtime scaling",
        xlabel="Atom / qubit count",
        ylabel="Wall seconds per sample (log scale)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    filename = "mnist_runtime_scaling.png"
    figure.savefig(output_dir / filename, dpi=220)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def run_mnist_position_benchmark(
    *,
    raw_dir: Path,
    results_root: Path,
    config: MnistPositionBenchmarkConfig = MnistPositionBenchmarkConfig(),
    run_id: str | None = None,
) -> Path:
    config.validate()
    run_dir = begin_run(
        results_root,
        {
            "raw_dir": str(raw_dir),
            "benchmark": config.to_dict(),
            "scientific_scope": (
                "Common MNIST expressivity benchmark. Qubit count and PCA "
                "input dimension co-vary under position encoding."
            ),
        },
        run_id=run_id,
    )
    arrays = load_mnist(raw_dir)
    split = prepare_fixed_split(arrays, config)
    np.savez_compressed(
        run_dir / "sample_indices.npz",
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        y_train=split["y_train"],
        y_test=split["y_test"],
    )

    rows: list[dict[str, object]] = []
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for n_atoms in config.atom_counts:
        row, artifacts = benchmark_atom_count(split, int(n_atoms), config)
        rows.append(row)
        np.savez_compressed(
            feature_dir / f"n_atoms_{int(n_atoms)}.npz", **artifacts
        )

    metrics = pd.DataFrame(rows).sort_values("n_atoms").reset_index(drop=True)
    metrics.to_csv(run_dir / "mnist_metrics.csv", index=False)
    plots = _plot_metrics(metrics, run_dir / "plots")
    summary = {
        "schema_version": 1,
        "status": "mnist_position_benchmark_complete",
        "dataset": "MNIST official train/test split",
        "test_rows_are_mnist_test_rows": True,
        "financial_final_test_touched": False,
        "config": config.to_dict(),
        "best_reservoir_accuracy": float(metrics["reservoir_accuracy"].max()),
        "best_atom_count": int(
            metrics.loc[metrics["reservoir_accuracy"].idxmax(), "n_atoms"]
        ),
        "plots": plots,
        "known_limitations": [
            "Position encoding couples N atoms to N-1 PCA dimensions.",
            "A deterministic subset is used for exact-state feasibility.",
            "Interactions-off is a mechanism control, not a competitive baseline.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir
