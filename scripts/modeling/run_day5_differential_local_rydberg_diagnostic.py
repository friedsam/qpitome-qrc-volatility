#!/usr/bin/env python3
"""Mechanism diagnostic for differential-pair local-detuning encoding.

The outer market evaluation is intentionally not touched here. Synthetic
five-dimensional states are encoded into ten atoms, one complementary pair per
coordinate. Linear held-out readouts test whether the Rydberg map can represent
candidate nonlinear structures that matter for the day-5 branch problem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.qrc.local_detuning_reservoir import (
    LocalDetuningConfig,
    build_local_detuning_feature_matrix,
)
from qpitome_qrc.qrc.rydberg_reservoir import RydbergQRCConfig

N_COORDINATES = 5
PAIR_ATOMS = 2 * N_COORDINATES


def differential_pair_patterns(z: np.ndarray) -> np.ndarray:
    """Encode each signed coordinate in one complementary atom pair."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != N_COORDINATES:
        raise ValueError(f"Expected (samples, {N_COORDINATES}), got {z.shape}")
    positive = 0.5 * (1.0 + np.tanh(z))
    negative = 1.0 - positive
    patterns = np.empty((len(z), PAIR_ATOMS), dtype=float)
    patterns[:, 0::2] = positive
    patterns[:, 1::2] = negative
    return patterns


def diagnostic_config() -> LocalDetuningConfig:
    reservoir = RydbergQRCConfig(
        geometry="chain",
        chain_atoms=PAIR_ATOMS,
        chain_spacing_um=7.5,
        observable_mode="n_nn",
        collect_anchor_features=False,
        memory_mode="memoryless",
        shots=None,
    )
    return LocalDetuningConfig(
        reservoir=reservoir,
        evolution_time_us=0.55,
        global_omega_rad_us=6.0,
        global_delta_rad_us=6.0,
        local_delta_rad_us=4.0,
    )


def synthetic_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """Correlated standardized states with moderate tails."""
    covariance = np.array(
        [
            [1.0, -0.25, 0.20, 0.35, -0.20],
            [-0.25, 1.0, -0.40, -0.15, 0.25],
            [0.20, -0.40, 1.0, 0.30, -0.25],
            [0.35, -0.15, 0.30, 1.0, -0.45],
            [-0.20, 0.25, -0.25, -0.45, 1.0],
        ]
    )
    z = rng.multivariate_normal(np.zeros(N_COORDINATES), covariance, size=n)
    return np.clip(z, -3.0, 3.0)


def targets(z: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate structures motivated by the successful static probe."""
    out: dict[str, np.ndarray] = {}
    for index in range(N_COORDINATES):
        out[f"linear_z{index}"] = z[:, index]
        out[f"square_z{index}"] = z[:, index] ** 2 - 1.0
    for left in range(N_COORDINATES - 1):
        for right in range(left + 1, N_COORDINATES):
            out[f"product_z{left}_z{right}"] = z[:, left] * z[:, right]
    out["extrema_asymmetry"] = np.tanh(z[:, 3] - z[:, 4])
    out["endpoint_barrier_interaction"] = np.tanh(z[:, 0] * (z[:, 1] - z[:, 2]))
    out["closest_barrier_softmin"] = -np.logaddexp(-z[:, 3], -z[:, 4])
    out["mixed_state_score"] = np.tanh(
        0.7 * z[:, 0]
        - 0.4 * z[:, 1]
        + 0.3 * z[:, 2]
        + 0.8 * (z[:, 3] - z[:, 4])
        + 0.35 * z[:, 0] * z[:, 3]
    )
    return out


def random_tanh_features(
    train_z: np.ndarray,
    test_z: np.ndarray,
    seed: int = 20260711,
    width: int = 55,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical control matched approximately to 55 Rydberg observables."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / np.sqrt(N_COORDINATES), size=(N_COORDINATES, width))
    bias = rng.uniform(-1.0, 1.0, size=width)
    return np.tanh(train_z @ weights + bias), np.tanh(test_z @ weights + bias)


def heldout_capacity(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    alpha: float,
) -> float:
    scaler = StandardScaler().fit(train_features)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_features), train_target)
    prediction = model.predict(scaler.transform(test_features))
    variance = float(np.var(test_target))
    if variance <= 1e-12:
        return 0.0
    return float(1.0 - np.mean((prediction - test_target) ** 2) / variance)


def effective_rank(features: np.ndarray) -> dict[str, float]:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 0:
        return {"participation_ratio": 0.0, "n95": 0}
    participation = total**2 / float(np.sum(variance**2))
    cumulative = np.cumsum(variance) / total
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return {"participation_ratio": float(participation), "n95": n95}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/tmp/qpitome_day5_differential_rydberg_diagnostic"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_z = synthetic_states(args.n_train, rng)
    test_z = synthetic_states(args.n_test, rng)

    config = diagnostic_config()
    train_patterns = differential_pair_patterns(train_z)
    test_patterns = differential_pair_patterns(test_z)
    train_rydberg = build_local_detuning_feature_matrix(train_patterns, config)
    test_rydberg = build_local_detuning_feature_matrix(test_patterns, config)
    train_random, test_random = random_tanh_features(train_z, test_z)

    feature_sets = {
        "linear_raw": (train_z, test_z),
        "random_tanh_55": (train_random, test_random),
        "differential_local_rydberg": (train_rydberg, test_rydberg),
    }
    train_targets = targets(train_z)
    test_targets = targets(test_z)

    rows = []
    for feature_name, (train_features, test_features) in feature_sets.items():
        for target_name, train_target in train_targets.items():
            capacity = heldout_capacity(
                train_features,
                train_target,
                test_features,
                test_targets[target_name],
                args.ridge_alpha,
            )
            target_class = (
                "linear"
                if target_name.startswith("linear_")
                else "square"
                if target_name.startswith("square_")
                else "product"
                if target_name.startswith("product_")
                else "branch_motivated"
            )
            rows.append(
                {
                    "feature_map": feature_name,
                    "target": target_name,
                    "target_class": target_class,
                    "capacity": capacity,
                }
            )

    capacities = pd.DataFrame(rows)
    class_summary = (
        capacities.groupby(["feature_map", "target_class"], as_index=False)["capacity"]
        .mean()
        .sort_values(["target_class", "capacity"], ascending=[True, False])
    )
    rank_rows = []
    for feature_name, (train_features, _) in feature_sets.items():
        rank_rows.append({"feature_map": feature_name, **effective_rank(train_features)})
    ranks = pd.DataFrame(rank_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    capacities.to_csv(args.outdir / "target_capacities.csv", index=False)
    class_summary.to_csv(args.outdir / "capacity_by_class.csv", index=False)
    ranks.to_csv(args.outdir / "feature_rank.csv", index=False)
    manifest = {
        "purpose": "Pre-market mechanism diagnostic for differential-pair local detuning",
        "outer_market_evaluation_touched": False,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "encoding": "five complementary atom pairs",
        "feature_dimensions": {
            name: int(features[0].shape[1]) for name, features in feature_sets.items()
        },
        "rydberg_config": {
            "evolution_time_us": config.evolution_time_us,
            "global_omega_rad_us": config.global_omega_rad_us,
            "global_delta_rad_us": config.global_delta_rad_us,
            "local_delta_rad_us": config.local_delta_rad_us,
            "reservoir": config.reservoir.__dict__,
        },
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print("\nMean held-out capacity by target class")
    print(class_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFeature effective rank")
    print(ranks.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved diagnostic outputs to {args.outdir}")


if __name__ == "__main__":
    main()
