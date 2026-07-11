#!/usr/bin/env python3
"""Run one deterministic shard of the standardized day-5 spatial Rydberg assay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.logistic_offset import fit_offset_logistic, logit, sigmoid
from qpitome_qrc.qrc.local_detuning_reservoir import LocalDetuningConfig, build_local_detuning_feature_matrix
from qpitome_qrc.qrc.rydberg_reservoir import RydbergQRCConfig

MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
RNG_SEED = 20260711
D1 = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "barrier_width",
]
STATIC = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "closest_to_relapse",
    "closest_to_recovery",
]
JOINT_CS = (0.001, 0.01, 0.1)
OFFSET_L2 = (10.0, 100.0, 1000.0)
PCA_RANKS = (3, 5, 9)


def add_extrema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    running = np.column_stack([out[f"r_d{day}"].to_numpy(float) for day in range(1, 6)])
    endpoint = out["current_return_5d_from_branch"].to_numpy(float)
    lower = endpoint - out["distance_to_relapse_barrier"].to_numpy(float)
    upper = endpoint + out["distance_to_recovery_barrier"].to_numpy(float)
    out["closest_to_relapse"] = running.min(axis=1) - lower
    out["closest_to_recovery"] = upper - running.max(axis=1)
    return out


def load_frame(path_panel: Path, clusters_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path_panel, parse_dates=["branch_date", "landmark_date"])
    clusters = pd.read_csv(clusters_path)[["market_key", "episode_id", "cluster_id"]].drop_duplicates()
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left", validate="many_to_one")
    if frame["cluster_id"].isna().any():
        raise ValueError("path-panel rows are missing cluster ids")
    starts = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(starts, on="cluster_id", how="left", validate="many_to_one")
    frame = add_extrema(frame)
    needed = D1 + STATIC + [f"r_d{d}" for d in range(1, 6)] + ["y_recovery", "landmark_date", "cluster_start"]
    frame = frame.dropna(subset=needed).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)
    if not np.allclose(frame["r_d5"].to_numpy(float), frame["current_return_5d_from_branch"].to_numpy(float), atol=1e-12, rtol=0):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")
    return frame


def logistic_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")),
    ])


def differential_patterns(train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(train_X)
    z = scaler.transform(X)
    positive = 0.5 * (1.0 + np.tanh(z))
    negative = 1.0 - positive
    patterns = np.empty((len(z), 10), dtype=float)
    patterns[:, 0::2] = positive
    patterns[:, 1::2] = negative
    return patterns


def rydberg_config() -> LocalDetuningConfig:
    reservoir = RydbergQRCConfig(
        geometry="chain",
        chain_atoms=10,
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


def split_blocks(features: np.ndarray) -> dict[str, np.ndarray]:
    occ = features[:, :10]
    raw_pairs = features[:, 10:]
    connected = np.empty_like(raw_pairs)
    k = 0
    for i in range(9):
        for j in range(i + 1, 10):
            connected[:, k] = raw_pairs[:, k] - occ[:, i] * occ[:, j]
            k += 1
    return {
        "occupations": occ,
        "raw_pairs": raw_pairs,
        "connected_pairs": connected,
        "all_raw": features,
        "occ_plus_connected": np.column_stack([occ, connected]),
    }


def feature_diagnostics(X: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 1e-15:
        return {
            "n_features": int(X.shape[1]),
            "participation_ratio": 0.0,
            "n95": 0,
            "condition_number": np.inf,
            "near_constant": int(X.shape[1]),
        }
    participation = total**2 / float(np.sum(variance**2))
    n95 = int(np.searchsorted(np.cumsum(variance) / total, 0.95) + 1)
    nonzero = singular[singular > 1e-12]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else np.inf
    near_constant = int(np.sum(np.std(X, axis=0) < 1e-8))
    return {
        "n_features": int(X.shape[1]),
        "participation_ratio": float(participation),
        "n95": n95,
        "condition_number": condition,
        "near_constant": near_constant,
    }


def fit_joint_predict(d1_train: np.ndarray, H_train: np.ndarray, y: np.ndarray, d1_test: np.ndarray, H_test: np.ndarray, C: float) -> float:
    model = logistic_pipeline(C)
    model.fit(np.column_stack([d1_train, H_train]), y)
    return float(model.predict_proba(np.column_stack([d1_test, H_test]))[0, 1])


def fit_rydberg_only_predict(H_train: np.ndarray, y: np.ndarray, H_test: np.ndarray, C: float) -> float:
    model = logistic_pipeline(C)
    model.fit(H_train, y)
    return float(model.predict_proba(H_test)[0, 1])


def fit_offset_predict(H_train: np.ndarray, y: np.ndarray, H_test: np.ndarray, offset_train: np.ndarray, offset_test: float, l2: float) -> float:
    scaler = StandardScaler().fit(H_train)
    X_train = scaler.transform(H_train)
    X_test = scaler.transform(H_test)
    beta, intercept = fit_offset_logistic(X_train, y, offset_train, l2=l2)
    return float(sigmoid(np.array([offset_test + intercept + X_test[0] @ beta]))[0])


def pca_block(H_train: np.ndarray, H_test: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(H_train)
    train_scaled = scaler.transform(H_train)
    test_scaled = scaler.transform(H_test)
    actual_rank = min(rank, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=actual_rank, svd_solver="full").fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)


def eligible_rows(frame: pd.DataFrame) -> list[int]:
    result = []
    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) >= MIN_TRAIN and train["y_recovery"].nunique() >= 2:
            result.append(i)
    return result


def evaluate_shard(frame: pd.DataFrame, shard_index: int, num_shards: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = rydberg_config()
    eligible = eligible_rows(frame)
    selected = [row_i for position, row_i in enumerate(eligible) if position % num_shards == shard_index]
    rows: list[dict] = []
    diagnostics: list[dict] = []

    for count, i in enumerate(selected, start=1):
        row = frame.iloc[i]
        train_idx = np.flatnonzero((frame["landmark_date"] < row["cluster_start"]).to_numpy())
        train = frame.iloc[train_idx]
        y_train = train["y_recovery"].to_numpy(int)
        d1_train = train[D1].to_numpy(float)
        d1_test = frame.loc[[i], D1].to_numpy(float)

        d1_model = logistic_pipeline(1.0)
        d1_model.fit(d1_train, y_train)
        p_train_d1 = d1_model.predict_proba(d1_train)[:, 1]
        p_test_d1 = float(d1_model.predict_proba(d1_test)[0, 1])
        offset_train = logit(p_train_d1)
        offset_test = float(logit(p_test_d1))

        fold_idx = np.concatenate([train_idx, np.array([i], dtype=int)])
        patterns = differential_patterns(train[STATIC].to_numpy(float), frame.iloc[fold_idx][STATIC].to_numpy(float))
        features = build_local_detuning_feature_matrix(patterns, config)
        blocks = split_blocks(features)

        pred: dict[str, object] = {
            "row_id": int(i),
            "market_key": row["market_key"],
            "episode_id": int(row["episode_id"]),
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D1": p_test_d1,
            "shard_index": shard_index,
        }

        for block_name, block in blocks.items():
            H_train, H_test = block[:-1], block[-1:]
            diag = feature_diagnostics(H_train)
            diagnostics.append({"row_id": int(i), "block": block_name, **diag})
            for C in JOINT_CS:
                tag = str(C).replace(".", "p")
                pred[f"joint_{block_name}_C{tag}"] = fit_joint_predict(d1_train, H_train, y_train, d1_test, H_test, C)
                pred[f"rydberg_only_{block_name}_C{tag}"] = fit_rydberg_only_predict(H_train, y_train, H_test, C)
            for l2 in OFFSET_L2:
                pred[f"offset_{block_name}_l2{int(l2)}"] = fit_offset_predict(H_train, y_train, H_test, offset_train, offset_test, l2)

        base_train, base_test = blocks["occ_plus_connected"][:-1], blocks["occ_plus_connected"][-1:]
        for rank in PCA_RANKS:
            H_train, H_test = pca_block(base_train, base_test, rank)
            for C in JOINT_CS:
                tag = str(C).replace(".", "p")
                pred[f"joint_pca{rank}_C{tag}"] = fit_joint_predict(d1_train, H_train, y_train, d1_test, H_test, C)
                pred[f"rydberg_only_pca{rank}_C{tag}"] = fit_rydberg_only_predict(H_train, y_train, H_test, C)
            for l2 in OFFSET_L2:
                pred[f"offset_pca{rank}_l2{int(l2)}"] = fit_offset_predict(H_train, y_train, H_test, offset_train, offset_test, l2)

        rng = np.random.default_rng(RNG_SEED + int(i))
        gaussian_train = rng.normal(size=(len(train), 55))
        gaussian_test = rng.normal(size=(1, 55))
        duplicate_train = StandardScaler().fit_transform(d1_train)
        duplicate_test = StandardScaler().fit(d1_train).transform(d1_test)
        perm = rng.permutation(len(train))
        permuted_train = blocks["all_raw"][:-1][perm]
        permuted_test = blocks["all_raw"][-1:]
        constant_train = np.ones((len(train), 55))
        constant_test = np.ones((1, 55))

        for null_name, (H_train, H_test) in {
            "gaussian55": (gaussian_train, gaussian_test),
            "duplicate_d1": (duplicate_train, duplicate_test),
            "permuted_all_raw": (permuted_train, permuted_test),
            "constant55": (constant_train, constant_test),
        }.items():
            for C in JOINT_CS:
                tag = str(C).replace(".", "p")
                pred[f"joint_{null_name}_C{tag}"] = fit_joint_predict(d1_train, H_train, y_train, d1_test, H_test, C)
            if null_name != "constant55":
                for l2 in OFFSET_L2:
                    pred[f"offset_{null_name}_l2{int(l2)}"] = fit_offset_predict(H_train, y_train, H_test, offset_train, offset_test, l2)

        rows.append(pred)
        print(f"shard {shard_index}/{num_shards}: {count}/{len(selected)} row_id={i}", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-panel", type=Path, default=Path("scratch/path_panel_day0_5.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    frame = load_frame(args.path_panel, args.clusters)
    predictions, diagnostics = evaluate_shard(frame, args.shard_index, args.num_shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / f"predictions_shard_{args.shard_index}.csv", index=False)
    diagnostics.to_csv(args.outdir / f"feature_diagnostics_shard_{args.shard_index}.csv", index=False)
    manifest = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_predictions": int(len(predictions)),
        "joint_C": JOINT_CS,
        "offset_l2": OFFSET_L2,
        "pca_ranks": PCA_RANKS,
        "rydberg_config": rydberg_config().__dict__,
    }
    (args.outdir / f"manifest_shard_{args.shard_index}.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved shard {args.shard_index} to {args.outdir}")


if __name__ == "__main__":
    main()
