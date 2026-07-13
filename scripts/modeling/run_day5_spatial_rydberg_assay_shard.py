#!/usr/bin/env python3
"""Run one deterministic shard of the standardized day-5 spatial Rydberg assay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.logistic_offset import logit
from qpitome_qrc.day5.features import split_blocks
from qpitome_qrc.day5.protocol import (
    D1,
    EVAL_START,
    MIN_TRAIN,
    STATIC,
    add_extrema,
    differential_patterns,
    eligible_rows,
    load_frame,
    rydberg_config,
)
from qpitome_qrc.evaluation.binary import fit_offset_predict, logistic_pipeline
from qpitome_qrc.qrc.local_detuning_reservoir import build_local_detuning_feature_matrix

RNG_SEED = 20260711
JOINT_CS = (0.001, 0.01, 0.1)
OFFSET_L2 = (10.0, 100.0, 1000.0)
PCA_RANKS = (3, 5, 9)


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


def pca_block(H_train: np.ndarray, H_test: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(H_train)
    train_scaled = scaler.transform(H_train)
    test_scaled = scaler.transform(H_test)
    actual_rank = min(rank, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=actual_rank, svd_solver="full").fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)


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
