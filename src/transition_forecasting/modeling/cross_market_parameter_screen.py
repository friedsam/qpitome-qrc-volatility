from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import pool_trajectory, reservoir_trajectory
from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling.cross_market_comparison import (
    build_hybrid_sequences,
    build_original_enriched_sequences,
    mz_calibration,
    scale_sequences_train_only,
    shuffle_cross_market_channels,
)
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    load_stage_d_run,
    qlike_loss,
)

ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
CONFIGS = (
    {"id": "current", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.50},
    {"id": "less_sparse", "n": 300, "conn": 0.05, "sr": 0.90, "inp": 0.30, "leak": 0.50},
    {"id": "baseline_density", "n": 300, "conn": 0.10, "sr": 0.90, "inp": 0.30, "leak": 0.50},
    {"id": "lower_radius", "n": 300, "conn": 0.02, "sr": 0.75, "inp": 0.30, "leak": 0.50},
    {"id": "weaker_input", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.10, "leak": 0.50},
    {"id": "faster_leak", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.80},
)


def _har_prediction(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _features(sequences: np.ndarray, config: dict[str, object], seed: int) -> np.ndarray:
    w_in, w = make_esn_weights(
        n_inputs=sequences.shape[-1],
        n_reservoir=int(config["n"]),
        spectral_radius=float(config["sr"]),
        input_scale=float(config["inp"]),
        seed=int(seed),
        connectivity=float(config["conn"]),
    )
    states = reservoir_trajectory(sequences, w_in, w, float(config["leak"]))
    return pool_trajectory(states, "final_mean_std", washout=10)


def _evaluate_features(
    *,
    fold: int,
    config: dict[str, object],
    seed: int,
    features: np.ndarray,
    y: np.ndarray,
    har: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    order: str,
    alphas: tuple[float, ...],
) -> list[dict[str, object]]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[train_mask])
    val_x = scaler.transform(features[val_mask])
    residual = y[train_mask] - har[train_mask]
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(train_x, residual)
        prediction = har[val_mask] + model.predict(val_x)
        y_val = y[val_mask]
        mz_alpha, mz_beta, mz_r2 = mz_calibration(y_val, prediction)
        rows.append({
            "config_id": config["id"],
            "order": order,
            "fold": int(fold),
            "seed": int(seed),
            "alpha": float(alpha),
            "val_qlike": float(qlike_loss(y_val, prediction).mean()),
            "val_rmse": float(np.sqrt(np.mean((y_val - prediction) ** 2))),
            "mz_alpha": mz_alpha,
            "mz_beta": mz_beta,
            "mz_r2": mz_r2,
            **{key: config[key] for key in ("n", "conn", "sr", "inp", "leak")},
        })
    return rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = (
        results.sort_values(["config_id", "order", "fold", "seed", "val_qlike", "val_rmse"])
        .groupby(["config_id", "order", "fold", "seed"], as_index=False)
        .first()
    )
    leaderboard = best.groupby(
        ["config_id", "order", "n", "conn", "sr", "inp", "leak"], as_index=False
    ).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
        mean_mz_alpha=("mz_alpha", "mean"),
        mean_mz_beta=("mz_beta", "mean"),
        mean_mz_r2=("mz_r2", "mean"),
        best_alpha_median=("alpha", "median"),
    ).sort_values(["mean_val_qlike", "mean_val_rmse"])
    return best, leaderboard


def run_screen(
    *,
    stage_d_run: Path,
    rolling_manifest: Path,
    tensor_run: Path,
    run_dir: Path,
    seeds: tuple[int, ...],
    max_configs: int,
) -> dict[str, object]:
    stage_d = load_stage_d_run(stage_d_run)
    manifest = stage_d.manifest.reset_index(drop=True)
    original = np.asarray(stage_d.sequences, dtype=float)
    rolling = pd.read_csv(rolling_manifest)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    configs = CONFIGS[: max(1, min(max_configs, len(CONFIGS)))]
    frames: list[pd.DataFrame] = []

    for fold in sorted(rolling["fold"].unique()):
        fold_rows = rolling.loc[rolling["fold"].eq(fold)].copy()
        split_map = fold_rows.set_index(fold_rows["sample_id"].astype(str))["fold_split"]
        fold_manifest = manifest.copy()
        fold_manifest["fold_split"] = [split_map.loc[sample_id] for sample_id in sample_ids]
        with np.load(tensor_run / f"fold_{int(fold)}" / "compact_tensor.npz", allow_pickle=True) as data:
            compact = np.asarray(data["X"], dtype=float)
            compact_ids = data["sample_id"].astype(str)
            valid = np.asarray(data["valid"], dtype=bool)
        if not np.array_equal(compact_ids, sample_ids):
            raise ValueError(f"fold {fold} compact tensor is not aligned with Stage D")

        train_mask = fold_manifest["fold_split"].eq("train").to_numpy() & valid
        val_mask = fold_manifest["fold_split"].eq("val").to_numpy() & valid
        y = fold_manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _har_prediction(fold_manifest, y, train_mask)
        original_scaled = scale_sequences_train_only(original, train_mask)
        hybrid = build_hybrid_sequences(build_original_enriched_sequences(original_scaled), compact)

        for config in configs:
            print(f"fold={fold} config={config['id']}", flush=True)
            for seed in seeds:
                ordered = _features(hybrid, config, seed)
                rows = _evaluate_features(
                    fold=int(fold), config=config, seed=seed, features=ordered, y=y, har=har,
                    train_mask=train_mask, val_mask=val_mask, order="ordered", alphas=ALPHAS,
                )
                frames.append(pd.DataFrame(rows))
                best = min(rows, key=lambda row: (row["val_qlike"], row["val_rmse"]))
                print(
                    f"  seed={seed} alpha={best['alpha']:g} qlike={best['val_qlike']:.6f} "
                    f"rmse={best['val_rmse']:.6f}", flush=True,
                )

                if config["id"] == "current":
                    shuffled = _features(shuffle_cross_market_channels(hybrid, seed + 30000), config, seed)
                    frames.append(pd.DataFrame(_evaluate_features(
                        fold=int(fold), config=config, seed=seed, features=shuffled, y=y, har=har,
                        train_mask=train_mask, val_mask=val_mask, order="shuffled", alphas=ALPHAS,
                    )))

            current = pd.concat(frames, ignore_index=True)
            current.to_csv(run_dir / "market_parameter_results_live.csv", index=False)
            _, live = summarize(current)
            live.to_csv(run_dir / "market_parameter_leaderboard_live.csv", index=False)

    results = pd.concat(frames, ignore_index=True)
    best, leaderboard = summarize(results)
    results.to_csv(run_dir / "market_parameter_results.csv", index=False)
    best.to_csv(run_dir / "market_parameter_best_by_seed_fold.csv", index=False)
    leaderboard.to_csv(run_dir / "market_parameter_leaderboard.csv", index=False)
    payload = {
        "test_evaluated": False,
        "input": "original Stage D level/difference/time plus six compact cross-market channels",
        "pooling": "final_mean_std",
        "washout": 10,
        "configs_run": len(configs),
        "seeds": list(seeds),
        "shuffled_control": "current configuration only",
        "best": leaderboard.iloc[0].to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nFINAL LEADERBOARD", flush=True)
    print(leaderboard.to_string(index=False), flush=True)
    return payload
