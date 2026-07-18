from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import pool_trajectory, reservoir_trajectory, transform_input
from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling.cross_market_comparison import mz_calibration
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    load_stage_d_run,
    qlike_loss,
)

ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
REFINEMENT_ALPHAS = (300.0, 1000.0, 3000.0, 10000.0)
CONFIGS = (
    {"id": "baseline", "n": 300, "conn": 0.10, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "very_sparse", "n": 300, "conn": 0.005, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "sparse", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "mid_sparse", "n": 300, "conn": 0.05, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "dense", "n": 300, "conn": 0.20, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "short_memory", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.80},
    {"id": "moderate_memory", "n": 300, "conn": 0.02, "sr": 0.75, "inp": 0.30, "leak": 0.50},
    {"id": "near_critical_sparse", "n": 300, "conn": 0.02, "sr": 1.05, "inp": 0.30, "leak": 0.30},
    {"id": "weak_input", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.10, "leak": 0.30},
    {"id": "strong_input", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.70, "leak": 0.30},
    {"id": "small_sparse", "n": 150, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "large_sparse", "n": 500, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
)
REFINEMENT_CONFIGS = (
    {"id": "short_ref", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.80},
    {"id": "short_sr040", "n": 300, "conn": 0.02, "sr": 0.40, "inp": 0.30, "leak": 0.80},
    {"id": "short_sr070", "n": 300, "conn": 0.02, "sr": 0.70, "inp": 0.30, "leak": 0.80},
    {"id": "short_leak065", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.65},
    {"id": "short_leak095", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.95},
    {"id": "short_inp020", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"id": "short_inp045", "n": 300, "conn": 0.02, "sr": 0.55, "inp": 0.45, "leak": 0.80},
    {"id": "short_small", "n": 150, "conn": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.80},
    {"id": "compact_ref", "n": 150, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "compact_n100", "n": 100, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "compact_n200", "n": 200, "conn": 0.02, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "compact_conn001", "n": 150, "conn": 0.01, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "compact_conn005", "n": 150, "conn": 0.05, "sr": 0.90, "inp": 0.30, "leak": 0.30},
    {"id": "strong_ref", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.70, "leak": 0.30},
    {"id": "strong_inp045", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.45, "leak": 0.30},
    {"id": "strong_inp090", "n": 300, "conn": 0.02, "sr": 0.90, "inp": 0.90, "leak": 0.30},
)


def _scale_channels(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = x[train_mask].reshape(-1, x.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x - mean[None, None, :]) / scale[None, None, :]


def _fit_har(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def evaluate_config(
    assignments: pd.DataFrame,
    sequences: np.ndarray,
    config: dict[str, float | int | str],
    *,
    seeds: tuple[int, ...],
    alphas: tuple[float, ...] = ALPHAS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in sorted(assignments["fold"].unique()):
        manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        train_mask = manifest["fold_split"].eq("train").to_numpy()
        val_mask = manifest["fold_split"].eq("val").to_numpy()
        y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(manifest, y, train_mask)
        residual = y - har
        inputs = _scale_channels(transform_input(sequences, "level_diff_time"), train_mask)

        for seed in seeds:
            w_in, w = make_esn_weights(
                n_inputs=inputs.shape[-1],
                n_reservoir=int(config["n"]),
                spectral_radius=float(config["sr"]),
                input_scale=float(config["inp"]),
                seed=int(seed),
                connectivity=float(config["conn"]),
            )
            states = reservoir_trajectory(inputs, w_in, w, float(config["leak"]))
            features = pool_trajectory(states, "final_mean_std", washout=10)
            scaler = StandardScaler()
            train_features = scaler.fit_transform(features[train_mask])
            val_features = scaler.transform(features[val_mask])

            seed_rows = []
            for alpha in alphas:
                model = Ridge(alpha=float(alpha))
                model.fit(train_features, residual[train_mask])
                prediction = har[val_mask] + model.predict(val_features)
                y_val = y[val_mask]
                qlike = float(qlike_loss(y_val, prediction).mean())
                rmse = float(np.sqrt(np.mean((y_val - prediction) ** 2)))
                mz_alpha, mz_beta, mz_r2 = mz_calibration(y_val, prediction)
                seed_rows.append({
                    "config_id": config["id"],
                    "fold": int(fold),
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "val_qlike": qlike,
                    "val_rmse": rmse,
                    "mz_alpha": mz_alpha,
                    "mz_beta": mz_beta,
                    "mz_r2": mz_r2,
                    **{key: config[key] for key in ("n", "conn", "sr", "inp", "leak")},
                })
            rows.extend(seed_rows)
            best = min(seed_rows, key=lambda row: (row["val_qlike"], row["val_rmse"]))
            print(
                f"[{config['id']}] fold={fold} seed={seed} best_alpha={best['alpha']:g} "
                f"qlike={best['val_qlike']:.6f} rmse={best['val_rmse']:.6f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_by_seed_fold = (
        results.sort_values(["config_id", "fold", "seed", "val_qlike", "val_rmse"])
        .groupby(["config_id", "fold", "seed"], as_index=False)
        .first()
    )
    leaderboard = best_by_seed_fold.groupby(
        ["config_id", "n", "conn", "sr", "inp", "leak"], as_index=False
    ).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
        mean_mz_alpha=("mz_alpha", "mean"),
        mean_mz_beta=("mz_beta", "mean"),
        mean_mz_r2=("mz_r2", "mean"),
        best_alpha_median=("alpha", "median"),
    ).sort_values(["mean_val_qlike", "mean_val_rmse"])
    return best_by_seed_fold, leaderboard


def run_screen(
    *,
    stage_d_run: Path,
    rolling_run: Path,
    run_dir: Path,
    seeds: tuple[int, ...],
    max_configs: int,
    profile: str = "initial",
) -> dict[str, object]:
    if profile not in {"initial", "refinement"}:
        raise ValueError(f"unknown profile: {profile}")
    available_configs = CONFIGS if profile == "initial" else REFINEMENT_CONFIGS
    alphas = ALPHAS if profile == "initial" else REFINEMENT_ALPHAS

    data = load_stage_d_run(stage_d_run)
    sequences = np.asarray(data.sequences, dtype=float)
    assignments = pd.read_csv(rolling_run / "rolling_fold_manifest.csv")
    selected_configs = available_configs[: max(1, min(max_configs, len(available_configs)))]
    all_results: list[pd.DataFrame] = []

    for index, config in enumerate(selected_configs, start=1):
        print(
            f"\n=== config {index}/{len(selected_configs)}: {config['id']} "
            f"n={config['n']} conn={config['conn']} sr={config['sr']} "
            f"inp={config['inp']} leak={config['leak']} ===",
            flush=True,
        )
        all_results.append(evaluate_config(assignments, sequences, config, seeds=seeds, alphas=alphas))
        current = pd.concat(all_results, ignore_index=True)
        current.to_csv(run_dir / "parameter_results_live.csv", index=False)
        _, live = summarize(current)
        live.to_csv(run_dir / "parameter_leaderboard_live.csv", index=False)
        leader = live.iloc[0]
        print(
            f"running leader: {leader['config_id']} qlike={leader['mean_val_qlike']:.6f} "
            f"rmse={leader['mean_val_rmse']:.6f}",
            flush=True,
        )

    results = pd.concat(all_results, ignore_index=True)
    best_by_seed_fold, leaderboard = summarize(results)
    results.to_csv(run_dir / "parameter_results.csv", index=False)
    best_by_seed_fold.to_csv(run_dir / "parameter_best_by_seed_fold.csv", index=False)
    leaderboard.to_csv(run_dir / "parameter_leaderboard.csv", index=False)
    payload = {
        "stage_d_run": str(stage_d_run),
        "rolling_run": str(rolling_run),
        "test_evaluated": False,
        "profile": profile,
        "representation": "level_diff_time",
        "pooling": "final_mean_std",
        "washout": 10,
        "alphas": list(alphas),
        "configs_run": len(selected_configs),
        "seeds": list(seeds),
        "best": leaderboard.iloc[0].to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nFINAL LEADERBOARD", flush=True)
    print(leaderboard.to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return payload
