from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.esn_representation import pool_trajectory, reservoir_trajectory, transform_input
from baselines.numpy_esn import make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

ROLLING_SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_rolling_origin", ROLLING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROLLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLLING)

ALPHAS = (10.0, 100.0, 1000.0, 10000.0)

# Curated, deliberately small screen. It probes connectivity first, then memory,
# input drive, and size without exploding into a full Cartesian grid.
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


def _scale_channels(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = X[train_mask].reshape(-1, X.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (X - mean[None, None, :]) / scale[None, None, :]


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
            all_features = scaler.transform(features)

            seed_rows = []
            for alpha in alphas:
                model = Ridge(alpha=float(alpha))
                model.fit(train_features, residual[train_mask])
                prediction = har + model.predict(all_features)
                qlike = float(qlike_loss(y[val_mask], prediction[val_mask]).mean())
                rmse = float(np.sqrt(np.mean((y[val_mask] - prediction[val_mask]) ** 2)))
                seed_rows.append({
                    "config_id": config["id"],
                    "fold": int(fold),
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "val_qlike": qlike,
                    "val_rmse": rmse,
                    **{key: config[key] for key in ("n", "conn", "sr", "inp", "leak")},
                })
            rows.extend(seed_rows)
            best = min(seed_rows, key=lambda row: (row["val_qlike"], row["val_rmse"]))
            print(
                f"[{config['id']}] fold={fold} seed={seed} "
                f"best_alpha={best['alpha']:g} qlike={best['val_qlike']:.6f} "
                f"rmse={best['val_rmse']:.6f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small live-print ESN parameter screen on the winning representation.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-configs", type=int, default=len(CONFIGS))
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/stage_e_esn_parameter_screen"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    _, sequences = ROLLING.load_stage_d(args.stage_d_run)
    assignments = pd.read_csv(args.rolling_run / "rolling_fold_manifest.csv")
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    selected_configs = CONFIGS[: max(1, min(args.max_configs, len(CONFIGS)))]
    all_results: list[pd.DataFrame] = []

    for index, config in enumerate(selected_configs, start=1):
        print(
            f"\n=== config {index}/{len(selected_configs)}: {config['id']} "
            f"n={config['n']} conn={config['conn']} sr={config['sr']} "
            f"inp={config['inp']} leak={config['leak']} ===",
            flush=True,
        )
        frame = evaluate_config(assignments, sequences, config, seeds=tuple(args.seeds))
        all_results.append(frame)
        current = pd.concat(all_results, ignore_index=True)
        current.to_csv(run_dir / "parameter_results_live.csv", index=False)

        best_by_seed_fold = (
            current.sort_values(["config_id", "fold", "seed", "val_qlike", "val_rmse"])
            .groupby(["config_id", "fold", "seed"], as_index=False)
            .first()
        )
        leaderboard = best_by_seed_fold.groupby("config_id", as_index=False).agg(
            mean_val_qlike=("val_qlike", "mean"),
            std_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
        ).sort_values("mean_val_qlike")
        leaderboard.to_csv(run_dir / "parameter_leaderboard_live.csv", index=False)
        leader = leaderboard.iloc[0]
        print(
            f"running leader: {leader['config_id']} "
            f"qlike={leader['mean_val_qlike']:.6f} rmse={leader['mean_val_rmse']:.6f}",
            flush=True,
        )

    results = pd.concat(all_results, ignore_index=True)
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
        best_alpha_median=("alpha", "median"),
    ).sort_values("mean_val_qlike")
    best_by_seed_fold.to_csv(run_dir / "parameter_best_by_seed_fold.csv", index=False)
    leaderboard.to_csv(run_dir / "parameter_leaderboard.csv", index=False)
    payload = {
        "stage_d_run": str(args.stage_d_run),
        "rolling_run": str(args.rolling_run),
        "test_evaluated": False,
        "representation": "level_diff_time",
        "pooling": "final_mean_std",
        "washout": 10,
        "configs_run": len(selected_configs),
        "seeds": list(args.seeds),
        "best": leaderboard.iloc[0].to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nFINAL LEADERBOARD", flush=True)
    print(leaderboard.to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
