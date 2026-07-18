from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from baselines.numpy_esn import fit_continuous_ridge_scores, make_esn_weights


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.exp(np.clip(error, -30.0, 30.0)) - error - 1.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pooled_esn_features(
    sequences: np.ndarray,
    *,
    seed: int,
    reservoir_size: int = 300,
    spectral_radius: float = 0.9,
    input_scale: float = 0.3,
    leak: float = 0.5,
    connectivity: float = 0.02,
    washout: int = 10,
) -> np.ndarray:
    w_in, w = make_esn_weights(
        n_inputs=sequences.shape[2],
        n_reservoir=reservoir_size,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        seed=seed,
        connectivity=connectivity,
    )
    features = np.empty((len(sequences), reservoir_size * 3), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        state = np.zeros(reservoir_size, dtype=float)
        kept = []
        for t, value in enumerate(sequence):
            candidate = np.tanh(w_in @ value + w @ state)
            state = (1.0 - leak) * state + leak * candidate
            if t >= washout:
                kept.append(state.copy())
        states = np.asarray(kept)
        features[i] = np.concatenate([states[-1], states.mean(axis=0), states.std(axis=0)])
    return features


def shuffled_cross_market(x: np.ndarray, seed: int) -> np.ndarray:
    out = x.copy()
    rng = np.random.default_rng(seed)
    for i in range(len(out)):
        order = rng.permutation(out.shape[1])
        out[i, :, 2:8] = out[i, order, 2:8]
    return out


def target_matrix(manifest: pd.DataFrame) -> np.ndarray:
    columns = [f"target_x_h{i}" for i in range(1, 11)]
    missing = [column for column in columns if column not in manifest]
    if missing:
        raise ValueError(f"missing target columns: {missing}")
    return manifest[columns].to_numpy(dtype=float)


def evaluate_ridge_grid(
    model: str,
    fold: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: list[float],
    seed: int | None = None,
) -> list[dict[str, object]]:
    rows = []
    for alpha in alpha_grid:
        if model.endswith("esn") or model.endswith("shuffled"):
            pred = fit_continuous_ridge_scores(
                x_train,
                y_train,
                {"val": x_val},
                alpha,
            )["val"]
        else:
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train, y_train)
            pred = ridge.predict(x_val)
        rows.append({
            "fold": fold,
            "model": model,
            "seed": seed,
            "alpha": alpha,
            "val_qlike": qlike(y_val, pred),
            "val_rmse": rmse(y_val, pred),
            "val_samples": len(y_val),
        })
    return rows


def run_comparison(
    *,
    sample_manifest: Path,
    rolling_manifest: Path,
    tensor_run: Path,
    run_dir: Path,
    seeds: list[int],
    ridge_alphas: list[float],
    esn_alphas: list[float],
) -> dict[str, object]:
    manifest = pd.read_csv(sample_manifest).reset_index(drop=True)
    rolling = pd.read_csv(rolling_manifest)
    y = target_matrix(manifest)
    sample_to_index = {str(sample_id): i for i, sample_id in enumerate(manifest["sample_id"].astype(str))}
    rows: list[dict[str, object]] = []

    for fold in sorted(rolling["fold"].unique()):
        with np.load(tensor_run / f"fold_{int(fold)}" / "compact_tensor.npz", allow_pickle=True) as data:
            x = np.asarray(data["X"], dtype=float)
            valid = np.asarray(data["valid"], dtype=bool)
        fold_rows = rolling.loc[rolling["fold"].eq(fold)]
        train_ids = fold_rows.loc[fold_rows["fold_split"].eq("train"), "sample_id"].astype(str)
        val_ids = fold_rows.loc[fold_rows["fold_split"].eq("val"), "sample_id"].astype(str)
        train_idx = np.array([sample_to_index[s] for s in train_ids if s in sample_to_index and valid[sample_to_index[s]]])
        val_idx = np.array([sample_to_index[s] for s in val_ids if s in sample_to_index and valid[sample_to_index[s]]])
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"fold {fold} has no valid train or validation samples")

        x_uni_train = x[train_idx, :, :2].reshape(len(train_idx), -1)
        x_uni_val = x[val_idx, :, :2].reshape(len(val_idx), -1)
        x_cross_train = x[train_idx].reshape(len(train_idx), -1)
        x_cross_val = x[val_idx].reshape(len(val_idx), -1)
        rows.extend(evaluate_ridge_grid("univariate_sequence_ridge", int(fold), x_uni_train, y[train_idx], x_uni_val, y[val_idx], ridge_alphas))
        rows.extend(evaluate_ridge_grid("cross_market_sequence_ridge", int(fold), x_cross_train, y[train_idx], x_cross_val, y[val_idx], ridge_alphas))

        for seed in seeds:
            print(f"fold={int(fold)} seed={seed} ESN univariate", flush=True)
            f_uni_train = pooled_esn_features(x[train_idx, :, :2], seed=seed)
            f_uni_val = pooled_esn_features(x[val_idx, :, :2], seed=seed)
            rows.extend(evaluate_ridge_grid("univariate_esn", int(fold), f_uni_train, y[train_idx], f_uni_val, y[val_idx], esn_alphas, seed))

            print(f"fold={int(fold)} seed={seed} ESN cross-market", flush=True)
            f_cross_train = pooled_esn_features(x[train_idx], seed=seed)
            f_cross_val = pooled_esn_features(x[val_idx], seed=seed)
            rows.extend(evaluate_ridge_grid("cross_market_esn", int(fold), f_cross_train, y[train_idx], f_cross_val, y[val_idx], esn_alphas, seed))

            print(f"fold={int(fold)} seed={seed} ESN shuffled control", flush=True)
            shuffled_train = shuffled_cross_market(x[train_idx], seed)
            shuffled_val = shuffled_cross_market(x[val_idx], seed + 1000)
            f_shuf_train = pooled_esn_features(shuffled_train, seed=seed)
            f_shuf_val = pooled_esn_features(shuffled_val, seed=seed)
            rows.extend(evaluate_ridge_grid("cross_market_esn_shuffled", int(fold), f_shuf_train, y[train_idx], f_shuf_val, y[val_idx], esn_alphas, seed))

    detailed = pd.DataFrame(rows)
    detailed.to_csv(run_dir / "detailed_results.csv", index=False)
    best_per_fold = detailed.sort_values("val_qlike").groupby(["fold", "model", "seed"], dropna=False, as_index=False).first()
    best_per_fold.to_csv(run_dir / "best_per_fold.csv", index=False)
    aggregate = best_per_fold.groupby("model", as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
        runs=("val_qlike", "size"),
    ).sort_values("mean_val_qlike")
    aggregate.to_csv(run_dir / "aggregate_results.csv", index=False)
    summary = {
        "test_evaluated": False,
        "best_model": aggregate.iloc[0].to_dict(),
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
