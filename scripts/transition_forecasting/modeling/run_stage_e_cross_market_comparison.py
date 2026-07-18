from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from experiments.runs import begin_run


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.exp(np.clip(error, -30.0, 30.0)) - error - 1.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_val), dtype=float)


def make_esn_weights(
    input_dim: int,
    reservoir_size: int,
    seed: int,
    spectral_radius: float = 0.9,
    input_scale: float = 0.3,
    connectivity: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    win = rng.uniform(-input_scale, input_scale, size=(reservoir_size, input_dim))
    bias = rng.uniform(-input_scale, input_scale, size=reservoir_size)
    mask = rng.random((reservoir_size, reservoir_size)) < connectivity
    w = rng.normal(0.0, 1.0, size=(reservoir_size, reservoir_size)) * mask
    if not np.any(w):
        w[0, 0] = 1.0
    eigvals = np.linalg.eigvals(w)
    radius = float(np.max(np.abs(eigvals)))
    if radius <= 1e-12:
        w[0, 0] = 1.0
        radius = 1.0
    w *= spectral_radius / radius
    return win, w, bias


def esn_features(
    sequences: np.ndarray,
    *,
    seed: int,
    reservoir_size: int = 300,
    leak: float = 0.5,
    washout: int = 10,
) -> np.ndarray:
    win, w, bias = make_esn_weights(sequences.shape[2], reservoir_size, seed)
    features = np.empty((len(sequences), reservoir_size * 3), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        state = np.zeros(reservoir_size, dtype=float)
        kept = []
        for t, value in enumerate(sequence):
            candidate = np.tanh(win @ value + w @ state + bias)
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


def evaluate_model(
    model: str,
    fold: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha_grid: list[float],
    seed: int | None = None,
) -> list[dict[str, object]]:
    rows = []
    for alpha in alpha_grid:
        pred = ridge_fit_predict(x_train, y_train, x_val, alpha)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling validation comparison of univariate and compact cross-market ridge/ESN models.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--tensor-run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--esn-alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_cross_market_comparison"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    manifest = pd.read_csv(args.sample_manifest).reset_index(drop=True)
    rolling = pd.read_csv(args.rolling_manifest)
    y = target_matrix(manifest)
    sample_to_index = {str(sample_id): i for i, sample_id in enumerate(manifest["sample_id"].astype(str))}
    rows: list[dict[str, object]] = []

    for fold in sorted(rolling["fold"].unique()):
        with np.load(args.tensor_run / f"fold_{int(fold)}" / "compact_tensor.npz", allow_pickle=True) as data:
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
        rows.extend(evaluate_model("univariate_sequence_ridge", int(fold), x_uni_train, y[train_idx], x_uni_val, y[val_idx], alpha_grid=args.ridge_alphas))
        rows.extend(evaluate_model("cross_market_sequence_ridge", int(fold), x_cross_train, y[train_idx], x_cross_val, y[val_idx], alpha_grid=args.ridge_alphas))

        for seed in args.seeds:
            print(f"fold={int(fold)} seed={seed} ESN univariate", flush=True)
            f_uni_train = esn_features(x[train_idx, :, :2], seed=seed)
            f_uni_val = esn_features(x[val_idx, :, :2], seed=seed)
            rows.extend(evaluate_model("univariate_esn", int(fold), f_uni_train, y[train_idx], f_uni_val, y[val_idx], alpha_grid=args.esn_alphas, seed=seed))

            print(f"fold={int(fold)} seed={seed} ESN cross-market", flush=True)
            f_cross_train = esn_features(x[train_idx], seed=seed)
            f_cross_val = esn_features(x[val_idx], seed=seed)
            rows.extend(evaluate_model("cross_market_esn", int(fold), f_cross_train, y[train_idx], f_cross_val, y[val_idx], alpha_grid=args.esn_alphas, seed=seed))

            print(f"fold={int(fold)} seed={seed} ESN shuffled control", flush=True)
            shuffled_train = shuffled_cross_market(x[train_idx], seed)
            shuffled_val = shuffled_cross_market(x[val_idx], seed + 1000)
            f_shuf_train = esn_features(shuffled_train, seed=seed)
            f_shuf_val = esn_features(shuffled_val, seed=seed)
            rows.extend(evaluate_model("cross_market_esn_shuffled", int(fold), f_shuf_train, y[train_idx], f_shuf_val, y[val_idx], alpha_grid=args.esn_alphas, seed=seed))

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
    print("\nAGGREGATE", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
