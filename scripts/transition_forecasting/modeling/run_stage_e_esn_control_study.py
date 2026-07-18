from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    qlike_loss,
    validate_split_integrity,
)

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
COMPONENTS = (1, 2, 3, 5, 10, 20)


def _load_stage_d(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv").reset_index(drop=True)
    validate_split_integrity(manifest)
    with np.load(run_dir / "sequence_tensors.npz", allow_pickle=True) as data:
        sequences = np.asarray(data["X"], dtype=float)
        sample_ids = data["sample_id"].astype(str)
    expected = manifest["sample_id"].astype(str).to_numpy()
    if sequences.shape != (len(manifest), 40, 1) or not np.array_equal(sample_ids, expected):
        raise ValueError("Stage D manifest and sequence tensors are not aligned")
    return manifest, sequences


def _load_cache(path: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        states = np.asarray(data["states"], dtype=float)
        sample_ids = data["sample_id"].astype(str)
        config = json.loads(str(data["config_json"].item()))
        seed = int(data["seed"].item())
    if not np.array_equal(sample_ids, manifest["sample_id"].astype(str).to_numpy()):
        raise ValueError(f"cache does not align with manifest: {path}")
    return states, {**config, "seed": seed}


def _scale_sequences(sequences: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (sequences - mean[None, None, :]) / scale[None, None, :]


def _har_predictions(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _residual_order(pc_train: np.ndarray, residual_train: np.ndarray) -> np.ndarray:
    scores = []
    for index in range(pc_train.shape[1]):
        x = pc_train[:, index]
        correlations = []
        for horizon in range(residual_train.shape[1]):
            y = residual_train[:, horizon]
            if np.std(x) == 0.0 or np.std(y) == 0.0:
                correlations.append(0.0)
            else:
                correlations.append(abs(float(np.corrcoef(x, y)[0, 1])))
        scores.append(max(correlations))
    return np.argsort(np.asarray(scores))[::-1]


def build_feature_sets(
    states: np.ndarray,
    sequences: np.ndarray,
    train_mask: np.ndarray,
    residual: np.ndarray,
    metadata: dict[str, object],
    *,
    components: tuple[int, ...] = COMPONENTS,
) -> dict[str, np.ndarray]:
    scaled_sequences = _scale_sequences(sequences, train_mask)
    n_reservoir = int(metadata["n"])
    seed = int(metadata["seed"])

    features: dict[str, np.ndarray] = {
        "full_esn": states,
        "final_input": scaled_sequences[:, -1, :],
    }

    flat = scaled_sequences.reshape(len(scaled_sequences), -1)
    rng = np.random.default_rng(seed + 10000)
    projection = rng.normal(0.0, 1.0 / np.sqrt(flat.shape[1]), size=(flat.shape[1], n_reservoir))
    bias = rng.uniform(-1.0, 1.0, size=n_reservoir)
    features["random_tanh"] = np.tanh(flat @ projection + bias)

    shuffled = scaled_sequences.copy()
    shuffle_rng = np.random.default_rng(seed + 20000)
    for sample in range(len(shuffled)):
        shuffled[sample] = shuffled[sample, shuffle_rng.permutation(shuffled.shape[1]), :]
    w_in, w = make_esn_weights(
        n_inputs=scaled_sequences.shape[-1],
        n_reservoir=n_reservoir,
        spectral_radius=float(metadata["sr"]),
        input_scale=float(metadata["inp"]),
        seed=seed,
    )
    features["shuffled_esn"] = esn_states(shuffled, w_in, w, float(metadata["leak"]))

    scaler = StandardScaler()
    z_train = scaler.fit_transform(states[train_mask])
    z_all = scaler.transform(states)
    n_pca = min(z_train.shape)
    pca = PCA(n_components=n_pca, svd_solver="full")
    pc_train = pca.fit_transform(z_train)
    pc_all = pca.transform(z_all)
    residual_rank = _residual_order(pc_train, residual[train_mask])
    for count in components:
        if count > n_pca:
            continue
        features[f"pca_variance_{count}"] = pc_all[:, :count]
        features[f"pca_residual_{count}"] = pc_all[:, residual_rank[:count]]
    return features


def score_feature_sets(
    feature_sets: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    metadata: dict[str, object],
    *,
    alphas: tuple[float, ...] = ALPHAS,
) -> pd.DataFrame:
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _har_predictions(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []

    for feature_set, x in feature_sets.items():
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_mask])
        x_val = scaler.transform(x[val_mask])
        for alpha in alphas:
            model = Ridge(alpha=float(alpha))
            model.fit(x_train, residual[train_mask])
            train_prediction = har[train_mask] + model.predict(x_train)
            val_prediction = har[val_mask] + model.predict(x_val)
            rows.append({
                "config": metadata["name"],
                "seed": int(metadata["seed"]),
                "n_reservoir": int(metadata["n"]),
                "feature_set": feature_set,
                "n_features": int(x.shape[1]),
                "alpha": float(alpha),
                "train_qlike": float(qlike_loss(y[train_mask], train_prediction).mean()),
                "val_qlike": float(qlike_loss(y[val_mask], val_prediction).mean()),
                "val_rmse": float(np.sqrt(np.mean((y[val_mask] - val_prediction) ** 2))),
                "coefficient_norm": float(np.linalg.norm(model.coef_)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched controls for Stage E ESN residual forecasting.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument(
        "--reservoir-cache-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_esn_reservoir_features"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_esn_control_study"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    manifest, sequences = _load_stage_d(args.stage_d_run)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    residual = y - _har_predictions(manifest, y, train_mask)
    cache_files = sorted(args.reservoir_cache_dir.glob("reservoir_states__*.npz"))
    if not cache_files:
        raise FileNotFoundError(f"no reservoir cache files found under {args.reservoir_cache_dir}")

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    frames = []
    for cache_file in cache_files:
        states, metadata = _load_cache(cache_file, manifest)
        feature_sets = build_feature_sets(states, sequences, train_mask, residual, metadata)
        frames.append(score_feature_sets(feature_sets, manifest, metadata))

    results = pd.concat(frames, ignore_index=True)
    summary = (
        results.groupby(["config", "n_reservoir", "feature_set", "n_features", "alpha"], as_index=False)
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            std_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_train_qlike=("train_qlike", "mean"),
        )
    )
    best = (
        summary.sort_values(["feature_set", "mean_val_qlike", "mean_val_rmse"])
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("mean_val_qlike")
    )
    results.to_csv(run_dir / "control_results_by_seed.csv", index=False)
    summary.to_csv(run_dir / "control_summary.csv", index=False)
    best.to_csv(run_dir / "control_best_by_feature_set.csv", index=False)
    payload = {
        "stage_d_run": str(args.stage_d_run),
        "reservoir_cache_dir": str(args.reservoir_cache_dir),
        "cache_files": len(cache_files),
        "selection_split": "val",
        "test_evaluated": False,
        "best_rows": best.head(20).to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
