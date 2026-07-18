from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    load_stage_d_run,
    qlike_loss,
)

DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
ESTABLISHED_ESN = {
    "n": 300,
    "spectral_radius": 0.9,
    "input_scale": 0.3,
    "leak": 0.3,
    "connectivity": 0.10,
}
ENRICHED_ESN = {
    "n": 300,
    "spectral_radius": 0.9,
    "input_scale": 0.3,
    "leak": 0.5,
    "connectivity": 0.02,
    "washout": 10,
}


def metric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(qlike_loss(y_true, y_pred).mean()),
        float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
    )


def scale_sequences_train_only(sequences: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (sequences - mean[None, None, :]) / scale[None, None, :]


def build_original_enriched_sequences(original: np.ndarray) -> np.ndarray:
    level = np.asarray(original, dtype=float)
    difference = np.diff(level, axis=1, prepend=level[:, :1, :])
    time = np.broadcast_to(
        np.linspace(0.0, 1.0, level.shape[1], dtype=float)[None, :, None],
        (len(level), level.shape[1], 1),
    )
    return np.concatenate([level, difference, time], axis=2)


def build_hybrid_sequences(original_enriched: np.ndarray, compact: np.ndarray) -> np.ndarray:
    if original_enriched.shape[:2] != compact.shape[:2]:
        raise ValueError("original and compact tensors are not aligned")
    # Compact channels 2:8 contain only cross-market aggregates. Reconstructed
    # own-market channels are deliberately excluded so Stage D remains the base.
    return np.concatenate([original_enriched, compact[:, :, 2:8]], axis=2)


def shuffle_cross_market_channels(hybrid: np.ndarray, seed: int) -> np.ndarray:
    out = np.asarray(hybrid, dtype=float).copy()
    rng = np.random.default_rng(seed)
    for sample in range(len(out)):
        order = rng.permutation(out.shape[1])
        out[sample, :, 3:9] = out[sample, order, 3:9]
    return out


def pooled_esn_features(sequences: np.ndarray, *, seed: int) -> np.ndarray:
    config = ENRICHED_ESN
    w_in, w = make_esn_weights(
        n_inputs=sequences.shape[2],
        n_reservoir=int(config["n"]),
        spectral_radius=float(config["spectral_radius"]),
        input_scale=float(config["input_scale"]),
        seed=int(seed),
        connectivity=float(config["connectivity"]),
    )
    features = np.empty((len(sequences), int(config["n"]) * 3), dtype=np.float32)
    washout = int(config["washout"])
    leak = float(config["leak"])
    for sample, sequence in enumerate(sequences):
        state = np.zeros(w.shape[0], dtype=float)
        kept = []
        for step, value in enumerate(sequence):
            candidate = np.tanh(w_in @ value + w @ state)
            state = (1.0 - leak) * state + leak * candidate
            if step >= washout:
                kept.append(state.copy())
        states = np.asarray(kept)
        features[sample] = np.concatenate([states[-1], states.mean(axis=0), states.std(axis=0)])
    return features


def _har_prediction(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _ridge_rows(
    *,
    fold: int,
    model_name: str,
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alphas: tuple[float, ...],
    seed: int,
    base_prediction: np.ndarray | None = None,
) -> list[dict[str, object]]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[train_mask])
    all_x = scaler.transform(features)
    train_target = target[train_mask]
    if base_prediction is not None:
        train_target = train_target - base_prediction[train_mask]
    rows = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(train_x, train_target)
        prediction = model.predict(all_x)
        if base_prediction is not None:
            prediction = base_prediction + prediction
        qlike, rmse = metric_pair(target[val_mask], prediction[val_mask])
        rows.append({
            "fold": int(fold),
            "model": model_name,
            "seed": int(seed),
            "alpha": float(alpha),
            "val_qlike": qlike,
            "val_rmse": rmse,
            "val_samples": int(val_mask.sum()),
        })
    return rows


def evaluate_fold(
    *,
    fold: int,
    manifest: pd.DataFrame,
    original_sequences: np.ndarray,
    compact_sequences: np.ndarray,
    valid: np.ndarray,
    seeds: tuple[int, ...],
    alphas: tuple[float, ...],
) -> pd.DataFrame:
    train_mask = manifest["fold_split"].eq("train").to_numpy() & valid
    val_mask = manifest["fold_split"].eq("val").to_numpy() & valid
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has no valid train or validation samples")

    target = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _har_prediction(manifest, target, train_mask)
    rows: list[dict[str, object]] = []
    qlike, rmse = metric_pair(target[val_mask], har[val_mask])
    rows.append({
        "fold": fold,
        "model": "har",
        "seed": 0,
        "alpha": 100.0,
        "val_qlike": qlike,
        "val_rmse": rmse,
        "val_samples": int(val_mask.sum()),
    })

    rows.extend(_ridge_rows(
        fold=fold,
        model_name="sequence_ridge",
        features=original_sequences.reshape(len(original_sequences), -1),
        target=target,
        train_mask=train_mask,
        val_mask=val_mask,
        alphas=alphas,
        seed=0,
    ))

    original_scaled = scale_sequences_train_only(original_sequences, train_mask)
    enriched = build_original_enriched_sequences(original_scaled)
    hybrid = build_hybrid_sequences(enriched, compact_sequences)

    for seed in seeds:
        established = ESTABLISHED_ESN
        w_in, w = make_esn_weights(
            n_inputs=1,
            n_reservoir=int(established["n"]),
            spectral_radius=float(established["spectral_radius"]),
            input_scale=float(established["input_scale"]),
            seed=int(seed),
            connectivity=float(established["connectivity"]),
        )
        ordered = esn_states(original_scaled, w_in, w, float(established["leak"]))
        rng = np.random.default_rng(int(seed) + 20000)
        shuffled_original = original_scaled.copy()
        for sample in range(len(shuffled_original)):
            shuffled_original[sample] = shuffled_original[sample, rng.permutation(shuffled_original.shape[1]), :]
        shuffled = esn_states(shuffled_original, w_in, w, float(established["leak"]))

        ordered_scaler = StandardScaler()
        ordered_train = ordered_scaler.fit_transform(ordered[train_mask])
        ordered_all = ordered_scaler.transform(ordered)
        count = min(10, ordered_train.shape[0], ordered_train.shape[1])
        pca = PCA(n_components=count, svd_solver="full")
        pca_train = pca.fit_transform(ordered_train)
        pca_all = pca.transform(ordered_all)
        pca_features = np.empty((len(ordered), count), dtype=float)
        pca_features[train_mask] = pca_train
        pca_features[~train_mask] = pca_all[~train_mask]

        rows.extend(_ridge_rows(
            fold=fold,
            model_name="pca10_esn",
            features=pca_features,
            target=target,
            train_mask=train_mask,
            val_mask=val_mask,
            alphas=alphas,
            seed=seed,
            base_prediction=har,
        ))
        rows.extend(_ridge_rows(
            fold=fold,
            model_name="shuffled_esn",
            features=shuffled,
            target=target,
            train_mask=train_mask,
            val_mask=val_mask,
            alphas=alphas,
            seed=seed,
            base_prediction=har,
        ))

        enriched_features = pooled_esn_features(enriched, seed=seed)
        hybrid_features = pooled_esn_features(hybrid, seed=seed)
        shuffled_hybrid_features = pooled_esn_features(
            shuffle_cross_market_channels(hybrid, seed + 30000),
            seed=seed,
        )
        for model_name, features in (
            ("enriched_univariate_esn", enriched_features),
            ("cross_market_esn", hybrid_features),
            ("cross_market_esn_shuffled", shuffled_hybrid_features),
        ):
            rows.extend(_ridge_rows(
                fold=fold,
                model_name=model_name,
                features=features,
                target=target,
                train_mask=train_mask,
                val_mask=val_mask,
                alphas=alphas,
                seed=seed,
                base_prediction=har,
            ))
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeded = results[results["seed"] > 0]
    seeded_summary = seeded.groupby(["fold", "model", "alpha"], as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_seed_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
        val_samples=("val_samples", "first"),
    )
    deterministic = results[results["seed"] == 0].rename(columns={
        "val_qlike": "mean_val_qlike",
        "val_rmse": "mean_val_rmse",
    })
    deterministic["std_seed_qlike"] = 0.0
    fold_grid = pd.concat([
        seeded_summary,
        deterministic[[
            "fold", "model", "alpha", "mean_val_qlike", "std_seed_qlike",
            "mean_val_rmse", "val_samples",
        ]],
    ], ignore_index=True)
    best_per_fold = (
        fold_grid.sort_values(["fold", "model", "mean_val_qlike", "mean_val_rmse"])
        .groupby(["fold", "model"], as_index=False)
        .first()
    )
    aggregate = best_per_fold.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_val_qlike=("mean_val_qlike", "mean"),
        std_across_folds=("mean_val_qlike", "std"),
        mean_val_rmse=("mean_val_rmse", "mean"),
    ).sort_values(["mean_val_qlike", "mean_val_rmse"])
    return best_per_fold, aggregate


def run_comparison(
    *,
    stage_d_run: Path,
    rolling_manifest: Path,
    tensor_run: Path,
    run_dir: Path,
    seeds: list[int],
    alphas: list[float],
) -> dict[str, object]:
    stage_d = load_stage_d_run(stage_d_run)
    manifest = stage_d.manifest.reset_index(drop=True)
    original = np.asarray(stage_d.sequences, dtype=float)
    rolling = pd.read_csv(rolling_manifest)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    if not set(sample_ids).issubset(set(rolling["sample_id"].astype(str))):
        raise ValueError("rolling manifest does not contain every Stage D sample")

    frames = []
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
        frames.append(evaluate_fold(
            fold=int(fold),
            manifest=fold_manifest,
            original_sequences=original,
            compact_sequences=compact,
            valid=valid,
            seeds=tuple(seeds),
            alphas=tuple(alphas),
        ))

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(run_dir / "results_by_seed.csv", index=False)
    best_per_fold, aggregate = summarize_results(results)
    best_per_fold.to_csv(run_dir / "best_per_fold.csv", index=False)
    aggregate.to_csv(run_dir / "aggregate_results.csv", index=False)
    summary = {
        "test_evaluated": False,
        "comparison_basis": "original Stage D target/input with a common valid-sample intersection",
        "qlike_definition": "stage_e_classical_baselines.qlike_loss on log-volatility targets",
        "best_model": aggregate.iloc[0].to_dict(),
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
