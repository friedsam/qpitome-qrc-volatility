from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.classical_baselines.baselines import (
    TARGET_COLUMNS,
    load_frozen_folds,
    metric_records,
    prediction_frame,
    qlike,
)

ALPHAS = (100.0, 1000.0)
SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class FieldESNSpec:
    reservoir_size: int = 300
    spectral_radius: float = 0.9
    input_scale: float = 0.3
    leak: float = 0.5
    connectivity: float = 0.02
    washout: int = 10


def _ridge_predict(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[train])
    eval_x = scaler.transform(features[evaluate])
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=5000)
    model.fit(train_x, target[train])
    result = np.full((len(target), target.shape[1]), np.nan, dtype=float)
    result[evaluate] = model.predict(eval_x)
    return result


def _make_weights(seed: int, spec: FieldESNSpec) -> tuple[np.ndarray, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    w_in = rng.uniform(
        -spec.input_scale,
        spec.input_scale,
        size=(spec.reservoir_size, 3),
    )
    mask = rng.random((spec.reservoir_size, spec.reservoir_size)) < spec.connectivity
    recurrent = rng.normal(size=(spec.reservoir_size, spec.reservoir_size)) * mask
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    recurrent *= spec.spectral_radius / max(radius, 1e-12)
    return w_in, sparse.csr_matrix(recurrent)


def _field_sequence(
    level_sequence: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    mean = float(level_sequence[train].mean())
    std = float(level_sequence[train].std())
    level = (level_sequence - mean) / max(std, 1e-8)
    difference = np.diff(level, axis=1, prepend=level[:, :1])
    time = np.broadcast_to(
        np.linspace(0.0, 1.0, level.shape[1], dtype=float)[None, :],
        level.shape,
    )
    return np.stack([level, difference, time], axis=2).astype(np.float32)


def _pooled_esn_features(
    sequence: np.ndarray,
    active: np.ndarray,
    seed: int,
    *,
    shuffled: bool,
    spec: FieldESNSpec,
) -> np.ndarray:
    x = sequence.copy()
    if shuffled:
        rng = np.random.default_rng(seed + 10000)
        level = x[:, :, 0].copy()
        for row in np.where(active)[0]:
            rng.shuffle(level[row])
        difference = np.diff(level, axis=1, prepend=level[:, :1])
        x = np.stack([level, difference, x[:, :, 2]], axis=2)

    w_in, recurrent = _make_weights(seed, spec)
    state = np.zeros((len(x), spec.reservoir_size), dtype=np.float32)
    kept: list[np.ndarray] = []
    for step in range(x.shape[1]):
        candidate = np.tanh(x[:, step] @ w_in.T + state @ recurrent.T)
        state = (1.0 - spec.leak) * state + spec.leak * candidate
        if step >= spec.washout:
            kept.append(state.copy())
    states = np.stack(kept, axis=1)
    return np.concatenate(
        [states[:, -1], states.mean(axis=1), states.std(axis=1)],
        axis=1,
    )


def _choose_alpha(
    features: np.ndarray,
    mode: str,
    target: np.ndarray,
    har_features: np.ndarray,
    har_prediction: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    active: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    candidates: list[tuple[float, float, np.ndarray]] = []
    for alpha in ALPHAS:
        if mode == "direct":
            prediction = _ridge_predict(features, target, train, active, alpha)
        elif mode == "residual":
            correction = _ridge_predict(
                features,
                target - har_prediction,
                train,
                active,
                alpha,
            )
            prediction = har_prediction + correction
        elif mode == "joint":
            joint = np.concatenate([har_features, features], axis=1)
            prediction = _ridge_predict(joint, target, train, active, alpha)
        else:
            raise ValueError(f"unknown ESN readout mode: {mode}")
        score = float(qlike(target[val], prediction[val]).mean())
        candidates.append((score, alpha, prediction))
    return min(candidates, key=lambda item: item[0])


def run_comparable_esn_folds(
    fold_dir: Path,
    *,
    spec: FieldESNSpec = FieldESNSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest, level_sequence = load_frozen_folds(fold_dir)
    targets = manifest[TARGET_COLUMNS].to_numpy(dtype=float)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    fold_counts: list[dict[str, int]] = []

    fold_values = sorted(manifest["fold"].astype(int).unique())
    for fold in fold_values:
        fold_mask = manifest["fold"].astype(int).to_numpy() == fold
        train = fold_mask & manifest["fold_split"].eq("train").to_numpy()
        val = fold_mask & manifest["fold_split"].eq("val").to_numpy()
        test = fold_mask & manifest["fold_split"].eq("test").to_numpy()
        active = train | val
        if not train.any() or not val.any() or not test.any():
            raise ValueError(f"fold {fold} lacks train, validation, or reserved test rows")

        har_features = np.column_stack(
            [
                level_sequence[:, -1],
                level_sequence[:, -5:].mean(axis=1),
                level_sequence[:, -20:].mean(axis=1),
            ]
        )
        har_prediction = _ridge_predict(
            har_features,
            targets,
            train,
            active,
            alpha=100.0,
        )
        sequence = _field_sequence(level_sequence, train)
        val_frame = manifest.loc[val].reset_index(drop=True)
        val_targets = targets[val]

        all_metrics.extend(
            metric_records(
                dataset="frozen_transition_folds_comparable_esn",
                model="har",
                fold=int(fold),
                frame=val_frame,
                actual=val_targets,
                predicted=har_prediction[val],
            )
        )
        all_predictions.append(
            prediction_frame(
                dataset="frozen_transition_folds_comparable_esn",
                model="har",
                fold=int(fold),
                frame=val_frame,
                actual=val_targets,
                predicted=har_prediction[val],
            )
        )

        for seed in SEEDS:
            print(f"fold {fold}: generating ESN features for seed {seed}", flush=True)
            ordered = _pooled_esn_features(
                sequence,
                active,
                seed,
                shuffled=False,
                spec=spec,
            )
            shuffled = _pooled_esn_features(
                sequence,
                active,
                seed,
                shuffled=True,
                spec=spec,
            )
            for ordering, features in (("ordered", ordered), ("shuffled", shuffled)):
                for mode in ("direct", "residual", "joint"):
                    _, alpha, prediction = _choose_alpha(
                        features,
                        mode,
                        targets,
                        har_features,
                        har_prediction,
                        train,
                        val,
                        active,
                    )
                    model = f"esn_{mode}_{ordering}_seed{seed}"
                    all_metrics.extend(
                        metric_records(
                            dataset="frozen_transition_folds_comparable_esn",
                            model=model,
                            fold=int(fold),
                            frame=val_frame,
                            actual=val_targets,
                            predicted=prediction[val],
                        )
                    )
                    records = prediction_frame(
                        dataset="frozen_transition_folds_comparable_esn",
                        model=model,
                        fold=int(fold),
                        frame=val_frame,
                        actual=val_targets,
                        predicted=prediction[val],
                    )
                    records["alpha"] = float(alpha)
                    records["seed"] = int(seed)
                    records["ordering"] = ordering
                    records["readout"] = mode
                    all_predictions.append(records)

        fold_counts.append(
            {
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "validation_rows": int(val.sum()),
                "reserved_test_rows": int(test.sum()),
                "test_predictions_written": 0,
            }
        )

    metrics = pd.DataFrame(all_metrics)
    seeded = metrics[metrics["model"].str.startswith("esn_")].copy()
    seeded["family"] = seeded["model"].str.replace(r"_seed\d+$", "", regex=True)
    family_summary = (
        seeded.groupby(
            ["family", "fold", "group_type", "group_value", "horizon"],
            as_index=False,
        )
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_qlike=("qlike", "mean"),
            sd_qlike=("qlike", "std"),
            seeds=("model", "nunique"),
        )
    )
    return metrics, pd.concat(all_predictions, ignore_index=True), {
        "fold_counts": fold_counts,
        "family_summary": family_summary,
        "spec": {
            "reservoir_size": spec.reservoir_size,
            "spectral_radius": spec.spectral_radius,
            "input_scale": spec.input_scale,
            "leak": spec.leak,
            "connectivity": spec.connectivity,
            "washout": spec.washout,
            "features": "final + post-washout mean + post-washout std",
            "seeds": list(SEEDS),
            "alphas": list(ALPHAS),
            "alpha_selection": "minimum validation QLIKE within each fold, matching field test",
            "residual_readout": "field-test reproduction using in-sample HAR training residuals",
        },
    }
