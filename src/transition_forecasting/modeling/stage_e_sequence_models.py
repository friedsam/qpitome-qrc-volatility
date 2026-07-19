from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_ESN_CONFIG = {"name": "n300_sr0.9", "n": 300, "sr": 0.9, "inp": 0.3, "leak": 0.3}


def load_stage_d(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv").reset_index(drop=True)
    with np.load(run_dir / "sequence_tensors.npz", allow_pickle=True) as data:
        sequences = np.asarray(data["X"], dtype=float)
        tensor_ids = data["sample_id"].astype(str)
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if sequences.shape != (len(manifest), 40, 1):
        raise ValueError(f"expected sequence tensors shaped (n, 40, 1), got {sequences.shape}")
    if not np.array_equal(tensor_ids, manifest_ids):
        raise ValueError("manifest and tensor sample IDs are not aligned")
    return manifest, sequences


def chronology_audit(manifest: pd.DataFrame, *, date_column: str = "origin_date") -> pd.DataFrame:
    frame = manifest.copy()
    frame["_date"] = pd.to_datetime(frame[date_column], errors="raise", utc=True)
    frame["year"] = frame["_date"].dt.year
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby("year", sort=True):
        row: dict[str, object] = {
            "year": int(year),
            "episodes": int(group["episode_id"].nunique()),
            "markets": int(group["market_group"].nunique()) if "market_group" in group else 0,
            "samples": int(len(group)),
            "positives": int(group["label"].eq(1).sum()) if "label" in group else 0,
            "negatives": int(group["label"].eq(0).sum()) if "label" in group else 0,
            "mean_target": float(group[list(TARGET_COLUMNS)].to_numpy(dtype=float).mean()),
        }
        for lead in sorted(group["lead"].dropna().unique()) if "lead" in group else []:
            row[f"lead_{int(lead)}"] = int(group["lead"].eq(lead).sum())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def scale_sequences(sequences: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (sequences - mean[None, None, :]) / scale[None, None, :]


def har_predictions(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return (
        float(qlike_loss(y[mask], prediction[mask]).mean()),
        float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2))),
    )


def evaluate_fold(
    manifest: pd.DataFrame,
    sequences: np.ndarray,
    *,
    fold: int,
    seeds: tuple[int, ...] = (1, 2, 3),
    alphas: tuple[float, ...] = ALPHAS,
    pca_components: int = 10,
    config: dict[str, object] | None = None,
) -> pd.DataFrame:
    config = dict(DEFAULT_ESN_CONFIG if config is None else config)
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has an empty train or validation partition")

    source_rows = manifest.get("_source_row")
    if source_rows is not None:
        row_index = source_rows.to_numpy(dtype=int)
        sequences = sequences[row_index]

    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = har_predictions(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []
    qlike, rmse = metrics(y, har, val_mask)
    rows.append({"fold": fold, "model": "har", "seed": 0, "alpha": 100.0, "val_qlike": qlike, "val_rmse": rmse})

    flat = sequences.reshape(len(sequences), -1)
    scaler = StandardScaler()
    flat_train = scaler.fit_transform(flat[train_mask])
    flat_all = scaler.transform(flat)
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(flat_train, y[train_mask])
        prediction = model.predict(flat_all)
        qlike, rmse = metrics(y, prediction, val_mask)
        rows.append({"fold": fold, "model": "sequence_ridge", "seed": 0, "alpha": float(alpha), "val_qlike": qlike, "val_rmse": rmse})

    scaled_sequences = scale_sequences(sequences, train_mask)
    for seed in seeds:
        w_in, w = make_esn_weights(
            n_inputs=scaled_sequences.shape[-1],
            n_reservoir=int(config["n"]),
            spectral_radius=float(config["sr"]),
            input_scale=float(config["inp"]),
            seed=int(seed),
        )
        ordered = esn_states(scaled_sequences, w_in, w, float(config["leak"]))
        rng = np.random.default_rng(int(seed) + 20000)
        shuffled_sequences = scaled_sequences.copy()
        for sample in range(len(shuffled_sequences)):
            shuffled_sequences[sample] = shuffled_sequences[sample, rng.permutation(shuffled_sequences.shape[1]), :]
        shuffled = esn_states(shuffled_sequences, w_in, w, float(config["leak"]))

        ordered_scaler = StandardScaler()
        ordered_train = ordered_scaler.fit_transform(ordered[train_mask])
        ordered_all = ordered_scaler.transform(ordered)
        count = min(pca_components, ordered_train.shape[0], ordered_train.shape[1])
        pca = PCA(n_components=count, svd_solver="full")
        pca_train = pca.fit_transform(ordered_train)
        pca_all = pca.transform(ordered_all)

        shuffled_scaler = StandardScaler()
        shuffled_train = shuffled_scaler.fit_transform(shuffled[train_mask])
        shuffled_all = shuffled_scaler.transform(shuffled)

        for model_name, x_train, x_all in (
            ("pca10_esn", pca_train, pca_all),
            ("shuffled_esn", shuffled_train, shuffled_all),
        ):
            for alpha in alphas:
                model = Ridge(alpha=float(alpha))
                model.fit(x_train, residual[train_mask])
                prediction = har + model.predict(x_all)
                qlike, rmse = metrics(y, prediction, val_mask)
                rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "val_qlike": qlike,
                        "val_rmse": rmse,
                    }
                )
    return pd.DataFrame(rows)
