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
from transition_forecasting.modeling.stage_e_classical_baselines import HAR_FEATURES, TARGET_COLUMNS, qlike_loss

DATE_CANDIDATES = (
    "event_date", "anchor_date", "sample_date", "date", "event_timestamp", "timestamp",
    "event_time", "anchor_time", "target_date",
)
GROUP_CANDIDATES = ("global_cluster_id", "global_event_id", "cluster_id", "episode_id")
DEFAULT_CONFIGS = (
    {"name": "n300_sr0.9", "n": 300, "sr": 0.9, "inp": 0.3, "leak": 0.3},
    {"name": "n500_sr0.9", "n": 500, "sr": 0.9, "inp": 0.2, "leak": 0.5},
)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _load_stage_d(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
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


def _resolve_column(manifest: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for column in candidates:
        if column in manifest.columns:
            return column
    raise ValueError(f"could not resolve {label} column; available columns: {manifest.columns.tolist()}")


def chronological_split(
    manifest: pd.DataFrame,
    *,
    train_fraction: float = 0.75,
    val_fraction: float = 0.08,
    embargo_days: int = 10,
    date_column: str | None = None,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must lie in (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be below 1")

    date_column = date_column or _resolve_column(manifest, DATE_CANDIDATES, "date")
    group_column = group_column or _resolve_column(manifest, GROUP_CANDIDATES, "group")
    frame = manifest.copy()
    frame["_chronology"] = pd.to_datetime(frame[date_column], errors="raise", utc=True)
    group_dates = (
        frame.groupby(group_column, as_index=False)["_chronology"]
        .min()
        .sort_values(["_chronology", group_column])
        .reset_index(drop=True)
    )
    n_groups = len(group_dates)
    if n_groups < 3:
        raise ValueError("at least three chronological groups are required")
    train_end = max(1, min(n_groups - 2, int(np.floor(n_groups * train_fraction))))
    val_end = max(train_end + 1, min(n_groups - 1, int(np.floor(n_groups * (train_fraction + val_fraction)))))
    train_cut = group_dates.loc[train_end, "_chronology"]
    val_cut = group_dates.loc[val_end, "_chronology"]
    embargo = pd.Timedelta(days=int(embargo_days))

    group_split: dict[object, str] = {}
    for _, row in group_dates.iterrows():
        group_id = row[group_column]
        date = row["_chronology"]
        if abs(date - train_cut) <= embargo or abs(date - val_cut) <= embargo:
            split = "purged"
        elif date < train_cut:
            split = "train"
        elif date < val_cut:
            split = "val"
        else:
            split = "test"
        group_split[group_id] = split

    frame["original_split"] = frame["split"].astype(str) if "split" in frame.columns else ""
    frame["split"] = frame[group_column].map(group_split)
    frame = frame.drop(columns="_chronology")
    if frame.groupby(group_column)["split"].nunique().max() > 1:
        raise ValueError("chronological split leaked groups across partitions")
    if frame.groupby("episode_id")["split"].nunique().max() > 1:
        raise ValueError("chronological split leaked episodes across partitions")

    summary = {
        "date_column": date_column,
        "group_column": group_column,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": 1.0 - train_fraction - val_fraction,
        "embargo_days": embargo_days,
        "train_cut": str(train_cut),
        "val_cut": str(val_cut),
        "group_counts": frame.groupby("split")[group_column].nunique().to_dict(),
        "sample_counts": frame["split"].value_counts().to_dict(),
    }
    return frame, summary


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


def _score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return float(qlike_loss(y[mask], prediction[mask]).mean()), float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))


def run_study(
    manifest: pd.DataFrame,
    sequences: np.ndarray,
    *,
    configs: tuple[dict[str, object], ...] = DEFAULT_CONFIGS,
    seeds: tuple[int, ...] = (1, 2, 3),
    alphas: tuple[float, ...] = ALPHAS,
    pca_components: int = 10,
) -> pd.DataFrame:
    train_mask = manifest["split"].eq("train").to_numpy()
    val_mask = manifest["split"].eq("val").to_numpy()
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("chronological split produced an empty train or validation partition")

    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _har_predictions(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []
    har_qlike, har_rmse = _score(y, har, val_mask)
    rows.append({"model": "har", "config": "har", "seed": 0, "alpha": 100.0, "val_qlike": har_qlike, "val_rmse": har_rmse})

    flat = sequences.reshape(len(sequences), -1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(flat[train_mask])
    x_all = scaler.transform(flat)
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y[train_mask])
        prediction = model.predict(x_all)
        qlike, rmse = _score(y, prediction, val_mask)
        rows.append({"model": "sequence_ridge", "config": "sequence", "seed": 0, "alpha": alpha, "val_qlike": qlike, "val_rmse": rmse})

    scaled_sequences = _scale_sequences(sequences, train_mask)
    for config in configs:
        for seed in seeds:
            w_in, w = make_esn_weights(
                n_inputs=scaled_sequences.shape[-1], n_reservoir=int(config["n"]),
                spectral_radius=float(config["sr"]), input_scale=float(config["inp"]), seed=int(seed),
            )
            ordered = esn_states(scaled_sequences, w_in, w, float(config["leak"]))
            rng = np.random.default_rng(int(seed) + 20000)
            shuffled_sequences = scaled_sequences.copy()
            for sample in range(len(shuffled_sequences)):
                shuffled_sequences[sample] = shuffled_sequences[sample, rng.permutation(shuffled_sequences.shape[1]), :]
            shuffled = esn_states(shuffled_sequences, w_in, w, float(config["leak"]))

            flat_scaled = scaled_sequences.reshape(len(scaled_sequences), -1)
            random_rng = np.random.default_rng(int(seed) + 10000)
            projection = random_rng.normal(0.0, 1.0 / np.sqrt(flat_scaled.shape[1]), size=(flat_scaled.shape[1], int(config["n"])))
            bias = random_rng.uniform(-1.0, 1.0, size=int(config["n"]))
            random_tanh = np.tanh(flat_scaled @ projection + bias)

            ordered_scaler = StandardScaler()
            ordered_train = ordered_scaler.fit_transform(ordered[train_mask])
            ordered_all = ordered_scaler.transform(ordered)
            count = min(int(pca_components), ordered_train.shape[0], ordered_train.shape[1])
            pca = PCA(n_components=count, svd_solver="full")
            pca.fit(ordered_train)
            pca_all = pca.transform(ordered_all)

            for model_name, features in (("full_esn", ordered), ("pca10_esn", pca_all), ("shuffled_esn", shuffled), ("random_tanh", random_tanh)):
                feature_scaler = StandardScaler()
                feature_train = feature_scaler.fit_transform(features[train_mask])
                feature_all = feature_scaler.transform(features)
                for alpha in alphas:
                    model = Ridge(alpha=alpha)
                    model.fit(feature_train, residual[train_mask])
                    prediction = har + model.predict(feature_all)
                    qlike, rmse = _score(y, prediction, val_mask)
                    rows.append({
                        "model": model_name, "config": config["name"], "seed": int(seed), "alpha": float(alpha),
                        "val_qlike": qlike, "val_rmse": rmse,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict chronological robustness check for Stage E models.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--date-column")
    parser.add_argument("--group-column")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/stage_e_chronological_robustness"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    manifest, sequences = _load_stage_d(args.stage_d_run)
    chronological, split_summary = chronological_split(
        manifest, train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        embargo_days=args.embargo_days, date_column=args.date_column, group_column=args.group_column,
    )
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    chronological.to_csv(run_dir / "chronological_manifest.csv", index=False)
    results = run_study(chronological, sequences, seeds=tuple(args.seeds))
    results.to_csv(run_dir / "chronological_results_by_seed.csv", index=False)

    seed_models = results[results["seed"] > 0]
    grouped_seed = seed_models.groupby(["model", "config", "alpha"], as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"), std_val_qlike=("val_qlike", "std"), mean_val_rmse=("val_rmse", "mean"),
    )
    deterministic = results[results["seed"] == 0].rename(columns={"val_qlike": "mean_val_qlike", "val_rmse": "mean_val_rmse"})
    deterministic["std_val_qlike"] = 0.0
    summary_table = pd.concat([
        grouped_seed,
        deterministic[["model", "config", "alpha", "mean_val_qlike", "std_val_qlike", "mean_val_rmse"]],
    ], ignore_index=True)
    best = summary_table.sort_values(["model", "mean_val_qlike", "mean_val_rmse"]).groupby("model", as_index=False).first().sort_values("mean_val_qlike")
    summary_table.to_csv(run_dir / "chronological_summary.csv", index=False)
    best.to_csv(run_dir / "chronological_best_by_model.csv", index=False)
    payload = {
        "stage_d_run": str(args.stage_d_run), "selection_split": "chronological_val", "test_evaluated": False,
        "split": split_summary, "best_rows": best.to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
