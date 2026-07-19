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

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
CONFIG = {"name": "n300_sr0.9", "n": 300, "sr": 0.9, "inp": 0.3, "leak": 0.3}


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


def chronology_audit(manifest: pd.DataFrame, *, date_column: str, group_column: str) -> pd.DataFrame:
    frame = manifest.copy()
    frame["_date"] = pd.to_datetime(frame[date_column], errors="raise", utc=True)
    frame["year"] = frame["_date"].dt.year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        row = {
            "year": int(year),
            "episodes": int(group[group_column].nunique()),
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


def rolling_origin_folds(
    manifest: pd.DataFrame,
    *,
    date_column: str,
    group_column: str,
    n_folds: int = 3,
    test_fraction: float = 0.17,
    embargo_days: int = 10,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frame = manifest.copy()
    frame["_date"] = pd.to_datetime(frame[date_column], errors="raise", utc=True)
    group_dates = (
        frame.groupby(group_column, as_index=False)["_date"]
        .min()
        .sort_values(["_date", group_column])
        .reset_index(drop=True)
    )
    n_groups = len(group_dates)
    n_test = max(1, int(round(n_groups * test_fraction)))
    n_dev = n_groups - n_test
    if n_dev < n_folds + 1:
        raise ValueError("not enough chronological groups for requested rolling folds")

    blocks = [np.asarray(block, dtype=int) for block in np.array_split(np.arange(n_dev), n_folds + 1)]
    embargo = pd.Timedelta(days=int(embargo_days))
    assignments = []
    summaries: list[dict[str, object]] = []

    for fold in range(n_folds):
        train_indices = np.concatenate(blocks[: fold + 1])
        val_indices = blocks[fold + 1]
        train_groups = set(group_dates.loc[train_indices, group_column])
        val_groups = set(group_dates.loc[val_indices, group_column])
        boundary = group_dates.loc[val_indices[0], "_date"]
        purged_groups = set(group_dates.loc[(group_dates["_date"] - boundary).abs() <= embargo, group_column])
        train_groups -= purged_groups
        val_groups -= purged_groups

        fold_split = pd.Series("unused", index=frame.index, dtype=object)
        fold_split[frame[group_column].isin(train_groups)] = "train"
        fold_split[frame[group_column].isin(val_groups)] = "val"
        fold_split[frame[group_column].isin(purged_groups)] = "purged"
        test_groups = set(group_dates.loc[n_dev:, group_column])
        fold_split[frame[group_column].isin(test_groups)] = "test"

        fold_frame = frame.copy()
        fold_frame["fold"] = fold + 1
        fold_frame["fold_split"] = fold_split
        assignments.append(fold_frame.drop(columns="_date"))
        summaries.append({
            "fold": fold + 1,
            "boundary": str(boundary),
            "train_episodes": int(len(train_groups)),
            "val_episodes": int(len(val_groups)),
            "purged_episodes": int(len(purged_groups)),
            "test_episodes": int(len(test_groups)),
            "train_samples": int(fold_split.eq("train").sum()),
            "val_samples": int(fold_split.eq("val").sum()),
        })

    return pd.concat(assignments, ignore_index=True), summaries


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


def _metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
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
) -> pd.DataFrame:
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"fold {fold} has an empty train or validation partition")

    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _har_predictions(manifest, y, train_mask)
    residual = y - har
    rows: list[dict[str, object]] = []
    qlike, rmse = _metrics(y, har, val_mask)
    rows.append({"fold": fold, "model": "har", "seed": 0, "alpha": 100.0, "val_qlike": qlike, "val_rmse": rmse})

    flat = sequences.reshape(len(sequences), -1)
    scaler = StandardScaler()
    flat_train = scaler.fit_transform(flat[train_mask])
    flat_all = scaler.transform(flat)
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(flat_train, y[train_mask])
        prediction = model.predict(flat_all)
        qlike, rmse = _metrics(y, prediction, val_mask)
        rows.append({"fold": fold, "model": "sequence_ridge", "seed": 0, "alpha": float(alpha), "val_qlike": qlike, "val_rmse": rmse})

    scaled_sequences = _scale_sequences(sequences, train_mask)
    for seed in seeds:
        w_in, w = make_esn_weights(
            n_inputs=1,
            n_reservoir=int(CONFIG["n"]),
            spectral_radius=float(CONFIG["sr"]),
            input_scale=float(CONFIG["inp"]),
            seed=int(seed),
        )
        ordered = esn_states(scaled_sequences, w_in, w, float(CONFIG["leak"]))
        rng = np.random.default_rng(int(seed) + 20000)
        shuffled_sequences = scaled_sequences.copy()
        for sample in range(len(shuffled_sequences)):
            shuffled_sequences[sample] = shuffled_sequences[sample, rng.permutation(shuffled_sequences.shape[1]), :]
        shuffled = esn_states(shuffled_sequences, w_in, w, float(CONFIG["leak"]))

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
                qlike, rmse = _metrics(y, prediction, val_mask)
                rows.append({
                    "fold": fold,
                    "model": model_name,
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "val_qlike": qlike,
                    "val_rmse": rmse,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-fold rolling-origin Stage E baseline study.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--date-column", default="event_onset")
    parser.add_argument("--group-column", default="episode_id")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.17)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_rolling_origin"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    manifest, sequences = load_stage_d(args.stage_d_run)
    audit = chronology_audit(manifest, date_column=args.date_column, group_column=args.group_column)
    assignments, fold_summary = rolling_origin_folds(
        manifest,
        date_column=args.date_column,
        group_column=args.group_column,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        embargo_days=args.embargo_days,
    )
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    audit.to_csv(run_dir / "chronology_by_year.csv", index=False)
    assignments.to_csv(run_dir / "rolling_fold_manifest.csv", index=False)

    frames = []
    for fold in range(1, args.n_folds + 1):
        fold_manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        frames.append(evaluate_fold(fold_manifest, sequences, fold=fold, seeds=tuple(args.seeds)))
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(run_dir / "rolling_results_by_seed.csv", index=False)

    seeded = results[results["seed"] > 0]
    seeded_summary = seeded.groupby(["fold", "model", "alpha"], as_index=False).agg(
        mean_val_qlike=("val_qlike", "mean"),
        std_val_qlike=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
    )
    deterministic = results[results["seed"] == 0].rename(
        columns={"val_qlike": "mean_val_qlike", "val_rmse": "mean_val_rmse"}
    )
    deterministic["std_val_qlike"] = 0.0
    fold_table = pd.concat([
        seeded_summary,
        deterministic[["fold", "model", "alpha", "mean_val_qlike", "std_val_qlike", "mean_val_rmse"]],
    ], ignore_index=True)
    fold_table.to_csv(run_dir / "rolling_summary_by_fold.csv", index=False)

    best_per_fold = (
        fold_table.sort_values(["fold", "model", "mean_val_qlike", "mean_val_rmse"])
        .groupby(["fold", "model"], as_index=False)
        .first()
    )
    aggregate = best_per_fold.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_val_qlike=("mean_val_qlike", "mean"),
        std_across_folds=("mean_val_qlike", "std"),
        mean_val_rmse=("mean_val_rmse", "mean"),
        wins=("mean_val_qlike", lambda values: 0),
    )
    winners = best_per_fold.loc[best_per_fold.groupby("fold")["mean_val_qlike"].idxmin(), "model"].value_counts()
    aggregate["wins"] = aggregate["model"].map(winners).fillna(0).astype(int)
    aggregate = aggregate.sort_values("mean_val_qlike")
    best_per_fold.to_csv(run_dir / "rolling_best_by_fold.csv", index=False)
    aggregate.to_csv(run_dir / "rolling_aggregate.csv", index=False)

    payload = {
        "stage_d_run": str(args.stage_d_run),
        "test_evaluated": False,
        "folds": fold_summary,
        "aggregate": aggregate.to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
