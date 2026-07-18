from __future__ import annotations

import argparse
import importlib.util
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

ROLLING_SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_rolling_origin", ROLLING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROLLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLLING)

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MODELS = ("sequence_ridge", "pca10_esn", "shuffled_esn")


def select_nested_alphas(rolling_results: pd.DataFrame, evaluation_fold: int) -> dict[str, float]:
    prior = rolling_results[rolling_results["fold"] < evaluation_fold].copy()
    if prior.empty:
        raise ValueError("nested alpha selection requires at least one earlier fold")
    selected: dict[str, float] = {}
    for model in MODELS:
        candidates = prior[prior["model"].eq(model)]
        if candidates.empty:
            raise ValueError(f"missing prior-fold results for {model}")
        scores = candidates.groupby("alpha", as_index=False)["val_qlike"].mean()
        selected[model] = float(scores.sort_values(["val_qlike", "alpha"]).iloc[0]["alpha"])
    return selected


def _scale_sequences(sequences: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (sequences - mean[None, None, :]) / scale[None, None, :]


def _fit_har(manifest: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=100.0)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def evaluate_nested_fold(
    manifest: pd.DataFrame,
    sequences: np.ndarray,
    *,
    fold: int,
    selected_alphas: dict[str, float],
    seeds: tuple[int, ...] = (1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = manifest["fold_split"].eq("train").to_numpy()
    val_mask = manifest["fold_split"].eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har(manifest, y, train_mask)
    residual = y - har

    predictions: dict[str, list[np.ndarray]] = {"har": [har]}
    flat = sequences.reshape(len(sequences), -1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(flat[train_mask])
    x_all = scaler.transform(flat)
    sequence_model = Ridge(alpha=selected_alphas["sequence_ridge"])
    sequence_model.fit(x_train, y[train_mask])
    predictions["sequence_ridge"] = [sequence_model.predict(x_all)]

    scaled_sequences = _scale_sequences(sequences, train_mask)
    predictions["pca10_esn"] = []
    predictions["shuffled_esn"] = []
    for seed in seeds:
        config = ROLLING.CONFIG
        w_in, w = make_esn_weights(
            n_inputs=1,
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
        pca = PCA(n_components=min(10, ordered_train.shape[0], ordered_train.shape[1]), svd_solver="full")
        pca_train = pca.fit_transform(ordered_train)
        pca_all = pca.transform(ordered_all)
        pca_model = Ridge(alpha=selected_alphas["pca10_esn"])
        pca_model.fit(pca_train, residual[train_mask])
        predictions["pca10_esn"].append(har + pca_model.predict(pca_all))

        shuffled_scaler = StandardScaler()
        shuffled_train = shuffled_scaler.fit_transform(shuffled[train_mask])
        shuffled_all = shuffled_scaler.transform(shuffled)
        shuffled_model = Ridge(alpha=selected_alphas["shuffled_esn"])
        shuffled_model.fit(shuffled_train, residual[train_mask])
        predictions["shuffled_esn"].append(har + shuffled_model.predict(shuffled_all))

    metric_rows = []
    ensemble_predictions: dict[str, np.ndarray] = {}
    for model, model_predictions in predictions.items():
        ensemble = np.mean(np.stack(model_predictions, axis=0), axis=0)
        ensemble_predictions[model] = ensemble
        metric_rows.append({
            "fold": fold,
            "model": model,
            "alpha": 100.0 if model == "har" else selected_alphas[model],
            "val_qlike": float(qlike_loss(y[val_mask], ensemble[val_mask]).mean()),
            "val_rmse": float(np.sqrt(np.mean((y[val_mask] - ensemble[val_mask]) ** 2))),
        })

    val_manifest = manifest.loc[val_mask, ["sample_id", "episode_id"]].reset_index(drop=True)
    episode_rows = []
    reference = ensemble_predictions["har"][val_mask]
    reference_loss = qlike_loss(y[val_mask], reference).mean(axis=1)
    for model, prediction in ensemble_predictions.items():
        model_loss = qlike_loss(y[val_mask], prediction[val_mask]).mean(axis=1)
        frame = val_manifest.copy()
        frame["model_loss"] = model_loss
        frame["har_loss"] = reference_loss
        for episode_id, group in frame.groupby("episode_id"):
            episode_rows.append({
                "fold": fold,
                "episode_id": episode_id,
                "model": model,
                "episode_qlike": float(group["model_loss"].mean()),
                "delta_vs_har": float((group["model_loss"] - group["har_loss"]).mean()),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(episode_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested rolling-origin evaluation using prior folds for alpha selection.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-run", type=Path, required=True)
    parser.add_argument("--date-column", default="event_onset")
    parser.add_argument("--group-column", default="episode_id")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/stage_e_nested_rolling_origin"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    manifest, sequences = ROLLING.load_stage_d(args.stage_d_run)
    assignments = pd.read_csv(args.rolling_run / "rolling_fold_manifest.csv")
    rolling_results = pd.read_csv(args.rolling_run / "rolling_results_by_seed.csv")
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)

    metric_frames = []
    episode_frames = []
    selections = []
    for fold in sorted(assignments["fold"].unique()):
        fold = int(fold)
        if fold == 1:
            continue
        selected = select_nested_alphas(rolling_results, fold)
        selections.append({"evaluation_fold": fold, **selected})
        fold_manifest = assignments[assignments["fold"].eq(fold)].reset_index(drop=True)
        metrics, episodes = evaluate_nested_fold(
            fold_manifest,
            sequences,
            fold=fold,
            selected_alphas=selected,
            seeds=tuple(args.seeds),
        )
        metric_frames.append(metrics)
        episode_frames.append(episodes)

    metrics = pd.concat(metric_frames, ignore_index=True)
    episodes = pd.concat(episode_frames, ignore_index=True)
    selections_frame = pd.DataFrame(selections)
    aggregate = metrics.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_val_qlike=("val_qlike", "mean"),
        std_across_folds=("val_qlike", "std"),
        mean_val_rmse=("val_rmse", "mean"),
    ).sort_values("mean_val_qlike")
    paired = episodes.groupby("model", as_index=False).agg(
        episodes=("episode_id", "count"),
        mean_delta_vs_har=("delta_vs_har", "mean"),
        median_delta_vs_har=("delta_vs_har", "median"),
        probability_better_than_har=("delta_vs_har", lambda x: float((x < 0).mean())),
    )

    selections_frame.to_csv(run_dir / "nested_selected_alphas.csv", index=False)
    metrics.to_csv(run_dir / "nested_metrics_by_fold.csv", index=False)
    episodes.to_csv(run_dir / "nested_episode_deltas.csv", index=False)
    aggregate.to_csv(run_dir / "nested_aggregate.csv", index=False)
    paired.to_csv(run_dir / "nested_paired_summary.csv", index=False)
    payload = {
        "stage_d_run": str(args.stage_d_run),
        "rolling_run": str(args.rolling_run),
        "test_evaluated": False,
        "evaluation_folds": sorted(metrics["fold"].unique().tolist()),
        "aggregate": aggregate.to_dict("records"),
        "paired": paired.to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
