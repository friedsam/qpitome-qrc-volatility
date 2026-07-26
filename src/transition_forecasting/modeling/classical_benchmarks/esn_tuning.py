from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from baselines.esn_representation import transform_input
from transition_forecasting.modeling.classical_benchmarks.common import (
    REQUIRED_GROUPS,
    group_masks,
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    load_rematched_dataset,
    metric_row,
)
from transition_forecasting.modeling.classical_benchmarks.esn import (
    POOLING,
    REPRESENTATION,
    WASHOUT,
    _fold_rows,
    _ridge_path_predictions,
    _scale_channels,
    _seed_features,
)
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS

DEFAULT_SELECTION_FOLDS = (4, 5, 6)
DEFAULT_CONFIRMATION_FOLDS = (7, 8)
DEFAULT_COARSE_SEEDS = (1, 2)
DEFAULT_FINAL_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_COARSE_ALPHAS = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
DEFAULT_FINE_ALPHAS = (300.0, 600.0, 1000.0, 1500.0, 3000.0, 6000.0, 10000.0, 20000.0, 40000.0, 80000.0, 120000.0, 160000.0, 240000.0, 320000.0)

# Curated around the two dynamical regimes already established by the historical
# ESN screens: fast/weak-input and persistent/strong-input. This is intentionally
# not a broad architecture search.
DEFAULT_CONFIGS = (
    {"config_id": "short_base", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_sr040", "n": 300, "connectivity": 0.02, "sr": 0.40, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_sr070", "n": 300, "connectivity": 0.02, "sr": 0.70, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_leak065", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.65},
    {"config_id": "short_leak095", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.95},
    {"config_id": "short_inp010", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.10, "leak": 0.80},
    {"config_id": "short_inp030", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.30, "leak": 0.80},
    {"config_id": "short_inp045", "n": 300, "connectivity": 0.02, "sr": 0.55, "inp": 0.45, "leak": 0.80},
    {"config_id": "short_conn005", "n": 300, "connectivity": 0.005, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_conn010", "n": 300, "connectivity": 0.01, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_conn050", "n": 300, "connectivity": 0.05, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_n100", "n": 100, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_n150", "n": 150, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "short_n200", "n": 200, "connectivity": 0.02, "sr": 0.55, "inp": 0.20, "leak": 0.80},
    {"config_id": "strong_base", "n": 300, "connectivity": 0.02, "sr": 0.90, "inp": 0.90, "leak": 0.30},
    {"config_id": "strong_balanced", "n": 300, "connectivity": 0.02, "sr": 0.75, "inp": 0.70, "leak": 0.40},
    {"config_id": "strong_critical", "n": 300, "connectivity": 0.02, "sr": 1.05, "inp": 0.70, "leak": 0.30},
    {"config_id": "strong_inp045", "n": 300, "connectivity": 0.02, "sr": 0.90, "inp": 0.45, "leak": 0.30},
    {"config_id": "strong_inp070", "n": 300, "connectivity": 0.02, "sr": 0.90, "inp": 0.70, "leak": 0.30},
    {"config_id": "strong_leak050", "n": 300, "connectivity": 0.02, "sr": 0.90, "inp": 0.90, "leak": 0.50},
    {"config_id": "strong_n150", "n": 150, "connectivity": 0.02, "sr": 0.90, "inp": 0.90, "leak": 0.30},
    {"config_id": "strong_conn010", "n": 300, "connectivity": 0.01, "sr": 0.90, "inp": 0.90, "leak": 0.30},
    {"config_id": "hybrid_mid", "n": 200, "connectivity": 0.02, "sr": 0.70, "inp": 0.45, "leak": 0.65},
    {"config_id": "hybrid_fast_sparse", "n": 200, "connectivity": 0.01, "sr": 0.40, "inp": 0.10, "leak": 0.95},
)

DEFAULT_SEQUENCE_RIDGE_GUARDRAILS = {
    "Transition": {"qlike": 1.653876, "rmse": 0.593431},
    "Controls": {"qlike": 0.582816, "rmse": 0.458946},
    "Pooled": {"qlike": 0.855299, "rmse": 0.496626},
}


def _cache_path(cache_root: Path, config_id: str, fold: int, seed: int, order: str) -> Path:
    return cache_root / config_id / f"fold{fold}_seed{seed}_{order}.npz"


def _build_seed_cache(dataset, config, fold, seed, alphas, cache_root: Path) -> None:
    ordered_path = _cache_path(cache_root, str(config["config_id"]), fold, seed, "ordered")
    shuffled_path = _cache_path(cache_root, str(config["config_id"]), fold, seed, "shuffled")
    if ordered_path.exists() and shuffled_path.exists():
        with np.load(ordered_path, allow_pickle=False) as cached:
            if np.array_equal(cached["alphas"], np.asarray(alphas, dtype=float)):
                return

    frame, sequences, train_mask, val_mask = _fold_rows(dataset, fold)
    target = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    inputs = _scale_channels(transform_input(sequences, REPRESENTATION), train_mask)
    ordered, shuffled = _seed_features(inputs, config, seed)
    for features, path in ((ordered, ordered_path), (shuffled, shuffled_path)):
        predictions = _ridge_path_predictions(features, target, train_mask, val_mask, alphas)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alphas=np.asarray(alphas, dtype=float),
            predictions=np.stack([predictions[float(alpha)] for alpha in alphas], axis=0),
            sample_id=frame.loc[val_mask, "sample_id"].astype(str).to_numpy(),
        )


def _load_seed_prediction(cache_root: Path, config_id: str, fold: int, seed: int, order: str, alpha: float) -> np.ndarray:
    with np.load(_cache_path(cache_root, config_id, fold, seed, order), allow_pickle=False) as cached:
        indices = np.where(np.isclose(cached["alphas"], float(alpha)))[0]
        if len(indices) != 1:
            raise KeyError(f"alpha {alpha} missing or duplicated in ESN prediction cache")
        return np.asarray(cached["predictions"][indices[0]], dtype=float)


def _fold_group_metrics(dataset, config_id: str, fold: int, seeds, alphas, cache_root: Path) -> pd.DataFrame:
    frame, _, _, val_mask = _fold_rows(dataset, fold)
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    target = val_frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for order in ("ordered", "shuffled"):
        for alpha in alphas:
            prediction = np.mean(
                [_load_seed_prediction(cache_root, config_id, fold, seed, order, alpha) for seed in seeds],
                axis=0,
            )
            masks = group_masks(val_frame)
            for group in REQUIRED_GROUPS:
                mask = masks[group]
                row = metric_row(
                    model=order,
                    group=group,
                    y_true=target[mask],
                    y_pred=prediction[mask],
                    fold=fold,
                )
                row.update({"config_id": config_id, "alpha": float(alpha), "order": order})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_candidates(
    fold_metrics: pd.DataFrame,
    *,
    guardrails: dict[str, dict[str, float]] = DEFAULT_SEQUENCE_RIDGE_GUARDRAILS,
) -> pd.DataFrame:
    """Rank ordered ESNs by transition QLIKE with explicit baseline guardrails."""
    summary = (
        fold_metrics.groupby(["config_id", "alpha", "order", "group"], as_index=False)
        .agg(
            mean_qlike=("qlike", "mean"),
            std_qlike=("qlike", "std"),
            mean_rmse=("rmse", "mean"),
            mean_mz_beta=("mz_beta", "mean"),
            mean_mz_r2=("mz_r2", "mean"),
        )
    )
    wide = summary.pivot(
        index=["config_id", "alpha", "order"],
        columns="group",
        values=["mean_qlike", "std_qlike", "mean_rmse", "mean_mz_beta", "mean_mz_r2"],
    )
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    wide = wide.reset_index()
    ordered = wide[wide["order"].eq("ordered")].copy()
    shuffled = wide[wide["order"].eq("shuffled")][
        ["config_id", "alpha", "mean_qlike_Transition"]
    ].rename(columns={"mean_qlike_Transition": "shuffled_transition_qlike"})
    ordered = ordered.merge(shuffled, on=["config_id", "alpha"], validate="one_to_one")
    ordered["ordered_advantage_transition"] = (
        ordered["mean_qlike_Transition"] - ordered["shuffled_transition_qlike"]
    )
    ordered["transition_objective"] = (
        ordered["mean_qlike_Transition"] + 0.25 * ordered["std_qlike_Transition"].fillna(0.0)
    )
    ordered["admissible"] = (
        (ordered["mean_qlike_Controls"] <= guardrails["Controls"]["qlike"] + 0.03)
        & (ordered["mean_rmse_Controls"] <= guardrails["Controls"]["rmse"] + 0.005)
        & (ordered["mean_qlike_Pooled"] <= guardrails["Pooled"]["qlike"] + 0.03)
        & (ordered["mean_rmse_Pooled"] <= guardrails["Pooled"]["rmse"] + 0.005)
        & (ordered["mean_rmse_Transition"] <= guardrails["Transition"]["rmse"] + 0.005)
        & (ordered["ordered_advantage_transition"] < 0.0)
    )
    return ordered.sort_values(
        ["admissible", "transition_objective", "mean_qlike_Pooled", "mean_rmse_Transition"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )


def evaluate_candidate_set(dataset, configs, folds, seeds, alphas, cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for config in configs:
        for fold in folds:
            for seed in seeds:
                _build_seed_cache(dataset, config, fold, seed, alphas, cache_root)
            frames.append(
                _fold_group_metrics(
                    dataset,
                    str(config["config_id"]),
                    fold,
                    seeds,
                    alphas,
                    cache_root,
                )
            )
    fold_metrics = pd.concat(frames, ignore_index=True)
    return fold_metrics, summarize_candidates(fold_metrics)


def _top_unique_configs(ranking: pd.DataFrame, configs, count: int) -> tuple[dict[str, object], ...]:
    ids: list[str] = []
    for config_id in ranking["config_id"].astype(str):
        if config_id not in ids:
            ids.append(config_id)
        if len(ids) == count:
            break
    return tuple(dict(config) for config in configs if str(config["config_id"]) in ids)


def _predict_fold(dataset, config, alpha: float, fold: int, seeds, cache_root: Path) -> pd.DataFrame:
    for seed in seeds:
        _build_seed_cache(dataset, config, fold, seed, (alpha,), cache_root)
    frame, _, _, val_mask = _fold_rows(dataset, fold)
    val_frame = frame.loc[val_mask].reset_index(drop=True)
    target = val_frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    metadata = [
        "sample_id", "episode_id", "index", "market_group", "origin_date", "event_onset",
        "label", "lead", "control_stratum", "fold", "fold_split",
    ]
    rows: list[pd.DataFrame] = []
    for order, model in (("ordered", "esn_direct_tuned"), ("shuffled", "esn_shuffled_tuned")):
        prediction = np.mean(
            [
                _load_seed_prediction(
                    cache_root, str(config["config_id"]), fold, seed, order, alpha
                )
                for seed in seeds
            ],
            axis=0,
        )
        output = val_frame[metadata].copy()
        output.insert(0, "model", model)
        output["config_id"] = str(config["config_id"])
        output["alpha"] = float(alpha)
        output["seeds"] = ",".join(str(seed) for seed in seeds)
        for horizon in range(1, 11):
            output[f"actual_h{horizon}"] = target[:, horizon - 1]
            output[f"predicted_h{horizon}"] = prediction[:, horizon - 1]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def run_esn_tuning_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    run_id: str,
    configs=DEFAULT_CONFIGS,
    selection_folds=DEFAULT_SELECTION_FOLDS,
    confirmation_folds=DEFAULT_CONFIRMATION_FOLDS,
    coarse_seeds=DEFAULT_COARSE_SEEDS,
    final_seeds=DEFAULT_FINAL_SEEDS,
    coarse_alphas=DEFAULT_COARSE_ALPHAS,
    fine_alphas=DEFAULT_FINE_ALPHAS,
    finalist_count: int = 6,
) -> tuple[Path, dict[str, object]]:
    """Run the bounded transition-first ESN search and frozen confirmation."""
    started = time.perf_counter()
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_root = run_dir / "cache"
    dataset = load_rematched_dataset(dataset_root)
    params = {
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "coarse_seeds": list(coarse_seeds),
        "final_seeds": list(final_seeds),
        "coarse_alphas": list(coarse_alphas),
        "fine_alphas": list(fine_alphas),
        "representation": REPRESENTATION,
        "pooling": POOLING,
        "washout": WASHOUT,
        "selection_rule": (
            "transition mean-fold QLIKE plus 0.25 fold SD; pooled/control/RMSE/order guardrails"
        ),
        "test_evaluated": False,
        "configs": list(configs),
    }
    (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    coarse_started = time.perf_counter()
    coarse_metrics, coarse_ranking = evaluate_candidate_set(
        dataset, configs, selection_folds, coarse_seeds, coarse_alphas, cache_root
    )
    coarse_metrics.to_csv(run_dir / "coarse_metrics_by_fold.csv", index=False)
    coarse_ranking.to_csv(run_dir / "coarse_ranking.csv", index=False)
    (run_dir / "coarse_runtime.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - coarse_started}, indent=2) + "\n",
        encoding="utf-8",
    )

    finalists = _top_unique_configs(coarse_ranking, configs, finalist_count)
    fine_started = time.perf_counter()
    fine_metrics, fine_ranking = evaluate_candidate_set(
        dataset, finalists, selection_folds, final_seeds, fine_alphas, cache_root
    )
    fine_metrics.to_csv(run_dir / "fine_metrics_by_fold.csv", index=False)
    fine_ranking.to_csv(run_dir / "fine_ranking.csv", index=False)
    (run_dir / "fine_runtime.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - fine_started}, indent=2) + "\n",
        encoding="utf-8",
    )

    winner = fine_ranking.iloc[0]
    selected_config = next(
        dict(config) for config in finalists if str(config["config_id"]) == str(winner["config_id"])
    )
    selected_alpha = float(winner["alpha"])
    selected = {
        "selected_config": selected_config,
        "selected_alpha": selected_alpha,
        "selection_row": winner.to_dict(),
        "final_seeds": list(final_seeds),
    }
    (run_dir / "selected_spec.json").write_text(
        json.dumps(selected, indent=2, default=_json_scalar) + "\n", encoding="utf-8"
    )

    # Confirmation folds are accessed only after the specification is frozen.
    predictions = pd.concat(
        [
            _predict_fold(dataset, selected_config, selected_alpha, fold, final_seeds, cache_root)
            for fold in (*selection_folds, *confirmation_folds)
        ],
        ignore_index=True,
    )
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    selection = predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True)
    confirmation = predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True)
    metrics = pd.concat(
        [
            grouped_path_metrics(selection, stage="selection"),
            grouped_path_metrics(confirmation, stage="confirmation"),
            grouped_path_metrics(predictions, stage="development_all"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    grouped_fold_metrics(predictions).to_csv(run_dir / "metrics_by_fold.csv", index=False)
    pd.concat(
        [
            grouped_horizon_metrics(selection, stage="selection"),
            grouped_horizon_metrics(confirmation, stage="confirmation"),
            grouped_horizon_metrics(predictions, stage="development_all"),
        ],
        ignore_index=True,
    ).to_csv(run_dir / "metrics_by_horizon.csv", index=False)

    summary = {
        "status": "complete",
        "test_evaluated": False,
        "selected_config": selected_config,
        "selected_alpha": selected_alpha,
        "seeds": list(final_seeds),
        "wall_seconds": time.perf_counter() - started,
        "confirmation": metrics[
            metrics["stage"].eq("confirmation") & metrics["model"].eq("esn_direct_tuned")
        ].to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary


def _json_scalar(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
