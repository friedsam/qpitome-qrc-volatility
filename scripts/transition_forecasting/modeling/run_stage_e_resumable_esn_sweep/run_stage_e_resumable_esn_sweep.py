from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    har_predictions,
    load_rematched_rolling,
    metrics,
    scale_sequences,
)

METRIC_COLUMNS = (
    "val_qlike",
    "val_rmse",
    "mz_intercept",
    "mz_slope",
    "mz_r2",
    "mz_horizon_mean_abs_slope_error",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _parse_pls_specs(values: list[str]) -> tuple[tuple[int, float], ...]:
    parsed: list[tuple[int, float]] = []
    for value in values:
        pieces = value.split(":")
        if len(pieces) != 2:
            raise ValueError(f"invalid PLS specification {value!r}; expected COMPONENTS:SHRINKAGE")
        components = int(pieces[0])
        shrinkage = float(pieces[1])
        if components < 1 or not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"invalid PLS specification {value!r}")
        parsed.append((components, shrinkage))
    return tuple(parsed)


def _shuffle_sequences(sequences: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 20000)
    shuffled = sequences.copy()
    for sample in range(len(shuffled)):
        shuffled[sample] = shuffled[sample, rng.permutation(shuffled.shape[1]), :]
    return shuffled


def _diagnostics(
    *,
    task: dict[str, Any],
    order: str,
    representation: str,
    alpha: float | None,
    components: int | None,
    shrinkage: float | None,
    y: np.ndarray,
    prediction: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, Any]:
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        **task,
        "order": order,
        "representation": representation,
        "alpha": np.nan if alpha is None else float(alpha),
        "components": np.nan if components is None else int(components),
        "shrinkage": np.nan if shrinkage is None else float(shrinkage),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "val_qlike": qlike,
        "val_rmse": rmse,
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def _evaluate_states(
    *,
    task: dict[str, Any],
    order: str,
    states: np.ndarray,
    y: np.ndarray,
    har: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alphas: tuple[float, ...],
    pca_components: tuple[int, ...],
    pls_specs: tuple[tuple[int, float], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scaler = StandardScaler()
    state_train = scaler.fit_transform(states[train_mask])
    state_all = scaler.transform(states)

    for alpha in alphas:
        readout = Ridge(alpha=float(alpha))
        readout.fit(state_train, residual[train_mask])
        prediction = har + readout.predict(state_all)
        rows.append(
            _diagnostics(
                task=task,
                order=order,
                representation="raw_ridge",
                alpha=alpha,
                components=None,
                shrinkage=None,
                y=y,
                prediction=prediction,
                train_mask=train_mask,
                val_mask=val_mask,
            )
        )

    for requested in pca_components:
        count = min(int(requested), state_train.shape[0], state_train.shape[1])
        if count < 1:
            continue
        pca = PCA(n_components=count, svd_solver="full")
        pca_train = pca.fit_transform(state_train)
        pca_all = pca.transform(state_all)
        for alpha in alphas:
            readout = Ridge(alpha=float(alpha))
            readout.fit(pca_train, residual[train_mask])
            prediction = har + readout.predict(pca_all)
            rows.append(
                _diagnostics(
                    task=task,
                    order=order,
                    representation=f"pca{count}_ridge",
                    alpha=alpha,
                    components=count,
                    shrinkage=None,
                    y=y,
                    prediction=prediction,
                    train_mask=train_mask,
                    val_mask=val_mask,
                )
            )

    unique_components = sorted({components for components, _ in pls_specs})
    for components in unique_components:
        count = min(
            int(components),
            state_train.shape[0] - 1,
            state_train.shape[1],
            residual.shape[1],
        )
        if count != components:
            continue
        pls = PLSRegression(n_components=count, scale=False, max_iter=1000)
        pls.fit(state_train, residual[train_mask])
        correction = pls.predict(state_all)
        for requested_components, shrinkage in pls_specs:
            if requested_components != components:
                continue
            prediction = har + float(shrinkage) * correction
            rows.append(
                _diagnostics(
                    task=task,
                    order=order,
                    representation=f"pls{count}",
                    alpha=None,
                    components=count,
                    shrinkage=shrinkage,
                    y=y,
                    prediction=prediction,
                    train_mask=train_mask,
                    val_mask=val_mask,
                )
            )
    return rows


def _load_completed(checkpoint_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not checkpoint_dir.exists():
        return completed
    for path in checkpoint_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete":
            completed.add(str(payload["task_id"]))
    return completed


def _collect_rows(checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        statuses.append({key: value for key, value in payload.items() if key != "rows"})
        if payload.get("status") == "complete":
            rows.extend(payload.get("rows", []))
    return pd.DataFrame(rows), pd.DataFrame(statuses)


def _aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = [
        "n",
        "spectral_radius",
        "input_scale",
        "leak",
        "order",
        "representation",
        "alpha",
        "components",
        "shrinkage",
        "fold",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "mean") for column in METRIC_COLUMNS
    }
    aggregations.update(
        val_samples=("val_samples", "first"),
        train_samples=("train_samples", "first"),
        seeds=("seed", "nunique"),
    )
    per_fold = raw.groupby(keys, as_index=False, dropna=False).agg(**aggregations)

    pooled_keys = keys[:-1]
    pooled_rows: list[dict[str, Any]] = []
    for key, group in per_fold.groupby(pooled_keys, sort=True, dropna=False):
        row = dict(zip(pooled_keys, key, strict=True))
        weight = group["val_samples"].astype(float)
        row["folds"] = int(group["fold"].nunique())
        for column in METRIC_COLUMNS:
            row[f"weighted_{column}"] = float((group[column] * weight).sum() / weight.sum())
            row[f"median_{column}"] = float(group[column].median())
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)

    match_keys = [
        "n",
        "spectral_radius",
        "input_scale",
        "leak",
        "representation",
        "alpha",
        "components",
        "shrinkage",
    ]
    ordered = pooled[pooled["order"].eq("ordered")]
    shuffled = pooled[pooled["order"].eq("shuffled")]
    comparison = ordered.merge(
        shuffled,
        on=match_keys,
        suffixes=("_ordered", "_shuffled"),
        validate="one_to_one",
    )
    for column in ("weighted_val_qlike", "weighted_val_rmse", "weighted_mz_r2", "weighted_mz_horizon_mean_abs_slope_error"):
        comparison[f"ordered_minus_shuffled_{column}"] = (
            comparison[f"{column}_ordered"] - comparison[f"{column}_shuffled"]
        )
    return per_fold, pooled, comparison


def _refresh_outputs(run_dir: Path) -> None:
    raw, status = _collect_rows(run_dir / "checkpoints")
    _atomic_csv(run_dir / "task_status.csv", status)
    if raw.empty:
        return
    _atomic_csv(run_dir / "sweep_seed_results.csv", raw)
    per_fold, pooled, comparison = _aggregate(raw)
    _atomic_csv(run_dir / "sweep_fold_results.csv", per_fold)
    _atomic_csv(run_dir / "sweep_pooled_results.csv", pooled)
    _atomic_csv(run_dir / "sweep_ordered_shuffled.csv", comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Stage E ESN dynamics and readout sweep on development folds.")
    parser.add_argument("--chronology-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--spectral-radii", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    parser.add_argument("--input-scales", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--leaks", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0])
    parser.add_argument("--pca-components", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--pls-specs", nargs="+", default=["5:0.5", "10:0.75"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/transition_forecasting/modeling/run_stage_e_resumable_esn_sweep"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--refresh-every", type=int, default=10)
    parser.add_argument("--max-new-tasks", type=int)
    args = parser.parse_args()

    if any(size < 1 for size in args.sizes):
        raise ValueError("reservoir sizes must be positive")
    if any(not 0.0 < leak <= 1.0 for leak in args.leaks):
        raise ValueError("leaks must lie in (0, 1]")
    if any(radius <= 0.0 for radius in args.spectral_radii):
        raise ValueError("spectral radii must be positive")
    if any(scale <= 0.0 for scale in args.input_scales):
        raise ValueError("input scales must be positive")
    if any(alpha <= 0.0 for alpha in args.alphas):
        raise ValueError("alphas must be positive")

    run_dir = args.out_dir / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pls_specs = _parse_pls_specs(args.pls_specs)

    configuration = {
        "chronology_run": str(args.chronology_run),
        "folds": [int(value) for value in args.folds],
        "seeds": [int(value) for value in args.seeds],
        "sizes": [int(value) for value in args.sizes],
        "spectral_radii": [float(value) for value in args.spectral_radii],
        "input_scales": [float(value) for value in args.input_scales],
        "leaks": [float(value) for value in args.leaks],
        "alphas": [float(value) for value in args.alphas],
        "pca_components": [int(value) for value in args.pca_components],
        "pls_specs": [{"components": c, "shrinkage": s} for c, s in pls_specs],
        "test_rows_used": 0,
        "selection_scope": "development folds only",
        "checkpoint_policy": "one atomic JSON checkpoint per fold/configuration/seed",
    }
    config_path = run_dir / "configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != configuration:
            raise ValueError("run-id already exists with a different configuration")
    else:
        _atomic_text(config_path, json.dumps(configuration, indent=2) + "\n")

    manifest, tensors = load_rematched_rolling(args.chronology_run)
    fold_data: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold in (int(value) for value in args.folds):
        mask = manifest["fold"].eq(fold) & manifest["fold_split"].isin(["train", "val"])
        frame = manifest.loc[mask].reset_index(drop=True)
        sequences = tensors[mask.to_numpy()]
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har
        scaled = scale_sequences(sequences, train_mask)
        fold_data[fold] = (frame, scaled, train_mask, val_mask, y, har, residual)

    tasks: list[dict[str, Any]] = []
    for fold in (int(value) for value in args.folds):
        for size in (int(value) for value in args.sizes):
            for radius in (float(value) for value in args.spectral_radii):
                for input_scale in (float(value) for value in args.input_scales):
                    for leak in (float(value) for value in args.leaks):
                        for seed in (int(value) for value in args.seeds):
                            task = {
                                "fold": fold,
                                "n": size,
                                "spectral_radius": radius,
                                "input_scale": input_scale,
                                "leak": leak,
                                "seed": seed,
                            }
                            task["task_id"] = _stable_id(task)
                            tasks.append(task)

    completed = _load_completed(checkpoint_dir)
    pending = [task for task in tasks if task["task_id"] not in completed]
    if args.max_new_tasks is not None:
        pending = pending[: int(args.max_new_tasks)]
    print(json.dumps({"total_tasks": len(tasks), "completed": len(completed), "pending_this_run": len(pending)}, indent=2))

    completed_now = 0
    for task in pending:
        started = time.time()
        path = checkpoint_dir / f"{task['task_id']}.json"
        payload: dict[str, Any] = {**task, "status": "running", "started_unix": started}
        _atomic_text(path, json.dumps(payload, indent=2) + "\n")
        try:
            _, scaled, train_mask, val_mask, y, har, residual = fold_data[int(task["fold"])]
            w_in, w = make_esn_weights(
                n_inputs=scaled.shape[-1],
                n_reservoir=int(task["n"]),
                spectral_radius=float(task["spectral_radius"]),
                input_scale=float(task["input_scale"]),
                seed=int(task["seed"]),
            )
            ordered = esn_states(scaled, w_in, w, float(task["leak"]))
            shuffled_input = _shuffle_sequences(scaled, int(task["seed"]))
            shuffled = esn_states(shuffled_input, w_in, w, float(task["leak"]))
            rows: list[dict[str, Any]] = []
            rows.extend(
                _evaluate_states(
                    task=task,
                    order="ordered",
                    states=ordered,
                    y=y,
                    har=har,
                    residual=residual,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    alphas=tuple(float(value) for value in args.alphas),
                    pca_components=tuple(int(value) for value in args.pca_components),
                    pls_specs=pls_specs,
                )
            )
            rows.extend(
                _evaluate_states(
                    task=task,
                    order="shuffled",
                    states=shuffled,
                    y=y,
                    har=har,
                    residual=residual,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    alphas=tuple(float(value) for value in args.alphas),
                    pca_components=tuple(int(value) for value in args.pca_components),
                    pls_specs=pls_specs,
                )
            )
            payload = {
                **task,
                "status": "complete",
                "started_unix": started,
                "finished_unix": time.time(),
                "runtime_seconds": time.time() - started,
                "rows": rows,
            }
        except Exception as exc:  # continue the overnight run after isolated failures
            payload = {
                **task,
                "status": "failed",
                "started_unix": started,
                "finished_unix": time.time(),
                "runtime_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _atomic_text(path, json.dumps(payload, indent=2, allow_nan=True) + "\n")
        completed_now += 1
        print(f"[{completed_now}/{len(pending)}] {task['task_id']} {payload['status']} {payload['runtime_seconds']:.1f}s", flush=True)
        if completed_now % int(args.refresh_every) == 0:
            _refresh_outputs(run_dir)

    _refresh_outputs(run_dir)
    summary = {
        **configuration,
        "total_tasks": len(tasks),
        "completed_tasks": len(_load_completed(checkpoint_dir)),
        "failed_tasks": int(sum(1 for path in checkpoint_dir.glob("*.json") if json.loads(path.read_text()).get("status") == "failed")),
    }
    _atomic_text(run_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
