from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import (
    esn_states,
    fit_continuous_ridge_scores,
    historical_numpy_esn_grid,
    make_esn_weights,
)
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    StageEData,
    _metric_rows,
    qlike_loss,
    validate_split_integrity,
)


def _latest_stage_d_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "sample_manifest.csv").exists()
        and (path / "sequence_tensors.npz").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no Stage D run found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _load_stage_d_run(run_dir: Path) -> StageEData:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv")
    with np.load(run_dir / "sequence_tensors.npz", allow_pickle=True) as tensors:
        sequences = np.asarray(tensors["X"], dtype=float)
        tensor_ids = tensors["sample_id"].astype(str)
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if sequences.ndim != 3 or sequences.shape[1:] != (40, 1):
        raise ValueError(f"expected sequence tensor shape (n, 40, 1), got {sequences.shape}")
    if len(manifest) != len(sequences) or not np.array_equal(manifest_ids, tensor_ids):
        raise ValueError("sample manifest and tensor IDs are not aligned")
    return StageEData(manifest=manifest, sequences=sequences)


def _scale_sequences_train_only(
    sequences: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    train_values = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    scaled = (sequences - mean[None, None, :]) / scale[None, None, :]
    return scaled, {"mean": float(mean[0]), "scale": float(scale[0])}


def _fit_har_baseline(
    manifest: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    features = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features[train_mask], y[train_mask])
    return model.predict(features)


def _load_external_baseline(
    path: Path,
    manifest: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str).eq(model_name)].copy()
    predicted = [f"predicted_h{h}" for h in range(1, len(TARGET_COLUMNS) + 1)]
    required = {"sample_id", *predicted}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"external baseline missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("external baseline has duplicate sample IDs")
    aligned = manifest[["sample_id"]].merge(
        frame[["sample_id", *predicted]], on="sample_id", how="left", validate="one_to_one"
    )
    return aligned[predicted].to_numpy(dtype=float)


def run_stage_e_numpy_esn(
    data: StageEData,
    *,
    seeds: list[int],
    har_alpha: float = 100.0,
    garch_predictions: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = data.manifest.reset_index(drop=True)
    validate_split_integrity(manifest)
    sequences = np.asarray(data.sequences, dtype=float)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_train = y[train_mask]
    y_val = y[val_mask]
    val_frame = manifest.loc[val_mask].reset_index(drop=True)

    scaled_sequences, input_scaling = _scale_sequences_train_only(sequences, train_mask)
    har_prediction = _fit_har_baseline(manifest, y, train_mask, har_alpha)

    baselines: dict[str, np.ndarray | None] = {
        "direct": None,
        "har_residual": har_prediction,
    }
    if garch_predictions is not None:
        baselines["garch_residual"] = _load_external_baseline(
            garch_predictions, manifest, "garch_1_1_t"
        )

    tuning_rows: list[dict[str, object]] = []
    best: dict[str, dict[str, object]] = {}
    for config in historical_numpy_esn_grid(seeds):
        W_in, W = make_esn_weights(
            n_inputs=scaled_sequences.shape[-1],
            n_reservoir=int(config["n"]),
            spectral_radius=float(config["sr"]),
            input_scale=float(config["inp"]),
            seed=int(config["seed"]),
        )
        states = esn_states(scaled_sequences, W_in, W, float(config["leak"]))
        for formulation, baseline in baselines.items():
            if baseline is None:
                train_ok = train_mask.copy()
                train_target = y[train_ok]
                val_baseline = np.zeros_like(y_val)
            else:
                finite_baseline = np.all(np.isfinite(baseline), axis=1)
                train_ok = train_mask & finite_baseline
                if not train_ok.any():
                    raise ValueError(f"{formulation} has no finite training baseline predictions")
                if not np.all(finite_baseline[val_mask]):
                    raise ValueError(f"{formulation} has non-finite validation baseline predictions")
                train_target = y[train_ok] - baseline[train_ok]
                val_baseline = baseline[val_mask]

            scores = fit_continuous_ridge_scores(
                states[train_ok],
                train_target,
                {"val": states[val_mask]},
                alpha=float(config["alpha"]),
            )
            prediction = val_baseline + scores["val"]
            mean_qlike = float(qlike_loss(y_val, prediction).mean())
            rmse = float(np.sqrt(np.mean((y_val - prediction) ** 2)))
            row = {
                "formulation": formulation,
                **config,
                "n_train": int(train_ok.sum()),
                "val_mean_path_qlike": mean_qlike,
                "val_path_rmse": rmse,
            }
            tuning_rows.append(row)
            if formulation not in best or mean_qlike < float(best[formulation]["qlike"]):
                best[formulation] = {
                    "qlike": mean_qlike,
                    "prediction": prediction,
                    "config": dict(config),
                    "n_train": int(train_ok.sum()),
                }

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    selected: dict[str, object] = {}
    for formulation, result in best.items():
        model_name = f"numpy_esn_{formulation}"
        prediction = np.asarray(result["prediction"], dtype=float)
        output = val_frame[["sample_id", "episode_id", "label", "lead", "split"]].copy()
        output.insert(0, "model", model_name)
        for h in range(len(TARGET_COLUMNS)):
            output[f"actual_h{h + 1}"] = y_val[:, h]
            output[f"predicted_h{h + 1}"] = prediction[:, h]
        prediction_frames.append(output)
        metric_rows.extend(_metric_rows(model_name, val_frame, y_val, prediction))
        selected[formulation] = {
            "config": result["config"],
            "n_train": result["n_train"],
            "validation_mean_path_qlike": result["qlike"],
        }

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    tuning = pd.DataFrame(tuning_rows)
    pooled = metrics[(metrics["group_type"] == "pooled") & (metrics["horizon"] == "path")]
    summary = {
        "selection_split": "val",
        "test_evaluated": False,
        "primary_selection_metric": "pooled mean path QLIKE",
        "implementation": "src/baselines/numpy_esn.py",
        "input_scaling": input_scaling,
        "har_alpha": har_alpha,
        "garch_residual_included": garch_predictions is not None,
        "grid": "four frozen canonical configurations crossed with fixed seeds",
        "seeds": seeds,
        "selected": selected,
        "validation_ranking": pooled.sort_values(["qlike", "rmse"])[
            ["model", "qlike", "rmse", "peak_abs_error"]
        ].to_dict("records"),
    }
    return metrics, predictions, tuning, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical NumPy ESN direct and residual Stage E forecasts."
    )
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_dataset"),
    )
    parser.add_argument("--stage-d-run", type=Path)
    parser.add_argument("--garch-predictions", type=Path)
    parser.add_argument("--har-alpha", type=float, default=100.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_numpy_esn"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    stage_d_run = args.stage_d_run or _latest_stage_d_run(args.stage_d_root)
    resolved = vars(args).copy()
    resolved["stage_d_run"] = stage_d_run
    run_dir = begin_run(args.out_dir, resolved, run_id=args.run_id)
    data = _load_stage_d_run(stage_d_run)
    metrics, predictions, tuning, summary = run_stage_e_numpy_esn(
        data,
        seeds=args.seeds,
        har_alpha=args.har_alpha,
        garch_predictions=args.garch_predictions,
    )
    metrics.to_csv(run_dir / "validation_metrics.csv", index=False)
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    tuning.to_csv(run_dir / "validation_tuning.csv", index=False)
    (run_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage_d_run": str(stage_d_run), **summary}, indent=2))


if __name__ == "__main__":
    main()
