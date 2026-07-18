from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.numpy_esn import esn_states, make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
    StageEData,
    qlike_loss,
    validate_split_integrity,
)

DEFAULT_ALPHAS = tuple(10.0 ** exponent for exponent in range(-4, 9))
DEFAULT_CONFIGS = (
    {"name": "n300_sr0.9", "n": 300, "sr": 0.9, "inp": 0.3, "leak": 0.3},
    {"name": "n500_sr0.9", "n": 500, "sr": 0.9, "inp": 0.2, "leak": 0.5},
)


def _latest_stage_d_run(root: Path) -> Path:
    candidates = [
        path for path in root.iterdir()
        if path.is_dir()
        and (path / "sample_manifest.csv").exists()
        and (path / "sequence_tensors.npz").exists()
    ]
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


def _scale_sequences(sequences: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = sequences[train_mask].reshape(-1, sequences.shape[-1])
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (sequences - mean[None, None, :]) / scale[None, None, :], mean, scale


def _har_predictions(
    manifest: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _matrix_diagnostics(features: np.ndarray) -> dict[str, float]:
    scaled = StandardScaler().fit_transform(features)
    singular = np.linalg.svd(scaled, compute_uv=False)
    positive = singular[singular > singular.max() * 1e-12]
    weights = singular**2
    probabilities = weights / weights.sum()
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))))
    return {
        "matrix_rank": int(np.linalg.matrix_rank(scaled)),
        "effective_rank": entropy_rank,
        "condition_number": float(positive[0] / positive[-1]) if len(positive) else float("inf"),
    }


def _state_cache_path(cache_dir: Path, config: dict[str, object], seed: int) -> Path:
    return cache_dir / f"reservoir_states__{config['name']}__seed{seed}.npz"


def _load_or_build_states(
    *,
    scaled_sequences: np.ndarray,
    manifest: pd.DataFrame,
    config: dict[str, object],
    seed: int,
    cache_dir: Path | None,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
) -> tuple[np.ndarray, str]:
    cache_path = None if cache_dir is None else _state_cache_path(cache_dir, config, seed)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    expected_feature_count = int(config["n"]) + scaled_sequences.shape[-1]

    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            states = np.asarray(cached["states"], dtype=float)
            cached_ids = cached["sample_id"].astype(str)
            cached_config = json.loads(str(cached["config_json"].item()))
        if states.shape != (len(manifest), expected_feature_count):
            raise ValueError(f"cached state shape mismatch in {cache_path}: {states.shape}")
        if not np.array_equal(cached_ids, sample_ids):
            raise ValueError(f"cached sample IDs do not match current Stage D data in {cache_path}")
        expected = {key: config[key] for key in ("name", "n", "sr", "inp", "leak")}
        if cached_config != expected:
            raise ValueError(f"cached reservoir configuration mismatch in {cache_path}")
        return states, "loaded"

    W_in, W = make_esn_weights(
        n_inputs=scaled_sequences.shape[-1],
        n_reservoir=int(config["n"]),
        spectral_radius=float(config["sr"]),
        input_scale=float(config["inp"]),
        seed=int(seed),
    )
    states = esn_states(scaled_sequences, W_in, W, float(config["leak"]))
    if states.shape != (len(manifest), expected_feature_count):
        raise ValueError(f"generated state shape mismatch: {states.shape}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        config_json = json.dumps({key: config[key] for key in ("name", "n", "sr", "inp", "leak")}, sort_keys=True)
        np.savez_compressed(
            cache_path,
            states=np.asarray(states, dtype=np.float64),
            sample_id=sample_ids,
            split=manifest["split"].astype(str).to_numpy(),
            episode_id=manifest["episode_id"].astype(str).to_numpy(),
            label=manifest["label"].to_numpy(),
            lead=manifest["lead"].to_numpy(),
            input_mean=np.asarray(input_mean, dtype=float),
            input_scale=np.asarray(input_scale, dtype=float),
            config_json=np.asarray(config_json),
            seed=np.asarray(seed),
        )
    return states, "built"


def run_regularization_audit(
    data: StageEData,
    *,
    configs: tuple[dict[str, object], ...] = DEFAULT_CONFIGS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    seeds: tuple[int, ...] = (1, 2, 3),
    har_alpha: float = 100.0,
    reservoir_cache_dir: Path | None = None,
) -> pd.DataFrame:
    manifest = data.manifest.reset_index(drop=True)
    validate_split_integrity(manifest)
    sequences = np.asarray(data.sequences, dtype=float)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    scaled_sequences, input_mean, input_scale = _scale_sequences(sequences, train_mask)
    har = _har_predictions(manifest, y, train_mask, har_alpha)

    rows: list[dict[str, object]] = []
    for config in configs:
        for seed in seeds:
            states, state_source = _load_or_build_states(
                scaled_sequences=scaled_sequences,
                manifest=manifest,
                config=config,
                seed=int(seed),
                cache_dir=reservoir_cache_dir,
                input_mean=input_mean,
                input_scale=input_scale,
            )
            diagnostics = _matrix_diagnostics(states[train_mask])
            for formulation, baseline in (("direct", None), ("har_residual", har)):
                train_target = y[train_mask] if baseline is None else y[train_mask] - baseline[train_mask]
                train_base = np.zeros_like(y[train_mask]) if baseline is None else baseline[train_mask]
                val_base = np.zeros_like(y[val_mask]) if baseline is None else baseline[val_mask]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(states[train_mask])
                x_val = scaler.transform(states[val_mask])
                for alpha in alphas:
                    model = Ridge(alpha=float(alpha))
                    model.fit(x_train, train_target)
                    train_pred = train_base + model.predict(x_train)
                    val_pred = val_base + model.predict(x_val)
                    rows.append({
                        "config": config["name"],
                        "n_reservoir": int(config["n"]),
                        "spectral_radius": float(config["sr"]),
                        "input_scale": float(config["inp"]),
                        "leak": float(config["leak"]),
                        "seed": int(seed),
                        "state_source": state_source,
                        "formulation": formulation,
                        "alpha": float(alpha),
                        "train_qlike": float(qlike_loss(y[train_mask], train_pred).mean()),
                        "val_qlike": float(qlike_loss(y[val_mask], val_pred).mean()),
                        "train_rmse": float(np.sqrt(np.mean((y[train_mask] - train_pred) ** 2))),
                        "val_rmse": float(np.sqrt(np.mean((y[val_mask] - val_pred) ** 2))),
                        "coefficient_norm": float(np.linalg.norm(model.coef_)),
                        "val_prediction_std": float(np.std(val_pred)),
                        **diagnostics,
                    })
    return pd.DataFrame(rows)


def summarize(audit: pd.DataFrame) -> dict[str, object]:
    grouped = (
        audit.groupby(["config", "n_reservoir", "formulation", "alpha"], as_index=False)
        .agg(
            mean_val_qlike=("val_qlike", "mean"),
            std_val_qlike=("val_qlike", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_train_qlike=("train_qlike", "mean"),
            mean_coefficient_norm=("coefficient_norm", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_condition_number=("condition_number", "mean"),
        )
    )
    best = (
        grouped.sort_values(["formulation", "mean_val_qlike", "mean_val_rmse"])
        .groupby("formulation", as_index=False)
        .first()
    )
    return {
        "selection_split": "val",
        "test_evaluated": False,
        "purpose": "diagnostic regularization ladder; no architecture selection",
        "alpha_grid": sorted(audit["alpha"].unique().tolist()),
        "state_sources": sorted(audit["state_source"].unique().tolist()),
        "best_seed_averaged_rows": best.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ridge regularization on frozen Stage E ESN states.")
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_dataset"),
    )
    parser.add_argument("--stage-d-run", type=Path)
    parser.add_argument("--har-alpha", type=float, default=100.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_esn_regularization"),
    )
    parser.add_argument(
        "--reservoir-cache-dir",
        type=Path,
        help="Persistent directory for reusable reservoir-state NPZ files. Defaults to <run>/reservoir_features.",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    stage_d_run = args.stage_d_run or _latest_stage_d_run(args.stage_d_root)
    resolved = vars(args).copy()
    resolved["stage_d_run"] = stage_d_run
    run_dir = begin_run(args.out_dir, resolved, run_id=args.run_id)
    reservoir_cache_dir = args.reservoir_cache_dir or (run_dir / "reservoir_features")
    audit = run_regularization_audit(
        _load_stage_d_run(stage_d_run),
        seeds=tuple(args.seeds),
        har_alpha=args.har_alpha,
        reservoir_cache_dir=reservoir_cache_dir,
    )
    grouped = (
        audit.groupby(["config", "n_reservoir", "formulation", "alpha"], as_index=False)
        .agg(
            mean_train_qlike=("train_qlike", "mean"),
            mean_val_qlike=("val_qlike", "mean"),
            std_val_qlike=("val_qlike", "std"),
            mean_train_rmse=("train_rmse", "mean"),
            mean_val_rmse=("val_rmse", "mean"),
            mean_coefficient_norm=("coefficient_norm", "mean"),
            mean_prediction_std=("val_prediction_std", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_condition_number=("condition_number", "mean"),
        )
    )
    feature_manifest = (
        audit[["config", "n_reservoir", "spectral_radius", "input_scale", "leak", "seed", "state_source"]]
        .drop_duplicates()
        .sort_values(["config", "seed"])
        .assign(
            feature_file=lambda frame: frame.apply(
                lambda row: str(reservoir_cache_dir / f"reservoir_states__{row['config']}__seed{int(row['seed'])}.npz"),
                axis=1,
            )
        )
    )
    summary = summarize(audit)
    summary["reservoir_cache_dir"] = str(reservoir_cache_dir)
    audit.to_csv(run_dir / "regularization_by_seed.csv", index=False)
    grouped.to_csv(run_dir / "regularization_summary.csv", index=False)
    feature_manifest.to_csv(run_dir / "reservoir_feature_manifest.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage_d_run": str(stage_d_run), **summary}, indent=2))


if __name__ == "__main__":
    main()
