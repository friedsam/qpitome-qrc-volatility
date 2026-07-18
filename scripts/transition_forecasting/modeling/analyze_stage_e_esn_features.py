from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    EXTENDED_HAR_FEATURES,
    HAR_FEATURES,
    TARGET_COLUMNS,
    qlike_loss,
    validate_split_integrity,
)

PLS_COMPONENTS = (1, 2, 3, 5, 10)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _latest_stage_d_run(root: Path) -> Path:
    candidates = [
        path for path in root.iterdir()
        if path.is_dir()
        and (path / "sample_manifest.csv").exists()
        and (path / "sequence_tensors.npz").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no complete Stage D run found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _load_manifest(run_dir: Path) -> pd.DataFrame:
    if not (run_dir / "sequence_tensors.npz").exists():
        raise FileNotFoundError(f"incomplete Stage D run lacks sequence_tensors.npz: {run_dir}")
    manifest = pd.read_csv(run_dir / "sample_manifest.csv").reset_index(drop=True)
    validate_split_integrity(manifest)
    return manifest


def _load_state_file(path: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    # These NPZ files are generated locally by the paired regularization script.
    # The first cache version stored string metadata as NumPy object arrays, so
    # pickle support is required for backward-compatible reuse without rebuilding.
    with np.load(path, allow_pickle=True) as data:
        states = np.asarray(data["states"], dtype=float)
        sample_ids = data["sample_id"].astype(str)
        config = json.loads(str(data["config_json"].item()))
        seed = int(data["seed"].item())
    expected_ids = manifest["sample_id"].astype(str).to_numpy()
    if states.shape[0] != len(manifest) or not np.array_equal(sample_ids, expected_ids):
        raise ValueError(f"state cache does not align with Stage D manifest: {path}")
    return states, {**config, "seed": seed, "state_file": str(path)}


def _fit_har_prediction(manifest: pd.DataFrame, train_mask: np.ndarray, alpha: float = 100.0) -> np.ndarray:
    x = manifest[list(HAR_FEATURES)].to_numpy(dtype=float)
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
    return model.predict(scaler.transform(x))


def _mean_path_qlike(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(qlike_loss(y, prediction).mean())


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_state_file(
    states: np.ndarray,
    manifest: pd.DataFrame,
    metadata: dict[str, object],
    *,
    pls_components: tuple[int, ...] = PLS_COMPONENTS,
    ridge_alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> dict[str, pd.DataFrame]:
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()
    y = manifest[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har_prediction(manifest, train_mask)
    residual = y - har

    state_scaler = StandardScaler()
    z_train = state_scaler.fit_transform(states[train_mask])
    z_all = state_scaler.transform(states)

    max_components = min(z_train.shape[0], z_train.shape[1])
    pca = PCA(n_components=max_components, svd_solver="full")
    pc_train = pca.fit_transform(z_train)
    pc_all = pca.transform(z_all)

    common = {
        "config": metadata["name"],
        "seed": int(metadata["seed"]),
        "n_reservoir": int(metadata["n"]),
    }

    pca_rows: list[dict[str, object]] = []
    for index, ratio in enumerate(pca.explained_variance_ratio_):
        pc = pc_all[:, index]
        row = {
            **common,
            "pc": index + 1,
            "explained_variance_ratio": float(ratio),
            "cumulative_explained_variance": float(pca.explained_variance_ratio_[: index + 1].sum()),
            "corr_level": _safe_corr(pc[train_mask], manifest.loc[train_mask, "level"].to_numpy(dtype=float)),
            "corr_mean5": _safe_corr(pc[train_mask], manifest.loc[train_mask, "mean5"].to_numpy(dtype=float)),
            "corr_mean20": _safe_corr(pc[train_mask], manifest.loc[train_mask, "mean20"].to_numpy(dtype=float)),
            "max_abs_corr_har_prediction": max(
                abs(_safe_corr(pc[train_mask], har[train_mask, horizon]))
                for horizon in range(har.shape[1])
            ),
            "max_abs_corr_har_residual": max(
                abs(_safe_corr(pc[train_mask], residual[train_mask, horizon]))
                for horizon in range(residual.shape[1])
            ),
        }
        pca_rows.append(row)

    coordinate_rows: list[dict[str, object]] = []
    for feature in range(z_all.shape[1]):
        coordinate_rows.append({
            **common,
            "feature": feature,
            "max_abs_corr_har_residual": max(
                abs(_safe_corr(z_all[train_mask, feature], residual[train_mask, horizon]))
                for horizon in range(residual.shape[1])
            ),
            "max_abs_corr_har_prediction": max(
                abs(_safe_corr(z_all[train_mask, feature], har[train_mask, horizon]))
                for horizon in range(har.shape[1])
            ),
            "corr_level": _safe_corr(
                z_all[train_mask, feature], manifest.loc[train_mask, "level"].to_numpy(dtype=float)
            ),
        })

    reconstruction_targets = list(dict.fromkeys([*HAR_FEATURES, *EXTENDED_HAR_FEATURES]))
    reconstruction_rows: list[dict[str, object]] = []
    for target_name in reconstruction_targets:
        target = manifest[target_name].to_numpy(dtype=float)
        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha)
            model.fit(z_train, target[train_mask])
            prediction = model.predict(z_all[val_mask])
            reconstruction_rows.append({
                **common,
                "target": target_name,
                "alpha": alpha,
                "val_r2": float(r2_score(target[val_mask], prediction)),
            })

    for horizon in range(har.shape[1]):
        target = har[:, horizon]
        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha)
            model.fit(z_train, target[train_mask])
            prediction = model.predict(z_all[val_mask])
            reconstruction_rows.append({
                **common,
                "target": f"har_prediction_h{horizon + 1}",
                "alpha": alpha,
                "val_r2": float(r2_score(target[val_mask], prediction)),
            })

    pls_rows: list[dict[str, object]] = []
    max_pls = min(z_train.shape[1], z_train.shape[0] - 1)
    for n_components in pls_components:
        if n_components > max_pls:
            continue
        pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
        pls.fit(z_train, residual[train_mask])
        score_train = pls.transform(z_train)
        score_val = pls.transform(z_all[val_mask])
        for alpha in ridge_alphas:
            correction = Ridge(alpha=alpha, fit_intercept=False)
            correction.fit(score_train, residual[train_mask])
            train_prediction = har[train_mask] + correction.predict(score_train)
            val_prediction = har[val_mask] + correction.predict(score_val)
            pls_rows.append({
                **common,
                "n_components": n_components,
                "alpha": alpha,
                "train_qlike": _mean_path_qlike(y[train_mask], train_prediction),
                "val_qlike": _mean_path_qlike(y[val_mask], val_prediction),
                "val_rmse": float(np.sqrt(np.mean((y[val_mask] - val_prediction) ** 2))),
                "coefficient_norm": float(np.linalg.norm(correction.coef_)),
            })

    return {
        "pca": pd.DataFrame(pca_rows),
        "coordinates": pd.DataFrame(coordinate_rows),
        "reconstruction": pd.DataFrame(reconstruction_rows),
        "pls": pd.DataFrame(pls_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved Stage E ESN reservoir features.")
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_dataset"),
    )
    parser.add_argument("--stage-d-run", type=Path)
    parser.add_argument(
        "--reservoir-cache-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_esn_reservoir_features"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_esn_feature_assay"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    stage_d_run = args.stage_d_run or _latest_stage_d_run(args.stage_d_root)
    manifest = _load_manifest(stage_d_run)
    state_files = sorted(args.reservoir_cache_dir.glob("reservoir_states__*.npz"))
    if not state_files:
        raise FileNotFoundError(f"no reservoir state files found in {args.reservoir_cache_dir}")

    run_dir = begin_run(args.out_dir, vars(args) | {"stage_d_run": stage_d_run}, run_id=args.run_id)
    outputs: dict[str, list[pd.DataFrame]] = {name: [] for name in ("pca", "coordinates", "reconstruction", "pls")}
    for state_file in state_files:
        states, metadata = _load_state_file(state_file, manifest)
        result = analyze_state_file(states, manifest, metadata)
        for name, frame in result.items():
            outputs[name].append(frame)

    pca = pd.concat(outputs["pca"], ignore_index=True)
    coordinates = pd.concat(outputs["coordinates"], ignore_index=True)
    reconstruction = pd.concat(outputs["reconstruction"], ignore_index=True)
    pls = pd.concat(outputs["pls"], ignore_index=True)

    pca.to_csv(run_dir / "pca_diagnostics.csv", index=False)
    coordinates.to_csv(run_dir / "coordinate_relevance.csv", index=False)
    reconstruction.to_csv(run_dir / "feature_reconstruction.csv", index=False)
    pls.to_csv(run_dir / "residual_pls.csv", index=False)

    best_pls = (
        pls.groupby(["config", "n_reservoir", "n_components", "alpha"], as_index=False)
        .agg(mean_val_qlike=("val_qlike", "mean"), std_val_qlike=("val_qlike", "std"), mean_val_rmse=("val_rmse", "mean"))
        .sort_values(["mean_val_qlike", "mean_val_rmse"])
        .head(20)
    )
    best_pls.to_csv(run_dir / "residual_pls_seed_mean_top20.csv", index=False)

    summary = {
        "stage_d_run": str(stage_d_run),
        "reservoir_cache_dir": str(args.reservoir_cache_dir),
        "state_files": len(state_files),
        "test_evaluated": False,
        "best_pls_seed_mean": best_pls.head(1).to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
