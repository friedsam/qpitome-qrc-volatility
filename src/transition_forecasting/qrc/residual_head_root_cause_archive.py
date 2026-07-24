from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.qrc.frozen_chain_readout_tools import (
    chronological_inner_split,
)
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    select_inner_configuration,
)
from transition_forecasting.qrc.residual_head_root_cause_assay import (
    MODEL_SPECS,
    READOUTS,
    ResidualHeadRootCauseConfig,
    _direction_summary,
    _early_lambda_rows,
    _fit_readouts,
    _flatten_cells,
    _render_plots,
    _select_lambdas,
)


_ARCHIVE_PATTERN = re.compile(
    r"(?:^|/)feature_archives/ladder_symmetric_modes_fold_(\d+)\.npz$"
)
_REQUIRED_ARRAYS = {
    "fold",
    "sample_id",
    "fold_split",
    "lead",
    "label",
    "episode_id",
    "origin_date",
    "mode_matrix",
    "target_path",
    "har_prediction_path",
    "prequential_residual_path",
    "prequential_residual_valid",
}


def _bundle_payload(bundle: np.lib.npyio.NpzFile, *, source: str) -> dict[str, np.ndarray]:
    missing = _REQUIRED_ARRAYS.difference(bundle.files)
    if missing:
        raise ValueError(f"{source} is missing arrays: {sorted(missing)}")
    payload = {name: np.asarray(bundle[name]) for name in _REQUIRED_ARRAYS}
    rows = len(payload["sample_id"])
    row_arrays = {
        "fold": payload["fold"],
        "fold_split": payload["fold_split"],
        "lead": payload["lead"],
        "label": payload["label"],
        "episode_id": payload["episode_id"],
        "origin_date": payload["origin_date"],
        "mode_matrix": payload["mode_matrix"],
        "target_path": payload["target_path"],
        "har_prediction_path": payload["har_prediction_path"],
        "prequential_residual_path": payload["prequential_residual_path"],
        "prequential_residual_valid": payload["prequential_residual_valid"],
    }
    lengths = {name: len(value) for name, value in row_arrays.items()}
    if any(length != rows for length in lengths.values()):
        raise ValueError(f"{source} has inconsistent row counts: {lengths}")
    modes = np.asarray(payload["mode_matrix"], dtype=float)
    target = np.asarray(payload["target_path"], dtype=float)
    har = np.asarray(payload["har_prediction_path"], dtype=float)
    residual = np.asarray(payload["prequential_residual_path"], dtype=float)
    if modes.ndim != 2 or modes.shape[1] < 9:
        raise ValueError(f"{source} has invalid mode matrix shape {modes.shape}")
    if target.ndim != 2 or target.shape[1] != 10:
        raise ValueError(f"{source} has invalid target shape {target.shape}")
    if har.shape != target.shape or residual.shape != target.shape:
        raise ValueError(f"{source} target/HAR/residual paths are not aligned")
    return payload


def load_ladder_feature_archives(
    source: Path,
    *,
    folds: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Load fold feature archives from an assay directory or its ZIP package."""

    path = Path(source)
    requested = set(int(value) for value in folds)
    loaded: dict[int, dict[str, np.ndarray]] = {}
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches: dict[int, str] = {}
            for name in archive.namelist():
                match = _ARCHIVE_PATTERN.search(name)
                if match:
                    fold = int(match.group(1))
                    if fold in matches:
                        raise ValueError(f"ZIP contains duplicate feature archive for fold {fold}")
                    matches[fold] = name
            for fold in sorted(requested):
                if fold not in matches:
                    continue
                data = io.BytesIO(archive.read(matches[fold]))
                with np.load(data, allow_pickle=True) as bundle:
                    loaded[fold] = _bundle_payload(
                        bundle,
                        source=f"{path}!{matches[fold]}",
                    )
    elif path.is_dir():
        matches: dict[int, Path] = {}
        for candidate in path.rglob("ladder_symmetric_modes_fold_*.npz"):
            match = re.search(r"fold_(\d+)\.npz$", candidate.name)
            if match is None or candidate.parent.name != "feature_archives":
                continue
            fold = int(match.group(1))
            if fold in matches:
                raise ValueError(f"directory contains duplicate feature archive for fold {fold}")
            matches[fold] = candidate
        for fold in sorted(requested):
            if fold not in matches:
                continue
            with np.load(matches[fold], allow_pickle=True) as bundle:
                loaded[fold] = _bundle_payload(bundle, source=str(matches[fold]))
    else:
        raise FileNotFoundError(f"feature archive source does not exist: {path}")

    missing = sorted(requested.difference(loaded))
    if missing:
        raise FileNotFoundError(f"feature archive source is missing folds: {missing}")
    return loaded


def _frame_from_payload(payload: dict[str, np.ndarray], *, fold: int) -> pd.DataFrame:
    fold_values = np.asarray(payload["fold"], dtype=int)
    if not np.all(fold_values == int(fold)):
        raise ValueError(f"fold {fold}: archive contains inconsistent fold values")
    frame = pd.DataFrame(
        {
            "fold": fold_values,
            "sample_id": np.asarray(payload["sample_id"]).astype(str),
            "fold_split": np.asarray(payload["fold_split"]).astype(str),
            "lead": np.asarray(payload["lead"], dtype=int),
            "label": np.asarray(payload["label"], dtype=int),
            "episode_id": np.asarray(payload["episode_id"]).astype(str),
            "origin_date": np.asarray(payload["origin_date"]).astype(str),
        }
    )
    if frame["fold_split"].eq("test").any():
        raise RuntimeError("archive root-cause assay must not receive test rows")
    if not frame["fold_split"].eq("train").any():
        raise ValueError(f"fold {fold}: archive has no training rows")
    if not frame["fold_split"].eq("val").any():
        raise ValueError(f"fold {fold}: archive has no validation rows")
    return frame


def run_residual_head_root_cause_from_archive(
    *,
    feature_archive_source: Path,
    results_root: Path,
    config: ResidualHeadRootCauseConfig = ResidualHeadRootCauseConfig(),
    run_id: str | None = None,
) -> Path:
    """Run the signed-correction audit directly from the immutable handoff archive."""

    config.validate()
    readout_config = config.readout_config()
    readout_config.validate()
    archives = load_ladder_feature_archives(
        Path(feature_archive_source),
        folds=config.folds,
    )
    run_dir = begin_run(
        Path(results_root),
        {
            "feature_archive_source": str(feature_archive_source),
            "config": config.to_dict(),
            "source_contract": "ladder_readout_upgrade feature_archives",
            "new_quantum_simulation": False,
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    cell_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    intercept_rows: list[dict[str, object]] = []

    for fold in config.folds:
        payload = archives[int(fold)]
        frame = _frame_from_payload(payload, fold=int(fold))
        y = np.asarray(payload["target_path"], dtype=float)
        har = np.asarray(payload["har_prediction_path"], dtype=float)
        residuals = np.asarray(payload["prequential_residual_path"], dtype=float)
        residual_train = np.asarray(
            payload["prequential_residual_valid"], dtype=bool
        )
        validation = frame["fold_split"].eq("val").to_numpy()
        inner_fit, inner_tune = chronological_inner_split(
            frame["origin_date"].astype(str).to_numpy(),
            residual_train,
            holdout_fraction=config.inner_holdout_fraction,
        )
        full_modes = np.asarray(payload["mode_matrix"], dtype=float)

        for model_family, indices in MODEL_SPECS.items():
            matrix = full_modes[:, indices]
            diagnostics, _, _, _ = select_inner_configuration(
                matrix,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train,
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                readout_kind="linear",
                calibration_kind="segmented",
                config=readout_config,
            )
            inner_predictions, inner_intercept = _fit_readouts(
                matrix,
                residuals,
                inner_fit,
                alpha=config.ridge_alpha,
            )
            full_predictions, full_intercept = _fit_readouts(
                matrix,
                residuals,
                residual_train,
                alpha=config.ridge_alpha,
            )
            for horizon, value in enumerate(full_intercept, start=1):
                intercept_rows.append(
                    {
                        "fold": int(fold),
                        "model_family": model_family,
                        "horizon": int(horizon),
                        "intercept": float(value),
                        "inner_intercept": float(inner_intercept[horizon - 1]),
                    }
                )

            for readout in READOUTS:
                early_lambda, late_lambda, candidates = _select_lambdas(
                    y,
                    har,
                    inner_predictions[readout],
                    inner_tune,
                    config,
                )
                candidates.insert(0, "fold", int(fold))
                candidates.insert(1, "model_family", model_family)
                candidates.insert(2, "readout", readout)
                lambda_frames.append(candidates)
                selection_rows.append(
                    {
                        "fold": int(fold),
                        "model_family": model_family,
                        "readout": readout,
                        "selected_early_lambda": early_lambda,
                        "selected_transition_lambda": late_lambda,
                        "production_selected_early_lambda": float(
                            diagnostics["early_lambda"]
                        ),
                        "production_selected_transition_lambda": float(
                            diagnostics["transition_lambda"]
                        ),
                    }
                )
                cell_frames.extend(
                    [
                        _flatten_cells(
                            frame,
                            y,
                            har,
                            inner_predictions[readout],
                            inner_tune,
                            fold=int(fold),
                            model_family=model_family,
                            readout=readout,
                            population="inner_tune",
                            split_horizon=config.split_horizon,
                        ),
                        _flatten_cells(
                            frame,
                            y,
                            har,
                            full_predictions[readout],
                            validation,
                            fold=int(fold),
                            model_family=model_family,
                            readout=readout,
                            population="validation",
                            split_horizon=config.split_horizon,
                        ),
                    ]
                )

    cells = pd.concat(cell_frames, ignore_index=True)
    lambda_candidates = pd.concat(lambda_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    intercepts = pd.DataFrame(intercept_rows)
    early_lambda = pd.concat(
        [
            _early_lambda_rows(group, config)
            for _, group in cells.loc[cells["population"].eq("inner_tune")].groupby(
                ["fold", "model_family", "readout"], sort=True
            )
        ],
        ignore_index=True,
    )
    direction = _direction_summary(cells)
    plots = _render_plots(early_lambda, cells, intercepts, run_dir / "plots")

    selections.to_csv(run_dir / "selected_lambdas_by_fold.csv", index=False)
    lambda_candidates.to_csv(
        run_dir / "lambda_candidates.csv.gz", index=False, compression="gzip"
    )
    early_lambda.to_csv(run_dir / "early_lambda_subgroup_audit.csv", index=False)
    direction.to_csv(run_dir / "direction_summary.csv", index=False)
    intercepts.to_csv(run_dir / "ridge_intercepts.csv", index=False)
    cells.to_csv(
        run_dir / "raw_correction_cells.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "status": "residual_head_root_cause_archive_assay_complete",
        "test_rows_used": 0,
        "new_quantum_simulation": False,
        "source": str(feature_archive_source),
        "folds": list(config.folds),
        "plots": plots,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
