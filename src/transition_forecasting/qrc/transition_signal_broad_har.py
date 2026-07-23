from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.control_strata import HARD_NEGATIVE
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.ladder_finite_shot_sampling import load_probability_cache
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import _select_rows_for_fold
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    load_rolling_fold_dataset,
)
from transition_forecasting.qrc.transition_signal_readout_assay import (
    MODEL_SPECS,
    TransitionSignalAssayConfig,
    _candidate_metrics,
    _freeze_recommendation,
    _load_source_parameters,
    _pooled_predictions_metrics,
    _strata,
    apply_segmented_calibration,
    select_transition_signal_configuration,
)


def evaluate_exact_matrix_broad_har(
    matrix: np.ndarray,
    *,
    frame: pd.DataFrame,
    config: TransitionSignalAssayConfig,
    model_name: str,
    readout_kind: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Preserve the established all-row HAR baseline; exclude hard negatives only from QRC fitting/calibration."""
    train = frame["fold_split"].eq("train").to_numpy()
    validation = frame["fold_split"].eq("val").to_numpy()
    if not train.any() or not validation.any():
        raise RuntimeError("selected frame requires nonempty train and validation rows")

    y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    har = _fit_har(frame, y, train)
    residuals, residual_train = _prequential_har_residuals(
        frame,
        y,
        train,
        blocks=config.prequential_blocks,
    )
    diagnostics, candidates, correction, width = select_transition_signal_configuration(
        matrix,
        frame=frame,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=residual_train,
        readout_kind=readout_kind,
        config=config,
    )
    prediction = apply_segmented_calibration(
        har,
        correction,
        early_lambda=float(diagnostics["early_lambda"]),
        late_lambda=float(diagnostics["late_lambda"]),
        split_horizon=config.split_horizon,
    )
    validation_metrics = _candidate_metrics(
        frame=frame,
        y=y,
        har=har,
        prediction=prediction,
        correction=correction,
        tune_mask=validation,
    )
    strata = _strata(frame)
    metric_row: dict[str, object] = {
        "fold": int(frame["fold"].iloc[0]),
        "model_name": model_name,
        "readout_kind": readout_kind,
        "validation_rows": int(validation.sum()),
        "hard_negative_validation_rows": int(
            (validation & (strata == HARD_NEGATIVE)).sum()
        ),
        "ridge_alpha": float(diagnostics["ridge_alpha"]),
        "transition_weight": float(diagnostics["transition_weight"]),
        "fit_intercept": bool(diagnostics["fit_intercept"]),
        "early_lambda": float(diagnostics["early_lambda"]),
        "late_lambda": float(diagnostics["late_lambda"]),
        "design_feature_width": int(width),
        "selected_inner_eligible": bool(diagnostics["eligible"]),
        **validation_metrics,
    }
    candidates = candidates.copy()
    candidates.insert(0, "fold", int(frame["fold"].iloc[0]))
    candidates.insert(1, "model_name", model_name)

    selected = frame.loc[validation].reset_index(drop=True)
    horizons = len(TARGET_COLUMNS)
    predictions = pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(
                selected["sample_id"].astype(str).to_numpy(), horizons
            ),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "evaluation_stratum": np.repeat(
                selected["evaluation_stratum"].astype(str).to_numpy(), horizons
            ),
            "episode_id": np.repeat(
                selected["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                selected["origin_date"].astype(str).to_numpy(), horizons
            ),
            "model_name": np.repeat(
                model_name, int(validation.sum()) * horizons
            ),
            "horizon": np.tile(
                np.arange(1, horizons + 1), int(validation.sum())
            ),
            "y_true": y[validation].reshape(-1),
            "y_pred": prediction[validation].reshape(-1),
            "har_pred": har[validation].reshape(-1),
            "qrc_correction": (prediction - har)[validation].reshape(-1),
            "raw_qrc_correction": correction[validation].reshape(-1),
        }
    )
    return metric_row, candidates, predictions


def run_transition_signal_broad_har_assay(
    *,
    source_run: Path,
    results_root: Path,
    config: TransitionSignalAssayConfig = TransitionSignalAssayConfig(),
    run_id: str | None = None,
) -> Path:
    """Reanalyse cached exact ladder modes while preserving the established HAR baseline."""
    config.validate()
    source_run = Path(source_run)
    source_parameters = _load_source_parameters(source_run)
    fold_dir = Path(str(source_parameters["fold_dir"]))
    dataset = load_rolling_fold_dataset(fold_dir)
    cache_manifest = pd.read_csv(source_run / "cache_manifest.csv")
    config_payload = config.to_dict()
    config_payload["har_scope"] = "all_selected_rows"
    run_dir = begin_run(
        results_root,
        {
            "source_run": str(source_run),
            "source_cache_manifest": str(source_run / "cache_manifest.csv"),
            "fold_dir": str(fold_dir),
            "assay": config_payload,
            "har_scope": "all_selected_rows",
            "hard_negative_role": "included_in_har; excluded_from_qrc_fit_and_calibration; validation_diagnostic",
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    fold_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("broad-HAR assay must not receive test rows")
        cache_row = cache_manifest.loc[cache_manifest["fold"].eq(int(fold))]
        if len(cache_row) != 1:
            raise ValueError(f"source cache manifest requires one row for fold {fold}")
        cache = load_probability_cache(Path(str(cache_row.iloc[0]["cache_path"])))
        cached_ids = np.asarray(cache["sample_id"]).astype(str)
        selected_ids = frame["sample_id"].astype(str).to_numpy()
        if set(cached_ids) != set(selected_ids):
            raise RuntimeError(f"fold {fold}: source cache and selected panel differ")
        by_id = frame.set_index(frame["sample_id"].astype(str), drop=False)
        frame = by_id.loc[cached_ids].reset_index(drop=True)
        matrix = np.asarray(cache["exact_modes"], dtype=float)
        for model_name, readout_kind in MODEL_SPECS:
            metric_row, candidates, predictions = evaluate_exact_matrix_broad_har(
                matrix,
                frame=frame,
                config=config,
                model_name=model_name,
                readout_kind=readout_kind,
            )
            fold_rows.append(metric_row)
            candidate_frames.append(candidates)
            prediction_frames.append(predictions)

    fold_metrics = pd.DataFrame(fold_rows)
    inner_candidates = pd.concat(candidate_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = _pooled_predictions_metrics(predictions)
    recommendation = _freeze_recommendation(pooled, config)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    inner_candidates.to_csv(run_dir / "inner_candidates.csv.gz", index=False)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False)
    pooled.to_csv(run_dir / "pooled_metrics_by_stratum_lead.csv", index=False)
    (run_dir / "freeze_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n"
    )
    summary = {
        "source_run": str(source_run),
        "folds": list(config.folds),
        "models": [name for name, _ in MODEL_SPECS],
        "prediction_rows": int(len(predictions)),
        "har_scope": "all_selected_rows",
        "hard_negatives_used_for_har": True,
        "hard_negatives_used_for_qrc_fit_or_calibration": False,
        "test_evaluated": False,
        "recommendation": recommendation,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
