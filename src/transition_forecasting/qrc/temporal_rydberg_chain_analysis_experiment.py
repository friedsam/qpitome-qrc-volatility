from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
)
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    har_predictions,
    metrics,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
    controlled_level_windows,
    effective_rank,
    fit_level_rate_scaler,
    transform_level_windows,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    best_alpha_per_condition,
    feature_names_from_metadata,
    make_prediction_frame,
    signal_diagnostics,
    write_feature_archive,
    write_transition_overlays,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    TemporalRydbergExperimentConfig,
    _condition_seed,
    _load_excluded_ids,
    _select_rows_for_fold,
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)


def _metric_row(
    *,
    fold: int,
    condition: str,
    alpha: float | None,
    readout_mode: str,
    y: np.ndarray,
    prediction: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, object]:
    train_qlike, train_rmse = metrics(y, prediction, train_mask)
    val_qlike, val_rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": int(fold),
        "condition": condition,
        "readout_mode": readout_mode,
        "alpha": np.nan if alpha is None else float(alpha),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "train_qlike": float(train_qlike),
        "train_rmse": float(train_rmse),
        "val_qlike": float(val_qlike),
        "val_rmse": float(val_rmse),
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
        **signal_diagnostics(val_y, val_prediction),
    }


def _group_metric_rows(
    *,
    fold: int,
    condition: str,
    alpha: float | None,
    readout_mode: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    val_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    validation = frame.loc[val_mask].reset_index(drop=True)
    y_val = y[val_mask]
    prediction_val = prediction[val_mask]
    for lead in sorted(validation["lead"].dropna().unique()):
        for label in sorted(validation["label"].dropna().unique()):
            mask = (
                validation["lead"].eq(lead).to_numpy()
                & validation["label"].eq(label).to_numpy()
            )
            if not mask.any():
                continue
            local_mask = np.ones(int(mask.sum()), dtype=bool)
            observed = y_val[mask]
            forecast = prediction_val[mask]
            qlike, rmse = metrics(observed, forecast, local_mask)
            rows.append(
                {
                    "fold": int(fold),
                    "condition": condition,
                    "readout_mode": readout_mode,
                    "alpha": np.nan if alpha is None else float(alpha),
                    "lead": int(lead),
                    "label": int(label),
                    "samples": int(mask.sum()),
                    "qlike": float(qlike),
                    "rmse": float(rmse),
                    **mincer_zarnowitz(observed, forecast),
                    **horizon_mz_summary(observed, forecast),
                    **signal_diagnostics(observed, forecast),
                }
            )
    return rows


def _feature_diagnostic_row(
    *,
    fold: int,
    condition: str,
    features: np.ndarray,
    train_mask: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    train = np.asarray(features[train_mask], dtype=float)
    std = train.std(axis=0)
    centered = train - train.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    variance_fraction = (
        variance / variance.sum()
        if variance.sum() > 0.0
        else np.zeros_like(variance)
    )
    cumulative = np.cumsum(variance_fraction)

    def dimensions_for_fraction(fraction: float) -> int:
        if cumulative.size == 0 or cumulative[-1] <= 0.0:
            return 0
        return int(np.searchsorted(cumulative, fraction) + 1)

    return {
        "fold": int(fold),
        "condition": condition,
        "samples": int(len(features)),
        "features": int(features.shape[1]),
        "effective_rank_train": effective_rank(train),
        "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
        "dimensions_90pct_variance": dimensions_for_fraction(0.90),
        "dimensions_95pct_variance": dimensions_for_fraction(0.95),
        "dimensions_99pct_variance": dimensions_for_fraction(0.99),
        "near_constant_features": int((std < 1e-8).sum()),
        "mean_feature_std": float(std.mean()),
        "max_feature_std": float(std.max(initial=0.0)),
        "probe_steps": json.dumps(metadata["probe_steps"]),
        "total_evolution_time_us": float(
            metadata["total_evolution_time_us"]
        ),
    }


def run_temporal_rydberg_chain_analysis_experiment(
    *,
    fold_dir: Path,
    results_root: Path,
    experiment: TemporalRydbergExperimentConfig,
    reservoir: TemporalRydbergChainConfig,
    contaminated_samples: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Run the development assay and persist readout-ready diagnostics."""
    experiment.validate()
    reservoir.validate()
    dataset = load_rolling_fold_dataset(fold_dir)
    excluded_ids = _load_excluded_ids(contaminated_samples)
    level_channel = resolve_level_channel(
        dataset,
        name=experiment.level_channel_name,
        fallback=experiment.fallback_level_channel,
    )

    parameters = {
        "fold_dir": fold_dir,
        "contaminated_samples": contaminated_samples,
        "experiment": experiment.to_dict(),
        "reservoir": reservoir.to_dict(),
    }
    run_dir = begin_run(
        results_root,
        parameters,
        run_id=run_id,
    )

    metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    retained_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_blocks: list[dict[str, object]] = []
    window_records: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []
    canonical_feature_names: tuple[str, ...] | None = None

    for fold in experiment.folds:
        selected = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=experiment.leads,
            max_per_class=experiment.max_per_class,
            excluded_ids=excluded_ids,
            seed=experiment.seed,
        )
        tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=experiment.sequence_length,
            level_channel=level_channel,
        )
        usable = (
            dataset.valid[tensor_rows]
            & np.isfinite(level).all(axis=1)
        )
        frame = selected.loc[usable].reset_index(drop=True)
        level = level[usable]
        tensor_rows = tensor_rows[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no valid rows remain")
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        if not train_mask.any() or not val_mask.any():
            raise RuntimeError(
                f"fold {fold}: no train or validation rows remain"
            )

        scaler = fit_level_rate_scaler(level, train_mask)
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        metric_rows.append(
            _metric_row(
                fold=int(fold),
                condition="har_only",
                alpha=None,
                readout_mode="baseline",
                y=y,
                prediction=har,
                train_mask=train_mask,
                val_mask=val_mask,
            )
        )
        group_rows.extend(
            _group_metric_rows(
                fold=int(fold),
                condition="har_only",
                alpha=None,
                readout_mode="baseline",
                frame=frame,
                y=y,
                prediction=har,
                val_mask=val_mask,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                frame,
                y,
                har,
                fold=int(fold),
                condition="har_only",
                alpha=None,
                readout_mode="baseline",
            )
        )
        window_records.extend(
            {
                "fold": int(fold),
                "sample_id": str(row.sample_id),
                "fold_split": str(row.fold_split),
                "lead": int(row.lead),
                "label": int(row.label),
                "episode_id": str(row.episode_id),
                "origin_date": str(row.origin_date),
                "level_window": level[index].copy(),
            }
            for index, row in enumerate(frame.itertuples(index=False))
        )

        condition_features: dict[str, np.ndarray] = {}
        condition_metadata: dict[str, dict[str, object]] = {}
        for condition in experiment.conditions:
            controlled = controlled_level_windows(
                level,
                condition,
                seed=_condition_seed(
                    experiment.seed,
                    int(fold),
                    condition,
                ),
            )
            windows = transform_level_windows(controlled, scaler)
            features, metadata = build_temporal_rydberg_chain_features(
                windows,
                reservoir,
                condition=condition,
            )
            names = feature_names_from_metadata(metadata)
            if canonical_feature_names is None:
                canonical_feature_names = names
            elif names != canonical_feature_names:
                raise RuntimeError(
                    "reservoir feature ordering changed across folds/conditions"
                )

            condition_features[condition] = features
            condition_metadata[condition] = metadata
            diagnostic_rows.append(
                _feature_diagnostic_row(
                    fold=int(fold),
                    condition=condition,
                    features=features,
                    train_mask=train_mask,
                    metadata=metadata,
                )
            )
            feature_blocks.append(
                {
                    "condition": condition,
                    "frame": frame.copy(),
                    "features": features.copy(),
                    "targets": y.copy(),
                    "har_predictions": har.copy(),
                    "source_level": level.copy(),
                    "encoded_level": controlled.copy(),
                }
            )

            feature_scaler = StandardScaler()
            feature_train = feature_scaler.fit_transform(
                features[train_mask]
            )
            feature_all = feature_scaler.transform(features)
            target = y if experiment.readout_mode == "direct" else y - har
            for alpha in experiment.alphas:
                readout = Ridge(alpha=float(alpha))
                readout.fit(feature_train, target[train_mask])
                correction = readout.predict(feature_all)
                prediction = (
                    correction
                    if experiment.readout_mode == "direct"
                    else har + correction
                )
                metric_rows.append(
                    _metric_row(
                        fold=int(fold),
                        condition=condition,
                        alpha=float(alpha),
                        readout_mode=experiment.readout_mode,
                        y=y,
                        prediction=prediction,
                        train_mask=train_mask,
                        val_mask=val_mask,
                    )
                )
                group_rows.extend(
                    _group_metric_rows(
                        fold=int(fold),
                        condition=condition,
                        alpha=float(alpha),
                        readout_mode=experiment.readout_mode,
                        frame=frame,
                        y=y,
                        prediction=prediction,
                        val_mask=val_mask,
                    )
                )
                prediction_frames.append(
                    make_prediction_frame(
                        frame,
                        y,
                        prediction,
                        fold=int(fold),
                        condition=condition,
                        alpha=float(alpha),
                        readout_mode=experiment.readout_mode,
                    )
                )

        ordered = condition_features.get("ordered")
        if ordered is not None:
            for condition, features in condition_features.items():
                if condition == "ordered":
                    continue
                diagnostic_rows.append(
                    {
                        "fold": int(fold),
                        "condition": f"ordered_minus_{condition}",
                        "samples": int(len(features)),
                        "features": int(features.shape[1]),
                        "effective_rank_train": np.nan,
                        "numerical_rank_train": np.nan,
                        "dimensions_90pct_variance": np.nan,
                        "dimensions_95pct_variance": np.nan,
                        "dimensions_99pct_variance": np.nan,
                        "near_constant_features": np.nan,
                        "mean_feature_std": np.nan,
                        "max_feature_std": np.nan,
                        "probe_steps": json.dumps(
                            condition_metadata[condition]["probe_steps"]
                        ),
                        "total_evolution_time_us": float(
                            condition_metadata[condition][
                                "total_evolution_time_us"
                            ]
                        ),
                        "mean_absolute_feature_difference": float(
                            np.mean(np.abs(ordered - features))
                        ),
                    }
                )

        retained = frame[
            [
                "sample_id",
                "fold",
                "fold_split",
                "lead",
                "label",
                "episode_id",
                "origin_date",
            ]
        ].copy()
        retained["tensor_row"] = tensor_rows
        retained_rows.append(retained)
        fold_metadata.append(
            {
                "fold": int(fold),
                "fold_dir": str(fold_dir),
                "tensor_shape": list(dataset.values.shape),
                "channel_names": list(dataset.channel_names),
                "level_channel": level_channel,
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "scaler": scaler.to_dict(),
                "reservoir_metadata": condition_metadata.get("ordered"),
            }
        )

    if canonical_feature_names is None:
        raise RuntimeError("no reservoir features were generated")

    metrics_frame = pd.DataFrame(metric_rows)
    group_frame = pd.DataFrame(group_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    retained_frame = pd.concat(retained_rows, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    windows_frame = pd.DataFrame(window_records)
    best_alphas = best_alpha_per_condition(metrics_frame)

    metrics_frame.to_csv(run_dir / "fold_metrics.csv", index=False)
    group_frame.to_csv(run_dir / "group_metrics.csv", index=False)
    diagnostics_frame.to_csv(
        run_dir / "feature_diagnostics.csv", index=False
    )
    retained_frame.to_csv(run_dir / "retained_samples.csv", index=False)
    predictions_frame.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    best_alphas.to_csv(
        run_dir / "best_alpha_per_condition.csv", index=False
    )
    write_feature_archive(
        run_dir / "reservoir_features.npz",
        feature_blocks=feature_blocks,
        feature_names=canonical_feature_names,
    )

    candidates = metrics_frame.loc[
        ~metrics_frame["condition"].eq("har_only")
    ]
    best_per_fold = (
        candidates.sort_values(
            [
                "fold",
                "val_qlike",
                "val_rmse",
                "condition",
                "alpha",
            ]
        )
        .groupby("fold", as_index=False)
        .head(1)
    )
    best_per_fold.to_csv(run_dir / "best_per_fold.csv", index=False)

    plot_files = write_transition_overlays(
        run_dir,
        predictions=predictions_frame,
        window_records=windows_frame,
        best_alphas=best_alphas,
    )

    summary = {
        "schema_version": 2,
        "status": "development_assay",
        "test_rows_used": 0,
        "fold_dir": str(fold_dir),
        "experiment": experiment.to_dict(),
        "reservoir": reservoir.to_dict(),
        "folds": fold_metadata,
        "files": {
            "fold_metrics": "fold_metrics.csv",
            "group_metrics": "group_metrics.csv",
            "feature_diagnostics": "feature_diagnostics.csv",
            "retained_samples": "retained_samples.csv",
            "predictions": "predictions.csv.gz",
            "reservoir_features": "reservoir_features.npz",
            "best_alpha_per_condition": "best_alpha_per_condition.csv",
            "best_per_fold": "best_per_fold.csv",
            "forecast_overlays": plot_files,
        },
        "feature_archive": {
            "rows": int(
                sum(len(block["frame"]) for block in feature_blocks)
            ),
            "feature_count": int(len(canonical_feature_names)),
            "pickle_required": False,
            "contains": [
                "feature_matrix",
                "feature_names",
                "target_path",
                "har_prediction_path",
                "source_level_windows",
                "encoded_level_windows",
                "sample and condition metadata",
            ],
        },
        "known_limitations": [
            "Development folds only; no untouched test rows are evaluated.",
            "Exact-state simulation is intended for the initial small-atom "
            "architecture assay.",
            "Validation metrics are used to inspect the alpha grid and are "
            "not final unbiased estimates.",
            "Best-alpha overlay plots are development diagnostics and not "
            "untouched performance estimates.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
