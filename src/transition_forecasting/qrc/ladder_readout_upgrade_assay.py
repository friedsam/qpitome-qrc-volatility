from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
)
from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    symmetric_ladder_mode_matrix,
)
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    MODEL_VARIANTS,
    LadderReadoutUpgradeConfig,
    apply_calibration,
    group_metric_table,
    horizon_mean_table,
    period_metric_table,
    pooled_metric_table,
    prediction_frame,
    select_inner_configuration,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _fit_har,
    _metric_payload,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    build_temporal_rydberg_ladder_features,
)


def _fold_metric_row(
    y: np.ndarray,
    prediction: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    model_name: str,
    diagnostics: dict[str, float],
) -> dict[str, object]:
    payload = _metric_payload(y, prediction, validation)
    return {
        "fold": int(fold),
        "model_name": model_name,
        "validation_rows": int(validation.sum()),
        **diagnostics,
        **{f"val_{key}": float(value) for key, value in payload.items()},
    }


def _l5_transition_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = predictions.loc[
        predictions["lead"].eq(5)
        & predictions["label"].eq(1)
        & predictions["fold"].isin([4, 5, 6, 7, 8])
    ].copy()
    if selected.empty:
        return
    actual = (
        selected.loc[selected["model_name"].eq("har")]
        .groupby("horizon")["y_true"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(actual.index, actual.values, marker="o", linewidth=2.5, label="Realized")
    order = (
        "har",
        "ladder_linear_global",
        "ladder_linear_segmented",
        "ladder_poly2_segmented",
    )
    for model_name in order:
        local = selected.loc[selected["model_name"].eq(model_name)]
        if local.empty:
            continue
        means = local.groupby("horizon")["y_pred"].mean()
        ax.plot(means.index, means.values, marker="o", label=model_name)
    ax.axvline(5, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Mean future log volatility")
    ax.set_title(
        "Segmented and polynomial readout challengers\n"
        "L5 transitions, later validation folds 4--8"
    )
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_ladder_readout_upgrade_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    config: LadderReadoutUpgradeConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    """Evaluate segmented calibration and degree-two ridge on the frozen ladder."""

    config.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the readout upgrade assay requires six atoms")
    if reservoir.shots is not None:
        raise ValueError("set reservoir.shots=None for the exact readout upgrade assay")

    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ladder_geometry": ladder_geometry.to_dict(),
            "frozen_quantum_core": {
                "geometry": "staggered_asymmetric_ladder_2x3",
                "interaction_scale": config.ladder_interaction_scale,
                "input": "level_plus_local_instability_5",
                "readout_modes": "nine_row_symmetric_modes",
            },
            "challengers": {
                "segmented_calibration": {
                    "early_horizons": [1, config.split_horizon],
                    "transition_horizons": [config.split_horizon + 1, 10],
                },
                "polynomial_ridge": {
                    "degree": config.polynomial_degree,
                    "alphas": list(config.polynomial_alphas),
                },
            },
            "test_rows_allowed": False,
            "common_test_block_evaluated": False,
            "post_confirmation_development_iteration": True,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(fold_dir)
    available_folds = set(dataset.manifest["fold"].astype(int).unique())
    missing = set(config.folds).difference(available_folds)
    if missing:
        raise ValueError(
            f"requested folds are absent: {sorted(missing)}; "
            f"available={sorted(available_folds)}"
        )
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )

    fold_rows: list[dict[str, object]] = []
    inner_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    feature_archive_blocks: list[dict[str, np.ndarray]] = []
    mode_names: tuple[str, ...] | None = None

    for fold in config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=config.leads,
            max_per_class=config.max_per_class,
            seed=config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("readout upgrade assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        source = source[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no usable rows remain")

        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if not train.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty train or validation split")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )

        fold_rows.append(
            _fold_metric_row(
                y,
                har,
                validation,
                fold=int(fold),
                model_name="har",
                diagnostics={
                    "readout_kind": "baseline",
                    "calibration_kind": "none",
                    "ridge_alpha": 0.0,
                    "early_lambda": 0.0,
                    "transition_lambda": 0.0,
                    "raw_feature_width": 0.0,
                    "design_feature_width": 0.0,
                },
            )
        )
        prediction_frames.append(
            prediction_frame(
                frame,
                y,
                har,
                har,
                mask=validation,
                model_name="har",
                readout_kind="baseline",
                calibration_kind="none",
                ridge_alpha=0.0,
                early_lambda=0.0,
                transition_lambda=0.0,
            )
        )

        raw_sequence = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequence, train)
        encoded = transform_candidate_sequences(raw_sequence, scaler)
        ladder_features, metadata = build_temporal_rydberg_ladder_features(
            encoded,
            reservoir,
            ladder_geometry,
            interaction_scale=config.ladder_interaction_scale,
            condition="ordered",
        )
        feature_names = feature_names_from_metadata(metadata)
        matrix, current_mode_names = symmetric_ladder_mode_matrix(
            ladder_features,
            feature_names,
            tuple(int(value) for value in metadata["probe_steps"]),
        )
        if mode_names is None:
            mode_names = current_mode_names
        elif mode_names != current_mode_names:
            raise RuntimeError("symmetric mode names changed across folds")
        centered = matrix[train] - matrix[train].mean(axis=0, keepdims=True)
        feature_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(matrix)),
                "train_rows": int(train.sum()),
                "raw_feature_width": int(matrix.shape[1]),
                "feature_names": "|".join(current_mode_names),
                "effective_rank_train": float(effective_rank(matrix[train])),
                "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
                "near_constant_features": int(
                    (matrix[train].std(axis=0) < 1e-8).sum()
                ),
            }
        )
        feature_archive_blocks.append(
            {
                "fold": np.full(len(frame), int(fold), dtype=int),
                "sample_id": frame["sample_id"].astype(str).to_numpy(),
                "fold_split": frame["fold_split"].astype(str).to_numpy(),
                "lead": frame["lead"].to_numpy(dtype=int),
                "label": frame["label"].to_numpy(dtype=int),
                "episode_id": frame["episode_id"].astype(str).to_numpy(),
                "origin_date": frame["origin_date"].astype(str).to_numpy(),
                "mode_matrix": matrix,
                "target_path": y,
                "har_prediction_path": har,
                "prequential_residual_path": residuals,
                "prequential_residual_valid": residual_train,
            }
        )

        for model_name, readout_kind, calibration_kind in MODEL_VARIANTS:
            diagnostics, candidates, correction, design_width = (
                select_inner_configuration(
                    matrix,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train,
                    origin_date=frame["origin_date"].astype(str).to_numpy(),
                    readout_kind=readout_kind,
                    calibration_kind=calibration_kind,
                    config=config,
                )
            )
            candidates.insert(0, "fold", int(fold))
            candidates.insert(1, "model_name", model_name)
            candidates.insert(2, "readout_kind", readout_kind)
            candidates.insert(3, "calibration_kind", calibration_kind)
            inner_frames.append(candidates)

            prediction = apply_calibration(
                har,
                correction,
                early_lambda=float(diagnostics["early_lambda"]),
                transition_lambda=float(diagnostics["transition_lambda"]),
                split_horizon=config.split_horizon,
            )
            fold_rows.append(
                _fold_metric_row(
                    y,
                    prediction,
                    validation,
                    fold=int(fold),
                    model_name=model_name,
                    diagnostics={
                        "readout_kind": readout_kind,
                        "calibration_kind": calibration_kind,
                        "ridge_alpha": float(diagnostics["ridge_alpha"]),
                        "early_lambda": float(diagnostics["early_lambda"]),
                        "transition_lambda": float(
                            diagnostics["transition_lambda"]
                        ),
                        "inner_fit_rows": float(diagnostics["inner_fit_rows"]),
                        "inner_tune_rows": float(diagnostics["inner_tune_rows"]),
                        "inner_qlike": float(diagnostics["inner_qlike"]),
                        "inner_rmse": float(diagnostics["inner_rmse"]),
                        "raw_feature_width": float(matrix.shape[1]),
                        "design_feature_width": float(design_width),
                    },
                )
            )
            prediction_frames.append(
                prediction_frame(
                    frame,
                    y,
                    prediction,
                    har,
                    mask=validation,
                    model_name=model_name,
                    readout_kind=readout_kind,
                    calibration_kind=calibration_kind,
                    ridge_alpha=float(diagnostics["ridge_alpha"]),
                    early_lambda=float(diagnostics["early_lambda"]),
                    transition_lambda=float(diagnostics["transition_lambda"]),
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not predictions["evaluation_split"].eq("val").all():
        raise RuntimeError("readout upgrade predictions contain non-validation rows")
    fold_metrics = pd.DataFrame(fold_rows)
    inner_candidates = pd.concat(inner_frames, ignore_index=True)
    pooled_metrics = pooled_metric_table(predictions)
    period_metrics = period_metric_table(predictions)
    group_metrics = group_metric_table(predictions)
    horizon_means = horizon_mean_table(predictions)
    feature_diagnostics = pd.DataFrame(feature_rows)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    inner_candidates.to_csv(run_dir / "inner_candidates.csv", index=False)
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pooled_metrics.to_csv(run_dir / "pooled_metrics.csv", index=False)
    period_metrics.to_csv(run_dir / "period_metrics.csv", index=False)
    group_metrics.to_csv(run_dir / "group_metrics.csv", index=False)
    horizon_means.to_csv(run_dir / "horizon_means.csv", index=False)
    feature_diagnostics.to_csv(
        run_dir / "feature_diagnostics.csv", index=False
    )

    if mode_names is None:
        raise RuntimeError("no ladder mode features were produced")
    archive_dir = run_dir / "feature_archives"
    archive_dir.mkdir(parents=True, exist_ok=False)
    mode_names_array = np.asarray(mode_names, dtype=str)
    for block in feature_archive_blocks:
        fold = int(block["fold"][0])
        np.savez_compressed(
            archive_dir / f"ladder_symmetric_modes_fold_{fold}.npz",
            **block,
            mode_names=mode_names_array,
        )

    _l5_transition_plot(
        predictions,
        run_dir / "l5_transition_path_challengers.png",
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    plot = period_metrics.loc[period_metrics["period"].eq("folds_4_8")]
    ax.bar(plot["model_name"], plot["qlike"])
    ax.set_ylabel("QLIKE")
    ax.set_title("Readout challengers on later validation folds 4--8")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / "later_fold_qlike.png", dpi=200)
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "status": "post_confirmation_readout_upgrade_development_assay",
        "evaluation_split": "validation",
        "folds": list(config.folds),
        "test_rows_used": 0,
        "common_test_block_evaluated": False,
        "quantum_core_changed": False,
        "readout_upgrades": [
            "two_segment_horizon_calibration",
            "degree_two_polynomial_ridge",
        ],
        "selection_protocol": (
            "ridge alpha and calibration weights selected only on a chronological "
            "inner holdout inside each rolling fold"
        ),
        "interpretation_warning": (
            "folds 4--8 have been inspected in earlier work and are no longer an "
            "independent confirmation set for this new readout iteration"
        ),
        "output_models": ["har", *[value[0] for value in MODEL_VARIANTS]],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
