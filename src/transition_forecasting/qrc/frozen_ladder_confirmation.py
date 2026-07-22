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
    FrozenLadderConfirmationConfig,
    confirmation_gate,
    fit_frozen_calibrated_prediction,
    grouped_metric_rows,
    metric_row,
    occupation_matrix,
    prediction_frame,
    symmetric_ladder_mode_matrix,
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
    build_temporal_rydberg_chain_features,
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


def _pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, local in predictions.groupby("model_name"):
        payload = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": model_name,
                **{key: float(value) for key, value in payload.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["qlike", "rmse"])


def _pooled_group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, local in predictions.groupby(
        ["model_name", "lead", "label"], dropna=False
    ):
        payload = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                "model_name": str(keys[0]),
                "lead": int(keys[1]),
                "label": int(keys[2]),
                **{key: float(value) for key, value in payload.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lead", "label", "qlike", "rmse"]
    )


def run_frozen_ladder_confirmation(
    *,
    fold_dir: Path,
    results_root: Path,
    confirmation: FrozenLadderConfirmationConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    confirmation.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the frozen ladder specification requires six atoms")

    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "confirmation": confirmation.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ladder_geometry": ladder_geometry.to_dict(),
            "frozen_models": {
                "chain_interaction_off": {
                    "readout": "occupation_pca4",
                    "interaction_scale": 0.0,
                },
                "chain_interacting_1p00": {
                    "readout": "occupation_pca4",
                    "interaction_scale": 1.0,
                },
                "frozen_ladder_symmetric_1p25": {
                    "readout": "symmetric_modes",
                    "interaction_scale": confirmation.ladder_interaction_scale,
                },
            },
        },
        run_id=run_id,
    )
    dataset = load_rolling_fold_dataset(fold_dir)
    available_folds = set(dataset.manifest["fold"].astype(int).unique())
    missing_folds = set(confirmation.folds).difference(available_folds)
    if missing_folds:
        raise ValueError(
            f"requested confirmation folds are absent: {sorted(missing_folds)}; "
            f"available={sorted(available_folds)}"
        )
    level_channel = resolve_level_channel(
        dataset,
        name=confirmation.level_channel_name,
        fallback=confirmation.fallback_level_channel,
    )

    metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    for fold in confirmation.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=confirmation.leads,
            max_per_class=confirmation.max_per_class,
            seed=confirmation.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("confirmation assay must not receive test rows")
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=confirmation.sequence_length,
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
            blocks=confirmation.prequential_blocks,
        )

        metric_rows.append(
            metric_row(
                y,
                har,
                validation,
                fold=int(fold),
                model_name="har",
                readout_family="baseline",
                selected_lambda=0.0,
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
                selected_lambda=0.0,
                readout_family="baseline",
            )
        )
        group_rows.extend(
            grouped_metric_rows(
                frame,
                y,
                har,
                validation,
                fold=int(fold),
                model_name="har",
                readout_family="baseline",
                selected_lambda=0.0,
            )
        )

        raw_sequence = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequence, train)
        encoded = transform_candidate_sequences(raw_sequence, scaler)

        model_features: list[tuple[str, np.ndarray, tuple[str, ...], str]] = []
        chain_off, chain_off_metadata = build_temporal_rydberg_chain_features(
            encoded,
            reservoir,
            condition="interaction_off",
        )
        off_names = feature_names_from_metadata(chain_off_metadata)
        off_matrix, off_readout_names = occupation_matrix(
            chain_off,
            off_names,
            tuple(int(value) for value in chain_off_metadata["probe_steps"]),
        )
        model_features.append(
            (
                "chain_interaction_off",
                off_matrix,
                off_readout_names,
                "pca",
            )
        )

        chain_on, chain_on_metadata = build_temporal_rydberg_chain_features(
            encoded,
            reservoir,
            condition="ordered",
        )
        on_names = feature_names_from_metadata(chain_on_metadata)
        on_matrix, on_readout_names = occupation_matrix(
            chain_on,
            on_names,
            tuple(int(value) for value in chain_on_metadata["probe_steps"]),
        )
        model_features.append(
            (
                "chain_interacting_1p00",
                on_matrix,
                on_readout_names,
                "pca",
            )
        )

        ladder, ladder_metadata = build_temporal_rydberg_ladder_features(
            encoded,
            reservoir,
            ladder_geometry,
            interaction_scale=confirmation.ladder_interaction_scale,
            condition="ordered",
        )
        ladder_names = feature_names_from_metadata(ladder_metadata)
        ladder_matrix, ladder_readout_names = symmetric_ladder_mode_matrix(
            ladder,
            ladder_names,
            tuple(int(value) for value in ladder_metadata["probe_steps"]),
        )
        model_features.append(
            (
                "frozen_ladder_symmetric_1p25",
                ladder_matrix,
                ladder_readout_names,
                "direct",
            )
        )

        for model_name, matrix, readout_names, transform in model_features:
            prediction, selected_lambda, diagnostics = (
                fit_frozen_calibrated_prediction(
                    matrix,
                    transform=transform,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train,
                    origin_date=frame["origin_date"].astype(str).to_numpy(),
                    config=confirmation,
                )
            )
            readout_family = (
                "symmetric_modes"
                if model_name.startswith("frozen_ladder")
                else "occupation_pca4"
            )
            metric_rows.append(
                metric_row(
                    y,
                    prediction,
                    validation,
                    fold=int(fold),
                    model_name=model_name,
                    readout_family=readout_family,
                    selected_lambda=selected_lambda,
                    diagnostics=diagnostics,
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
                    selected_lambda=selected_lambda,
                    readout_family=readout_family,
                )
            )
            group_rows.extend(
                grouped_metric_rows(
                    frame,
                    y,
                    prediction,
                    validation,
                    fold=int(fold),
                    model_name=model_name,
                    readout_family=readout_family,
                    selected_lambda=selected_lambda,
                )
            )
            centered = matrix[train] - matrix[train].mean(axis=0, keepdims=True)
            diagnostic_rows.append(
                {
                    "fold": int(fold),
                    "model_name": model_name,
                    "readout_family": readout_family,
                    "feature_width": int(matrix.shape[1]),
                    "feature_names": "|".join(readout_names),
                    "effective_rank_train": effective_rank(matrix[train]),
                    "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
                    "near_constant_features": int(
                        (matrix[train].std(axis=0) < 1e-8).sum()
                    ),
                }
            )

    fold_metrics = pd.DataFrame(metric_rows)
    group_metrics = pd.DataFrame(group_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not predictions["evaluation_split"].eq("val").all():
        raise RuntimeError("confirmation predictions contain a non-validation row")
    pooled_metrics = _pooled_metrics(predictions)
    pooled_groups = _pooled_group_metrics(predictions)
    gate = confirmation_gate(pooled_metrics, fold_metrics)

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    group_metrics.to_csv(run_dir / "group_metrics.csv", index=False)
    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pooled_metrics.to_csv(run_dir / "pooled_metrics.csv", index=False)
    pooled_groups.to_csv(run_dir / "pooled_group_metrics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        run_dir / "feature_diagnostics.csv", index=False
    )
    (run_dir / "confirmation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    plot = pooled_metrics.sort_values("qlike")
    ax.bar(plot["model_name"], plot["qlike"])
    ax.set_ylabel("Pooled confirmation-fold QLIKE")
    ax.set_xlabel("Frozen model")
    ax.set_title("Frozen ladder confirmation on validation folds 4--8")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / "confirmation_qlike.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model_name, local in fold_metrics.groupby("model_name"):
        ordered = local.sort_values("fold")
        ax.plot(
            ordered["fold"],
            ordered["qlike"],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Confirmation fold")
    ax.set_ylabel("QLIKE")
    ax.set_xticks(sorted(fold_metrics["fold"].unique()))
    ax.set_title("Frozen model performance by confirmation fold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "confirmation_fold_qlike.png", dpi=180)
    plt.close(fig)

    l5 = pooled_groups.loc[
        pooled_groups["lead"].eq(5) & pooled_groups["label"].eq(1)
    ].sort_values("qlike")
    l5.to_csv(run_dir / "l5_transition_summary.csv", index=False)
    if not l5.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.bar(l5["model_name"], l5["qlike"])
        ax.set_ylabel("L5 transition QLIKE")
        ax.set_xlabel("Frozen model")
        ax.set_title("L5 transitions on confirmation folds 4--8")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(run_dir / "confirmation_l5_qlike.png", dpi=180)
        plt.close(fig)

    summary = {
        "schema_version": 1,
        "status": "frozen_ladder_confirmation",
        "evaluation_split": "validation",
        "confirmation_folds": list(confirmation.folds),
        "development_folds_excluded": [1, 2, 3],
        "test_rows_used": 0,
        "common_test_block_evaluated": False,
        "frozen_primary_specification": {
            "geometry": "six_atom_staggered_asymmetric_ladder",
            "interaction_scale": confirmation.ladder_interaction_scale,
            "representation": "level_plus_local_instability_5",
            "readout_family": "symmetric_modes",
            "ridge_alpha": confirmation.ridge_alpha,
            "calibration": "one_global_lambda_on_chronological_inner_holdout",
            "lambda_grid": list(confirmation.global_lambdas),
        },
        "confirmation_gate": gate,
        "known_limitations": [
            "Folds 4--8 are additional non-test validation folds, not the final test block.",
            "The geometry and readout family were selected using development folds 1--3.",
            "The common final test block remains untouched after this run.",
            "The fold-local lambda is part of the frozen algorithm and is selected only inside each training partition.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir
