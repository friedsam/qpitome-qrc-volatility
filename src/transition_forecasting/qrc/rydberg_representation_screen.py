from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.numpy_esn import esn_states, make_esn_weights
from experiments.runs import begin_run
from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
)
from transition_forecasting.modeling.stage_e_classical_baselines import (
    HAR_FEATURES,
    TARGET_COLUMNS,
)
from transition_forecasting.modeling.stage_e_sequence_models import (
    DEFAULT_ESN_CONFIG,
    scale_sequences,
)
from transition_forecasting.qrc.representation_candidates import (
    REPRESENTATIONS,
    CandidateFeatureConfig,
    Representation,
    build_candidate_sequences,
    elementwise_quadratic_matrix,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    ReadoutMode,
    _best_mean_configuration,
    _evaluate_feature_matrix,
    _feature_diagnostic,
    _fit_har,
    _metric_payload,
    _prequential_har_residuals,
    _write_feature_archive,
    _write_summary_plot,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)


@dataclass(frozen=True)
class RepresentationScreenConfig:
    folds: tuple[int, ...] = (1, 2, 3)
    leads: tuple[int, ...] = (1, 5, 10)
    representations: tuple[Representation, ...] = REPRESENTATIONS
    max_per_class: int = 12
    alphas: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)
    components: tuple[int, ...] = (1, 2, 4, 8, 16, 0)
    readout_modes: tuple[ReadoutMode, ...] = (
        "direct",
        "prequential_har_residual",
    )
    esn_seeds: tuple[int, ...] = (1, 2, 3)
    prequential_blocks: int = 5
    seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or not self.leads or not self.representations:
            raise ValueError("folds, leads, and representations cannot be empty")
        if set(self.representations).difference(REPRESENTATIONS):
            raise ValueError("screen contains an unsupported representation")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must be positive")
        if not self.components or any(value < 0 for value in self.components):
            raise ValueError("components must be nonnegative; zero means full rank")
        if set(self.readout_modes).difference(
            {"direct", "prequential_har_residual"}
        ):
            raise ValueError("unsupported readout mode")
        if not self.esn_seeds:
            raise ValueError("esn_seeds cannot be empty")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least 2")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _select_rows_for_fold(
    manifest: pd.DataFrame,
    *,
    fold: int,
    leads: tuple[int, ...],
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    parts = [
        select_balanced_episode_rows(
            manifest,
            fold=fold,
            lead=int(lead),
            max_per_class=max_per_class,
            splits=DEVELOPMENT_FOLD_SPLITS,
            seed=seed + int(lead) * 1009,
        )
        for lead in leads
    ]
    selected = pd.concat(parts, ignore_index=True)
    if selected["_tensor_row"].duplicated().any():
        raise RuntimeError(f"fold {fold}: duplicate selected tensor rows")
    if selected["fold_split"].eq("test").any():
        raise RuntimeError("representation screen must not receive test rows")
    return selected.sort_values(
        [
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
            "sample_id",
        ]
    ).reset_index(drop=True)


def _archive_block(
    frame: pd.DataFrame,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    har: np.ndarray,
    input_sequences: np.ndarray,
    residuals: np.ndarray,
    residual_valid: np.ndarray,
    representation: str,
    seed: int,
) -> dict[str, object]:
    return {
        "features": features,
        "targets": targets,
        "har": har,
        "input_sequences": input_sequences,
        "prequential_residuals": residuals,
        "prequential_valid": residual_valid,
        "fold": frame["fold"].to_numpy(),
        "sample_id": frame["sample_id"].astype(str).to_numpy(),
        "fold_split": frame["fold_split"].astype(str).to_numpy(),
        "lead": frame["lead"].to_numpy(),
        "label": frame["label"].to_numpy(),
        "episode_id": frame["episode_id"].astype(str).to_numpy(),
        "origin_date": frame["origin_date"].astype(str).to_numpy(),
        "representation": representation,
        "seed": int(seed),
    }


def run_rydberg_representation_screen(
    *,
    fold_dir: Path,
    results_root: Path,
    screen: RepresentationScreenConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    run_id: str | None = None,
) -> Path:
    screen.validate()
    candidate_features.validate()
    reservoir.validate()
    dataset = load_rolling_fold_dataset(fold_dir)
    level_channel = resolve_level_channel(
        dataset,
        name=screen.level_channel_name,
        fallback=screen.fallback_level_channel,
    )
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": fold_dir,
            "screen": screen.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "esn_config": DEFAULT_ESN_CONFIG,
        },
        run_id=run_id,
    )

    metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    qrc_blocks: list[dict[str, object]] = []
    esn_blocks: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    qrc_feature_names: tuple[str, ...] | None = None
    esn_feature_names = tuple(
        [f"state_{index}" for index in range(int(DEFAULT_ESN_CONFIG["n"]))]
        + ["final_input_level", "final_input_second"]
    )

    for fold in screen.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=screen.leads,
            max_per_class=screen.max_per_class,
            seed=screen.seed,
        )
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=40,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        level = level[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no valid rows remain")
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train_mask)
        residuals, residual_train_mask = _prequential_har_residuals(
            frame,
            y,
            train_mask,
            blocks=screen.prequential_blocks,
        )

        baseline_payload = _metric_payload(y, har, val_mask)
        metric_rows.append(
            {
                "fold": int(fold),
                "representation": "baseline",
                "second_channel": "none",
                "model_family": "har",
                "seed": 0,
                "readout_mode": "direct",
                "components": len(HAR_FEATURES),
                "feature_width": len(HAR_FEATURES),
                "explained_variance": 1.0,
                "alpha": 100.0,
                "train_rows_used": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                **{
                    f"val_{key}": value
                    for key, value in baseline_payload.items()
                },
            }
        )
        validation = frame.loc[val_mask].reset_index(drop=True)
        y_val = y[val_mask]
        har_val = har[val_mask]
        for lead in sorted(validation["lead"].unique()):
            for label in sorted(validation["label"].unique()):
                local = (
                    validation["lead"].eq(lead).to_numpy()
                    & validation["label"].eq(label).to_numpy()
                )
                if not local.any():
                    continue
                local_mask = np.ones(int(local.sum()), dtype=bool)
                group_rows.append(
                    {
                        "fold": int(fold),
                        "representation": "baseline",
                        "model_family": "har",
                        "seed": 0,
                        "readout_mode": "direct",
                        "components": len(HAR_FEATURES),
                        "alpha": 100.0,
                        "lead": int(lead),
                        "label": int(label),
                        "samples": int(local.sum()),
                        **_metric_payload(
                            y_val[local],
                            har_val[local],
                            local_mask,
                        ),
                    }
                )

        for representation in screen.representations:
            raw_sequences = build_candidate_sequences(
                level,
                representation,
                candidate_features,
            )
            channel_scaler = fit_channel_scaler(
                raw_sequences,
                train_mask,
                q_low=candidate_features.q_low,
                q_high=candidate_features.q_high,
            )
            qrc_sequences = transform_candidate_sequences(
                raw_sequences,
                channel_scaler,
            )

            linear = raw_sequences.reshape(len(raw_sequences), -1)
            quadratic = elementwise_quadratic_matrix(raw_sequences)
            for family, features in (
                ("sequence_ridge", linear),
                ("quadratic_ridge", quadratic),
            ):
                rows, groups = _evaluate_feature_matrix(
                    feature_matrix=features,
                    frame=frame,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train_mask,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    fold=int(fold),
                    representation=representation,
                    model_family=family,
                    seed=0,
                    alphas=screen.alphas,
                    components=screen.components,
                    readout_modes=screen.readout_modes,
                )
                metric_rows.extend(rows)
                group_rows.extend(groups)
                diagnostic_rows.append(
                    _feature_diagnostic(
                        features=features,
                        train_mask=train_mask,
                        fold=int(fold),
                        representation=representation,
                        model_family=family,
                        seed=0,
                    )
                )

            scaled_esn_sequences = scale_sequences(raw_sequences, train_mask)
            for esn_seed in screen.esn_seeds:
                w_in, w = make_esn_weights(
                    n_inputs=2,
                    n_reservoir=int(DEFAULT_ESN_CONFIG["n"]),
                    spectral_radius=float(DEFAULT_ESN_CONFIG["sr"]),
                    input_scale=float(DEFAULT_ESN_CONFIG["inp"]),
                    seed=int(esn_seed),
                )
                states = esn_states(
                    scaled_esn_sequences,
                    w_in,
                    w,
                    float(DEFAULT_ESN_CONFIG["leak"]),
                )
                rows, groups = _evaluate_feature_matrix(
                    feature_matrix=states,
                    frame=frame,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_train_mask,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    fold=int(fold),
                    representation=representation,
                    model_family="esn",
                    seed=int(esn_seed),
                    alphas=screen.alphas,
                    components=screen.components,
                    readout_modes=screen.readout_modes,
                )
                metric_rows.extend(rows)
                group_rows.extend(groups)
                diagnostic_rows.append(
                    _feature_diagnostic(
                        features=states,
                        train_mask=train_mask,
                        fold=int(fold),
                        representation=representation,
                        model_family="esn",
                        seed=int(esn_seed),
                    )
                )
                esn_blocks.append(
                    _archive_block(
                        frame,
                        features=states,
                        targets=y,
                        har=har,
                        input_sequences=raw_sequences,
                        residuals=residuals,
                        residual_valid=residual_train_mask,
                        representation=representation,
                        seed=int(esn_seed),
                    )
                )

            qrc_features, qrc_metadata = build_temporal_rydberg_chain_features(
                qrc_sequences,
                reservoir,
                condition="ordered",
            )
            current_qrc_names = feature_names_from_metadata(qrc_metadata)
            if qrc_feature_names is None:
                qrc_feature_names = current_qrc_names
            elif qrc_feature_names != current_qrc_names:
                raise RuntimeError(
                    "QRC feature-name contract changed within the screen"
                )
            rows, groups = _evaluate_feature_matrix(
                feature_matrix=qrc_features,
                frame=frame,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                fold=int(fold),
                representation=representation,
                model_family="qrc",
                seed=0,
                alphas=screen.alphas,
                components=screen.components,
                readout_modes=screen.readout_modes,
            )
            metric_rows.extend(rows)
            group_rows.extend(groups)
            diagnostic_rows.append(
                {
                    **_feature_diagnostic(
                        features=qrc_features,
                        train_mask=train_mask,
                        fold=int(fold),
                        representation=representation,
                        model_family="qrc",
                        seed=0,
                    ),
                    "feature_names": json.dumps(current_qrc_names),
                }
            )
            qrc_blocks.append(
                _archive_block(
                    frame,
                    features=qrc_features,
                    targets=y,
                    har=har,
                    input_sequences=raw_sequences,
                    residuals=residuals,
                    residual_valid=residual_train_mask,
                    representation=representation,
                    seed=0,
                )
            )
            fold_records.append(
                {
                    "fold": int(fold),
                    "representation": representation,
                    "train_rows": int(train_mask.sum()),
                    "prequential_residual_rows": int(
                        residual_train_mask.sum()
                    ),
                    "val_rows": int(val_mask.sum()),
                    "channel_scaler": channel_scaler.to_dict(),
                    "qrc_feature_count": int(qrc_features.shape[1]),
                }
            )

    metrics_frame = pd.DataFrame(metric_rows)
    groups_frame = pd.DataFrame(group_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    metrics_frame.to_csv(run_dir / "model_metrics.csv", index=False)
    groups_frame.to_csv(run_dir / "group_metrics.csv", index=False)
    diagnostics_frame.to_csv(
        run_dir / "feature_diagnostics.csv",
        index=False,
    )

    candidates = metrics_frame.loc[
        ~metrics_frame["model_family"].eq("har")
    ].copy()
    best = _best_mean_configuration(candidates)
    best.to_csv(run_dir / "best_mean_configuration.csv", index=False)
    _write_summary_plot(best, run_dir / "representation_screen_qlike.png")
    if qrc_feature_names is None:
        raise RuntimeError("representation screen produced no QRC features")
    _write_feature_archive(
        run_dir / "qrc_features.npz",
        qrc_blocks,
        feature_names=qrc_feature_names,
    )
    _write_feature_archive(
        run_dir / "esn_features.npz",
        esn_blocks,
        feature_names=esn_feature_names,
    )

    summary = {
        "schema_version": 1,
        "status": "development_representation_screen",
        "test_rows_used": 0,
        "fold_dir": str(fold_dir),
        "screen": screen.to_dict(),
        "candidate_features": candidate_features.to_dict(),
        "reservoir": reservoir.to_dict(),
        "esn_config": DEFAULT_ESN_CONFIG,
        "folds": fold_records,
        "files": {
            "model_metrics": "model_metrics.csv",
            "group_metrics": "group_metrics.csv",
            "feature_diagnostics": "feature_diagnostics.csv",
            "best_mean_configuration": "best_mean_configuration.csv",
            "representation_plot": "representation_screen_qlike.png",
            "qrc_features": "qrc_features.npz",
            "esn_features": "esn_features.npz",
        },
        "known_limitations": [
            "Development folds only; no untouched test rows are evaluated.",
            "Hyperparameter grids are inspected on development validation folds.",
            "Prequential HAR residuals exclude the initial warm-up chronology.",
            "Elementwise quadratic ridge is a local degree-two mirror, not a full cross-time polynomial expansion.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
