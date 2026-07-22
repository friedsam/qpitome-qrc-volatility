from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    TARGET_COLUMNS,
)
from transition_forecasting.qrc.instability_mechanism_tools import (
    reorder_level,
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
from transition_forecasting.qrc.rydberg_ladder_challenger_reporting import (
    finalize_ladder_challenger_run,
)
from transition_forecasting.qrc.rydberg_ladder_challenger_tools import (
    ArchitectureCase,
    LadderChallengerConfig,
    _calibrated_readout,
    _feature_diagnostic_row,
    _group_metric_rows,
    _ladder_mode_rows,
    _prediction_frame,
    architecture_cases,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
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
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    build_temporal_rydberg_ladder_features,
)


def run_rydberg_ladder_challenger(
    *,
    fold_dir: Path,
    results_root: Path,
    challenger: LadderChallengerConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    challenger.validate()
    candidate_features.validate()
    reservoir.validate()
    ladder_geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("the architecture comparison requires six atoms")

    cases = architecture_cases(challenger)
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "challenger": challenger.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "ladder_geometry": ladder_geometry.to_dict(),
            "cases": [case.to_dict() for case in cases],
        },
        run_id=run_id,
    )
    dataset = load_rolling_fold_dataset(fold_dir)
    level_channel = resolve_level_channel(
        dataset,
        name=challenger.level_channel_name,
        fallback=challenger.fallback_level_channel,
    )

    fold_metric_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    ladder_mode_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    path_blocks: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
    archive_blocks: dict[str, list[dict[str, object]]] = {
        case.name: [] for case in cases
    }
    archive_names: dict[str, tuple[str, ...]] = {}
    architecture_metadata: dict[str, dict[str, object]] = {}
    equivalence_rows: list[dict[str, object]] = []

    for fold in challenger.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=challenger.leads,
            max_per_class=challenger.max_per_class,
            seed=challenger.seed,
        )
        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=challenger.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        source = source[usable]
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no usable samples remain")

        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train_mask)
        residuals, residual_train_mask = _prequential_har_residuals(
            frame,
            y,
            train_mask,
            blocks=challenger.prequential_blocks,
        )
        noninteracting_occupations: dict[str, np.ndarray] = {}
        ordered_raw = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(ordered_raw, train_mask)

        har_payload = _metric_payload(y, har, val_mask)
        fold_metric_rows.append(
            {
                "fold": int(fold),
                "case": "har",
                "architecture": "classical",
                "control": "har",
                "interaction_scale": np.nan,
                "readout_variant": "baseline",
                "selected_lambda": 0.0,
                **{f"val_{key}": value for key, value in har_payload.items()},
            }
        )
        val_frame = frame.loc[val_mask].reset_index(drop=True)
        prediction_frames.append(
            _prediction_frame(
                val_frame,
                y[val_mask],
                har[val_mask],
                case="har",
                architecture="classical",
                control="har",
                interaction_scale=np.nan,
                variant="baseline",
                selected_lambda=0.0,
            )
        )
        path_blocks.setdefault(("har", "baseline"), []).append(
            {"y": y[val_mask], "prediction": har[val_mask]}
        )
        group_rows.extend(
            _group_metric_rows(
                val_frame,
                y[val_mask],
                har[val_mask],
                fold=int(fold),
                case=ArchitectureCase(
                    name="har",
                    architecture="classical",
                    control="har",
                    interaction_scale=np.nan,
                ),
                variant="baseline",
                selected_lambda=0.0,
            )
        )

        for case in cases:
            control_seed = (
                challenger.seed
                + int(fold) * 1_000_003
                + sum(
                    (index + 1) * ord(character)
                    for index, character in enumerate(case.control)
                )
            )
            controlled = reorder_level(
                source,
                case.control,
                seed=control_seed,
                block_size=challenger.block_size,
            )
            raw = build_candidate_sequences(
                controlled,
                "level_instability",
                candidate_features,
            )
            encoded = transform_candidate_sequences(raw, scaler)

            if case.architecture == "chain":
                current = replace(
                    reservoir,
                    c6_rad_um6_per_us=(
                        reservoir.c6_rad_um6_per_us
                        * max(case.interaction_scale, 1.0)
                    ),
                )
                condition = (
                    "interaction_off"
                    if case.interaction_scale == 0
                    else "ordered"
                )
                features, metadata = build_temporal_rydberg_chain_features(
                    encoded,
                    current,
                    condition=condition,
                )
                metadata = dict(metadata)
                metadata["geometry"] = "asymmetric_alternating_chain"
                metadata["interaction_scale"] = case.interaction_scale
            else:
                condition = (
                    "interaction_off"
                    if case.interaction_scale == 0
                    else (
                        "reset" if case.control == "reset" else "ordered"
                    )
                )
                features, metadata = build_temporal_rydberg_ladder_features(
                    encoded,
                    reservoir,
                    ladder_geometry,
                    interaction_scale=case.interaction_scale,
                    condition=condition,
                )

            names = feature_names_from_metadata(metadata)
            occupation_indices = np.asarray(
                [
                    index
                    for index, name in enumerate(names)
                    if "occupation_site_" in name
                ],
                dtype=int,
            )
            occupation = features[:, occupation_indices]
            if case.name in {
                "chain_interaction_off",
                "ladder_interaction_off",
            }:
                noninteracting_occupations[case.name] = occupation
            (
                calibrated,
                uncalibrated,
                selected_lambda,
                calibration_diagnostics,
            ) = _calibrated_readout(
                occupation,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train_mask,
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                config=challenger,
            )

            for variant, prediction, lambda_value in (
                ("calibrated", calibrated, selected_lambda),
                ("uncalibrated", uncalibrated, 1.0),
            ):
                payload = _metric_payload(y, prediction, val_mask)
                fold_metric_rows.append(
                    {
                        "fold": int(fold),
                        "case": case.name,
                        "architecture": case.architecture,
                        "control": case.control,
                        "interaction_scale": case.interaction_scale,
                        "readout_variant": variant,
                        "selected_lambda": lambda_value,
                        **calibration_diagnostics,
                        **{
                            f"val_{key}": value
                            for key, value in payload.items()
                        },
                    }
                )
                prediction_frames.append(
                    _prediction_frame(
                        val_frame,
                        y[val_mask],
                        prediction[val_mask],
                        case=case.name,
                        architecture=case.architecture,
                        control=case.control,
                        interaction_scale=case.interaction_scale,
                        variant=variant,
                        selected_lambda=lambda_value,
                    )
                )
                group_rows.extend(
                    _group_metric_rows(
                        val_frame,
                        y[val_mask],
                        prediction[val_mask],
                        fold=int(fold),
                        case=case,
                        variant=variant,
                        selected_lambda=lambda_value,
                    )
                )
                path_blocks.setdefault((case.name, variant), []).append(
                    {
                        "y": y[val_mask],
                        "prediction": prediction[val_mask],
                    }
                )

            diagnostic_rows.append(
                _feature_diagnostic_row(
                    features=features,
                    occupation_features=occupation,
                    train_mask=train_mask,
                    fold=int(fold),
                    case=case,
                )
            )
            ladder_mode_rows.extend(
                _ladder_mode_rows(
                    features=features,
                    feature_names=names,
                    probe_steps=tuple(int(x) for x in metadata["probe_steps"]),
                    residuals=residuals,
                    residual_mask=residual_train_mask,
                    fold=int(fold),
                    case=case,
                )
            )
            archive_blocks[case.name].append(
                {
                    "features": features,
                    "source_level": source,
                    "controlled_level": controlled,
                    "encoded_sequence": raw,
                    "targets": y,
                    "har": har,
                    "residuals": residuals,
                    "residual_valid": residual_train_mask,
                    "fold": frame["fold"],
                    "sample_id": frame["sample_id"],
                    "fold_split": frame["fold_split"],
                    "lead": frame["lead"],
                    "label": frame["label"],
                    "episode_id": frame["episode_id"],
                    "origin_date": frame["origin_date"],
                    "case": case.name,
                    "architecture": case.architecture,
                    "control": case.control,
                    "interaction_scale": case.interaction_scale,
                }
            )
            if case.name in archive_names and archive_names[case.name] != names:
                raise RuntimeError(
                    f"feature contract changed for case {case.name}"
                )
            archive_names[case.name] = names
            architecture_metadata.setdefault(case.name, metadata)

        difference = float(
            np.max(
                np.abs(
                    noninteracting_occupations[
                        "chain_interaction_off"
                    ]
                    - noninteracting_occupations[
                        "ladder_interaction_off"
                    ]
                )
            )
        )
        equivalence_rows.append(
            {
                "fold": int(fold),
                "maximum_absolute_occupation_difference": difference,
            }
        )
        if difference > 1e-10:
            raise RuntimeError(
                "noninteracting chain and ladder occupations differ: "
                f"fold={fold}, maximum_difference={difference}"
            )

    return finalize_ladder_challenger_run(
        run_dir=run_dir,
        cases=cases,
        challenger=challenger,
        ladder_geometry=ladder_geometry,
        fold_metric_rows=fold_metric_rows,
        group_rows=group_rows,
        diagnostic_rows=diagnostic_rows,
        ladder_mode_rows=ladder_mode_rows,
        prediction_frames=prediction_frames,
        equivalence_rows=equivalence_rows,
        path_blocks=path_blocks,
        archive_blocks=archive_blocks,
        archive_names=archive_names,
        architecture_metadata=architecture_metadata,
    )
