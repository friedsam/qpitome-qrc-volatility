from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_sequence_models import metrics
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    evolve_ladder_probe_probabilities,
    probabilities_to_symmetric_modes,
)
from transition_forecasting.qrc.ladder_readout_upgrade_tools import (
    LadderReadoutUpgradeConfig,
    apply_calibration,
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
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)


MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("ladder_linear_segmented", "linear"),
    ("ladder_poly2_segmented", "polynomial_degree2"),
)


@dataclass(frozen=True)
class PrecontrolSpacingAssayConfig:
    """Compact row-spacing assay on the frozen pre-control model and selector."""

    folds: tuple[int, ...] = tuple(range(1, 9))
    later_folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    row_spacings_um: tuple[float, ...] = (8.0, 9.0, 10.0)
    incumbent_row_spacing_um: float = 9.0
    incumbent_interaction_scale: float = 1.25
    minimum_overall_qlike_gain: float = 0.005
    maximum_overall_rmse_deterioration: float = 0.005
    maximum_l5_rmse_deterioration: float = 0.005
    minimum_later_fold_wins: int = 3
    seed: int = 20260722

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if not self.later_folds or set(self.later_folds).difference(self.folds):
            raise ValueError("later_folds must be a nonempty subset of folds")
        if not self.leads or any(int(value) < 1 for value in self.leads):
            raise ValueError("leads must be positive")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if len(self.row_spacings_um) < 2 or any(
            value <= 0 for value in self.row_spacings_um
        ):
            raise ValueError(
                "row_spacings_um must contain at least two positive values"
            )
        if len(set(float(value) for value in self.row_spacings_um)) != len(
            self.row_spacings_um
        ):
            raise ValueError("row_spacings_um must be unique")
        if not any(
            np.isclose(value, self.incumbent_row_spacing_um)
            for value in self.row_spacings_um
        ):
            raise ValueError("incumbent row spacing must be included in the assay")
        if self.incumbent_interaction_scale <= 0:
            raise ValueError("incumbent_interaction_scale must be positive")
        if min(
            self.minimum_overall_qlike_gain,
            self.maximum_overall_rmse_deterioration,
            self.maximum_l5_rmse_deterioration,
        ) < 0:
            raise ValueError("selection tolerances must be nonnegative")
        if not 1 <= self.minimum_later_fold_wins <= len(self.later_folds):
            raise ValueError("minimum_later_fold_wins is outside later_folds")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def geometry_name(row_spacing_um: float) -> str:
    return f"row_{float(row_spacing_um):.1f}um".replace(".", "p")


def geometry_variants(
    incumbent: StaggeredLadderGeometryConfig,
    config: PrecontrolSpacingAssayConfig,
) -> dict[str, StaggeredLadderGeometryConfig]:
    config.validate()
    variants = {
        geometry_name(value): replace(incumbent, row_spacing_um=float(value))
        for value in config.row_spacings_um
    }
    for geometry in variants.values():
        geometry.validate()
    return variants


def median_designated_nearest_coupling(
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float,
) -> float:
    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    values = np.asarray(
        [
            precomputed.interaction_matrix[left, right]
            for left, right in precomputed.nearest_pairs
        ],
        dtype=float,
    )
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("designated nearest-pair coupling set is invalid")
    return float(np.median(values))


def normalized_interaction_scale(
    reservoir: TemporalRydbergChainConfig,
    incumbent: StaggeredLadderGeometryConfig,
    challenger: StaggeredLadderGeometryConfig,
    *,
    incumbent_scale: float,
) -> float:
    target = median_designated_nearest_coupling(
        reservoir,
        incumbent,
        interaction_scale=float(incumbent_scale),
    )
    unscaled = median_designated_nearest_coupling(
        reservoir,
        challenger,
        interaction_scale=1.0,
    )
    return float(target / unscaled)


def _metric_payload(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["y_true"].to_numpy(dtype=float)[:, None]
    predicted = frame["y_pred"].to_numpy(dtype=float)[:, None]
    qlike, rmse = metrics(observed, predicted, np.ones(len(frame), dtype=bool))
    return {
        "qlike": float(qlike),
        "rmse": float(rmse),
        "mean_error": float(np.mean(predicted - observed)),
    }


def _metric_table(
    predictions: pd.DataFrame,
    *,
    later_folds: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (geometry, model_name), group in predictions.groupby(
        ["geometry", "model_name"], sort=True
    ):
        scopes = {
            "all_folds_overall": np.ones(len(group), dtype=bool),
            "later_folds_overall": group["fold"].isin(later_folds).to_numpy(),
            "later_folds_l5_transition": (
                group["fold"].isin(later_folds)
                & group["lead"].eq(5)
                & group["label"].eq(1)
            ).to_numpy(),
            "later_folds_controls": (
                group["fold"].isin(later_folds) & group["label"].eq(0)
            ).to_numpy(),
        }
        for scope, mask in scopes.items():
            local = group.loc[mask]
            if local.empty:
                continue
            rows.append(
                {
                    "geometry": str(geometry),
                    "model_name": str(model_name),
                    "scope": scope,
                    "prediction_rows": int(len(local)),
                    **_metric_payload(local),
                }
            )
    return pd.DataFrame(rows)


def _fold_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (geometry, model_name, fold), group in predictions.groupby(
        ["geometry", "model_name", "fold"], sort=True
    ):
        rows.append(
            {
                "geometry": str(geometry),
                "model_name": str(model_name),
                "fold": int(fold),
                "prediction_rows": int(len(group)),
                **_metric_payload(group),
            }
        )
    return pd.DataFrame(rows)


def _one_metric(
    metric_table: pd.DataFrame,
    *,
    geometry: str,
    model_name: str,
    scope: str,
) -> pd.Series:
    rows = metric_table.loc[
        metric_table["geometry"].eq(geometry)
        & metric_table["model_name"].eq(model_name)
        & metric_table["scope"].eq(scope)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one metric row for {geometry} {model_name} {scope}; "
            f"got {len(rows)}"
        )
    return rows.iloc[0]


def select_geometry(
    metric_table: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    config: PrecontrolSpacingAssayConfig,
) -> dict[str, object]:
    model_name = "ladder_linear_segmented"
    incumbent_name = geometry_name(config.incumbent_row_spacing_um)
    incumbent_overall = _one_metric(
        metric_table,
        geometry=incumbent_name,
        model_name=model_name,
        scope="later_folds_overall",
    )
    incumbent_l5 = _one_metric(
        metric_table,
        geometry=incumbent_name,
        model_name=model_name,
        scope="later_folds_l5_transition",
    )

    candidates: list[dict[str, object]] = []
    for spacing in config.row_spacings_um:
        name = geometry_name(spacing)
        overall = _one_metric(
            metric_table,
            geometry=name,
            model_name=model_name,
            scope="later_folds_overall",
        )
        l5 = _one_metric(
            metric_table,
            geometry=name,
            model_name=model_name,
            scope="later_folds_l5_transition",
        )
        wins = 0
        for fold in config.later_folds:
            challenger = fold_metrics.loc[
                fold_metrics["geometry"].eq(name)
                & fold_metrics["model_name"].eq(model_name)
                & fold_metrics["fold"].eq(int(fold))
            ]
            incumbent = fold_metrics.loc[
                fold_metrics["geometry"].eq(incumbent_name)
                & fold_metrics["model_name"].eq(model_name)
                & fold_metrics["fold"].eq(int(fold))
            ]
            if len(challenger) != 1 or len(incumbent) != 1:
                raise RuntimeError(
                    f"missing fold metric for geometry comparison in fold {fold}"
                )
            if float(challenger.iloc[0]["qlike"]) < float(
                incumbent.iloc[0]["qlike"]
            ):
                wins += 1

        row = {
            "geometry": name,
            "row_spacing_um": float(spacing),
            "is_incumbent": bool(name == incumbent_name),
            "later_overall_qlike": float(overall["qlike"]),
            "later_overall_rmse": float(overall["rmse"]),
            "later_l5_qlike": float(l5["qlike"]),
            "later_l5_rmse": float(l5["rmse"]),
            "overall_qlike_delta_vs_incumbent": float(
                overall["qlike"] - incumbent_overall["qlike"]
            ),
            "overall_rmse_delta_vs_incumbent": float(
                overall["rmse"] - incumbent_overall["rmse"]
            ),
            "l5_qlike_delta_vs_incumbent": float(
                l5["qlike"] - incumbent_l5["qlike"]
            ),
            "l5_rmse_delta_vs_incumbent": float(
                l5["rmse"] - incumbent_l5["rmse"]
            ),
            "later_fold_qlike_wins": int(wins),
        }
        row["eligible_challenger"] = bool(
            not row["is_incumbent"]
            and float(row["overall_qlike_delta_vs_incumbent"])
            <= -config.minimum_overall_qlike_gain
            and float(row["overall_rmse_delta_vs_incumbent"])
            <= config.maximum_overall_rmse_deterioration
            and float(row["l5_qlike_delta_vs_incumbent"]) <= 0.0
            and float(row["l5_rmse_delta_vs_incumbent"])
            <= config.maximum_l5_rmse_deterioration
            and int(row["later_fold_qlike_wins"])
            >= config.minimum_later_fold_wins
        )
        candidates.append(row)

    eligible = [row for row in candidates if bool(row["eligible_challenger"])]
    selected = (
        min(
            eligible,
            key=lambda row: (
                float(row["later_overall_qlike"]),
                float(row["later_l5_qlike"]),
                float(row["later_overall_rmse"]),
            ),
        )
        if eligible
        else next(row for row in candidates if bool(row["is_incumbent"]))
    )
    return {
        "primary_model": model_name,
        "selection_population": "original pre-control binary sample selector",
        "selection_period": "validation folds 4-8",
        "selected": selected,
        "selected_is_challenger": bool(selected["eligible_challenger"]),
        "candidates": candidates,
        "rule": (
            "Replace 9.0 um only when the linear segmented challenger gains at "
            f"least {config.minimum_overall_qlike_gain:.3f} later-fold overall "
            "QLIKE, does not materially worsen overall or L5 RMSE, does not "
            "worsen L5 QLIKE, and wins at least "
            f"{config.minimum_later_fold_wins} of {len(config.later_folds)} "
            "later folds."
        ),
    }


def run_precontrol_ladder_spacing_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    spacing_config: PrecontrolSpacingAssayConfig,
    readout_config: LadderReadoutUpgradeConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    incumbent_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    spacing_config.validate()
    readout_config.validate()
    candidate_features.validate()
    reservoir.validate()
    incumbent_geometry.validate()
    if tuple(spacing_config.folds) != tuple(readout_config.folds):
        raise ValueError("spacing and readout fold contracts differ")
    if tuple(spacing_config.leads) != tuple(readout_config.leads):
        raise ValueError("spacing and readout lead contracts differ")
    if spacing_config.max_per_class != readout_config.max_per_class:
        raise ValueError("spacing and readout sample caps differ")
    if spacing_config.seed != readout_config.seed:
        raise ValueError("spacing and readout selection seeds differ")
    if not np.isclose(
        incumbent_geometry.row_spacing_um,
        spacing_config.incumbent_row_spacing_um,
    ):
        raise ValueError("incumbent geometry does not match the spacing contract")
    if reservoir.n_atoms != 6 or reservoir.shots is not None:
        raise ValueError("assay requires an exact six-atom reservoir")

    variants = geometry_variants(incumbent_geometry, spacing_config)
    target_coupling = median_designated_nearest_coupling(
        reservoir,
        incumbent_geometry,
        interaction_scale=spacing_config.incumbent_interaction_scale,
    )
    interaction_scales = {
        name: normalized_interaction_scale(
            reservoir,
            incumbent_geometry,
            geometry,
            incumbent_scale=spacing_config.incumbent_interaction_scale,
        )
        for name, geometry in variants.items()
    }

    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "spacing_assay": spacing_config.to_dict(),
            "readout": readout_config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
            "incumbent_geometry": incumbent_geometry.to_dict(),
            "geometry_variants": {
                name: geometry.to_dict() for name, geometry in variants.items()
            },
            "interaction_scales": interaction_scales,
            "coupling_normalization": (
                "fixed median designated nearest-pair coupling"
            ),
            "control_manifest_status": "original pre-redesign manifest",
            "har_scope": "all selected train rows",
            "residual_task": "original prequential HAR residual path",
            "test_rows_allowed": False,
        },
        run_id=run_id,
    )

    dataset = load_rolling_fold_dataset(fold_dir)
    available_folds = set(dataset.manifest["fold"].astype(int).unique())
    missing = set(spacing_config.folds).difference(available_folds)
    if missing:
        raise ValueError(f"requested folds are absent: {sorted(missing)}")
    level_channel = resolve_level_channel(
        dataset,
        name=readout_config.level_channel_name,
        fallback=readout_config.fallback_level_channel,
    )

    prediction_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    geometry_rows: list[dict[str, object]] = []
    cache_root = run_dir / "probability_cache"

    for name, geometry in variants.items():
        scale = interaction_scales[name]
        geometry_rows.append(
            {
                "geometry": name,
                "row_spacing_um": float(geometry.row_spacing_um),
                "interaction_scale": float(scale),
                "target_median_nearest_coupling": float(target_coupling),
                "actual_median_nearest_coupling": (
                    median_designated_nearest_coupling(
                        reservoir,
                        geometry,
                        interaction_scale=scale,
                    )
                ),
                "positions_um": json.dumps(
                    precompute_ladder(
                        reservoir,
                        geometry,
                        interaction_scale=scale,
                    ).positions.tolist()
                ),
                "geometry_config": json.dumps(
                    geometry.to_dict(), sort_keys=True
                ),
            }
        )

    for fold in spacing_config.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=spacing_config.leads,
            max_per_class=spacing_config.max_per_class,
            seed=spacing_config.seed,
        )
        if frame["fold_split"].eq("test").any():
            raise RuntimeError("spacing assay must not receive test rows")

        tensor_rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=readout_config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(source).all(axis=1)
        frame = frame.loc[usable].reset_index(drop=True)
        source = source[usable]
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if frame.empty or not train.any() or not validation.any():
            raise RuntimeError(
                f"fold {fold}: empty usable train or validation panel"
            )

        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=readout_config.prequential_blocks,
        )
        raw_sequence = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(raw_sequence, train)
        encoded = transform_candidate_sequences(raw_sequence, scaler)

        for name, geometry in variants.items():
            scale = interaction_scales[name]
            probabilities, metadata = evolve_ladder_probe_probabilities(
                encoded,
                reservoir,
                geometry,
                interaction_scale=scale,
                condition="ordered",
            )
            matrix = probabilities_to_symmetric_modes(probabilities)
            cache_path = cache_root / name / f"fold_{int(fold)}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                probabilities=probabilities,
                exact_modes=matrix,
                sample_id=frame["sample_id"].astype(str).to_numpy(),
                fold_split=frame["fold_split"].astype(str).to_numpy(),
                lead=frame["lead"].to_numpy(dtype=int),
                label=frame["label"].to_numpy(dtype=int),
                episode_id=frame["episode_id"].astype(str).to_numpy(),
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                probe_steps=np.asarray(metadata["probe_steps"], dtype=int),
            )

            for model_name, readout_kind in MODEL_SPECS:
                diagnostics, candidates, correction, _ = (
                    select_inner_configuration(
                        matrix,
                        y=y,
                        har=har,
                        residuals=residuals,
                        residual_train_mask=residual_train,
                        origin_date=(
                            frame["origin_date"].astype(str).to_numpy()
                        ),
                        readout_kind=readout_kind,
                        calibration_kind="segmented",
                        config=readout_config,
                    )
                )
                prediction = apply_calibration(
                    har,
                    correction,
                    early_lambda=float(diagnostics["early_lambda"]),
                    transition_lambda=float(
                        diagnostics["transition_lambda"]
                    ),
                    split_horizon=readout_config.split_horizon,
                )
                predictions = prediction_frame(
                    frame,
                    y,
                    prediction,
                    har,
                    mask=validation,
                    model_name=model_name,
                    readout_kind=readout_kind,
                    calibration_kind="segmented",
                    ridge_alpha=float(diagnostics["ridge_alpha"]),
                    early_lambda=float(diagnostics["early_lambda"]),
                    transition_lambda=float(
                        diagnostics["transition_lambda"]
                    ),
                )
                predictions.insert(0, "geometry", name)
                predictions.insert(
                    1, "row_spacing_um", float(geometry.row_spacing_um)
                )
                predictions.insert(2, "interaction_scale", float(scale))
                prediction_frames.append(predictions)

                candidates = candidates.copy()
                candidates.insert(0, "fold", int(fold))
                candidates.insert(1, "geometry", name)
                candidates.insert(2, "model_name", model_name)
                candidates.insert(3, "readout_kind", readout_kind)
                candidate_frames.append(candidates)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not predictions["evaluation_split"].eq("val").all():
        raise RuntimeError(
            "spacing assay predictions contain non-validation rows"
        )
    inner_candidates = pd.concat(candidate_frames, ignore_index=True)
    metric_table = _metric_table(
        predictions,
        later_folds=spacing_config.later_folds,
    )
    fold_metrics = _fold_metric_table(predictions)
    selection = select_geometry(
        metric_table,
        fold_metrics,
        config=spacing_config,
    )
    geometry_summary = pd.DataFrame(geometry_rows)

    predictions.to_csv(
        run_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    inner_candidates.to_csv(
        run_dir / "inner_candidates.csv.gz",
        index=False,
        compression="gzip",
    )
    metric_table.to_csv(run_dir / "metrics_by_scope.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    geometry_summary.to_csv(run_dir / "geometry_summary.csv", index=False)
    (run_dir / "geometry_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "precontrol_compact_spacing_assay_complete",
        "folds": list(spacing_config.folds),
        "geometries": list(variants),
        "models": [name for name, _ in MODEL_SPECS],
        "test_rows_used": 0,
        "control_manifest_changed": False,
        "har_or_residual_task_changed": False,
        "readout_changed": False,
        "selection": selection,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
