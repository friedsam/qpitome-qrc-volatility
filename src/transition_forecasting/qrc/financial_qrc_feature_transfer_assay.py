from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    evolve_carrier_mask_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    CROSSOVER_SCHEDULES,
    build_crossover_feature_banks,
    evolve_crossover_probabilities,
)
from transition_forecasting.qrc.frozen_chain_readout_tools import chronological_inner_split
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
from transition_forecasting.qrc.residual_head_root_cause_assay import (
    balanced_sign_accuracy,
    qlike_loss,
    rmse_loss,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    effective_rank,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    _select_rows_for_fold,
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


ARCHITECTURES = (
    "raw_input_linear",
    "incumbent_six_modes",
    "incumbent_occupation_pair",
    "repaired_palindrome_occupation_pair",
    "repaired_palindrome_pc1",
)


@dataclass(frozen=True)
class FinancialQRCFeatureTransferConfig:
    """Clean L5 transfer test for repaired versus incumbent QRC features."""

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
    lead: int = 5
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0)
    selection_seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0
    split_horizon: int = 4
    incumbent_step_duration_us: float = 0.03
    repaired_step_duration_us: float = 0.02
    interaction_scale: float = 1.25
    palindrome_schedule: str = "crossover_Ahalf_B_Ahalf"

    def validate(self) -> None:
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be nonempty and unique")
        if any(int(fold) <= 3 for fold in self.folds):
            raise ValueError("financial transfer is restricted to development folds 4-8")
        if self.lead != 5:
            raise ValueError("the first transfer assay is intentionally restricted to L5")
        if self.max_per_class < 1 or self.sequence_length < 2:
            raise ValueError("sample cap and sequence length must be positive")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.ridge_alphas or any(alpha <= 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if not 1 <= self.split_horizon < len(TARGET_COLUMNS):
            raise ValueError("split_horizon lies outside the target path")
        if self.incumbent_step_duration_us <= 0 or self.repaired_step_duration_us <= 0:
            raise ValueError("step durations must be positive")
        if self.interaction_scale <= 0:
            raise ValueError("interaction_scale must be positive")
        if self.palindrome_schedule not in {schedule.name for schedule in CROSSOVER_SCHEDULES}:
            raise ValueError("unknown palindrome schedule")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _schedule(name: str):
    for schedule in CROSSOVER_SCHEDULES:
        if schedule.name == name:
            return schedule
    raise ValueError(f"unknown crossover schedule: {name}")


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fit_signed_residual_probe(
    features: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    origin_date: np.ndarray,
    config: FinancialQRCFeatureTransferConfig,
    pc1_only: bool,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    """Select alpha chronologically, then fit a no-intercept residual probe."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(y) or not np.isfinite(matrix).all():
        raise ValueError("features must be finite and aligned with targets")
    fit_mask = np.asarray(residual_train_mask, dtype=bool)
    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        fit_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )

    candidate_rows: list[dict[str, object]] = []
    for alpha in config.ridge_alphas:
        scaler = StandardScaler().fit(matrix[inner_fit])
        transformed = scaler.transform(matrix)
        explained = 1.0
        if pc1_only:
            pca = PCA(n_components=1, svd_solver="full").fit(transformed[inner_fit])
            transformed = pca.transform(transformed)
            explained = float(pca.explained_variance_ratio_.sum())
        model = Ridge(alpha=float(alpha), fit_intercept=False).fit(
            transformed[inner_fit], residuals[inner_fit]
        )
        correction = np.asarray(model.predict(transformed), dtype=float)
        prediction = har + correction
        candidate_rows.append(
            {
                "alpha": float(alpha),
                "inner_fit_rows": int(inner_fit.sum()),
                "inner_tune_rows": int(inner_tune.sum()),
                "inner_tune_rmse": rmse_loss(y[inner_tune], prediction[inner_tune]),
                "inner_tune_qlike": qlike_loss(y[inner_tune], prediction[inner_tune]),
                "pc1_explained_variance": explained,
                "mean_abs_tune_correction": float(np.mean(np.abs(correction[inner_tune]))),
            }
        )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["inner_tune_rmse", "inner_tune_qlike", "alpha"],
        ascending=[True, True, True],
    )
    selected = candidates.iloc[0]

    scaler = StandardScaler().fit(matrix[fit_mask])
    transformed = scaler.transform(matrix)
    explained = 1.0
    if pc1_only:
        pca = PCA(n_components=1, svd_solver="full").fit(transformed[fit_mask])
        transformed = pca.transform(transformed)
        explained = float(pca.explained_variance_ratio_.sum())
    model = Ridge(alpha=float(selected["alpha"]), fit_intercept=False).fit(
        transformed[fit_mask], residuals[fit_mask]
    )
    correction = np.asarray(model.predict(transformed), dtype=float)
    diagnostics = {
        "selected_alpha": float(selected["alpha"]),
        "fit_rows": int(fit_mask.sum()),
        "inner_fit_rows": int(inner_fit.sum()),
        "inner_tune_rows": int(inner_tune.sum()),
        "feature_width": int(matrix.shape[1]),
        "transformed_width": int(transformed.shape[1]),
        "pc1_only": bool(pc1_only),
        "pc1_explained_variance": explained,
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "intercept_max_abs": float(np.max(np.abs(np.atleast_1d(model.intercept_)))),
    }
    if diagnostics["intercept_max_abs"] > 0.0:
        raise RuntimeError("no-intercept residual probe acquired a nonzero intercept")
    return correction, diagnostics, candidates


def _cell_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    har: np.ndarray,
    correction: np.ndarray,
    validation: np.ndarray,
    *,
    fold: int,
    architecture: str,
    split_horizon: int,
) -> pd.DataFrame:
    selected = frame.loc[validation].reset_index(drop=True)
    horizons = y.shape[1]
    horizon = np.tile(np.arange(1, horizons + 1), len(selected))
    truth = y[validation].reshape(-1)
    baseline = har[validation].reshape(-1)
    quantum = correction[validation].reshape(-1)
    residual = truth - baseline
    return pd.DataFrame(
        {
            "fold": np.repeat(int(fold), len(truth)),
            "architecture": np.repeat(str(architecture), len(truth)),
            "sample_id": np.repeat(selected["sample_id"].astype(str), horizons),
            "episode_id": np.repeat(selected["episode_id"].astype(str), horizons),
            "origin_date": np.repeat(selected["origin_date"].astype(str), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "horizon": horizon,
            "segment": np.where(horizon <= int(split_horizon), "early", "late"),
            "y_true": truth,
            "har_prediction": baseline,
            "har_residual": residual,
            "qrc_correction": quantum,
            "augmented_prediction": baseline + quantum,
        }
    )


def _directional_payload(cells: pd.DataFrame) -> dict[str, float]:
    residual = cells["har_residual"].to_numpy(dtype=float)
    correction = cells["qrc_correction"].to_numpy(dtype=float)
    truth = cells["y_true"].to_numpy(dtype=float)
    har = cells["har_prediction"].to_numpy(dtype=float)
    augmented = cells["augmented_prediction"].to_numpy(dtype=float)
    positive = residual > 0.0
    negative = residual < 0.0
    mean_positive = float(np.mean(correction[positive])) if positive.any() else float("nan")
    mean_negative = float(np.mean(correction[negative])) if negative.any() else float("nan")
    return {
        "cells": int(len(cells)),
        "har_qlike": qlike_loss(truth, har),
        "augmented_qlike": qlike_loss(truth, augmented),
        "qlike_delta": qlike_loss(truth, augmented) - qlike_loss(truth, har),
        "har_rmse": rmse_loss(truth, har),
        "augmented_rmse": rmse_loss(truth, augmented),
        "rmse_delta": rmse_loss(truth, augmented) - rmse_loss(truth, har),
        "correction_residual_correlation": _safe_correlation(correction, residual),
        "balanced_sign_accuracy": balanced_sign_accuracy(residual, correction),
        "mean_correction": float(np.mean(correction)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "mean_correction_residual_positive": mean_positive,
        "mean_correction_residual_negative": mean_negative,
        "residual_sign_gap": mean_positive - mean_negative,
        "wrong_up_rate": float((correction[negative] > 0.0).mean()) if negative.any() else float("nan"),
        "wrong_down_rate": float((correction[positive] < 0.0).mean()) if positive.any() else float("nan"),
    }


def _fold_metric_rows(cells: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for population, population_mask in (
        ("all", np.ones(len(cells), dtype=bool)),
        ("control", cells["label"].eq(0).to_numpy()),
        ("transition", cells["label"].eq(1).to_numpy()),
    ):
        for segment, segment_mask in (
            ("all", np.ones(len(cells), dtype=bool)),
            ("early", cells["segment"].eq("early").to_numpy()),
            ("late", cells["segment"].eq("late").to_numpy()),
        ):
            mask = population_mask & segment_mask
            if not mask.any():
                continue
            rows.append(
                {
                    "fold": int(cells["fold"].iloc[0]),
                    "architecture": str(cells["architecture"].iloc[0]),
                    "population": population,
                    "segment": segment,
                    **_directional_payload(cells.loc[mask]),
                }
            )
    return rows


def _label_gap_rows(cells: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for segment, mask in (
        ("all", np.ones(len(cells), dtype=bool)),
        ("early", cells["segment"].eq("early").to_numpy()),
        ("late", cells["segment"].eq("late").to_numpy()),
    ):
        local = cells.loc[mask]
        control = local.loc[local["label"].eq(0), "qrc_correction"].to_numpy(dtype=float)
        transition = local.loc[local["label"].eq(1), "qrc_correction"].to_numpy(dtype=float)
        if not len(control) or not len(transition):
            continue
        rows.append(
            {
                "fold": int(cells["fold"].iloc[0]),
                "architecture": str(cells["architecture"].iloc[0]),
                "segment": segment,
                "control_mean_correction": float(np.mean(control)),
                "transition_mean_correction": float(np.mean(transition)),
                "transition_minus_control_correction": float(np.mean(transition) - np.mean(control)),
            }
        )
    return rows


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in fold_metrics.groupby(
        ["architecture", "population", "segment"], sort=True
    ):
        architecture, population, segment = keys
        rows.append(
            {
                "architecture": architecture,
                "population": population,
                "segment": segment,
                "folds": int(group["fold"].nunique()),
                "qlike_fold_wins": int(group["qlike_delta"].lt(0.0).sum()),
                "rmse_fold_wins": int(group["rmse_delta"].lt(0.0).sum()),
                "positive_correlation_folds": int(
                    group["correction_residual_correlation"].gt(0.0).sum()
                ),
                "positive_residual_sign_gap_folds": int(group["residual_sign_gap"].gt(0.0).sum()),
                "mean_fold_qlike_delta": float(group["qlike_delta"].mean()),
                "mean_fold_rmse_delta": float(group["rmse_delta"].mean()),
                "mean_fold_correlation": float(group["correction_residual_correlation"].mean()),
                "mean_fold_balanced_sign_accuracy": float(group["balanced_sign_accuracy"].mean()),
                "mean_fold_residual_sign_gap": float(group["residual_sign_gap"].mean()),
                "mean_fold_wrong_up_rate": float(group["wrong_up_rate"].mean()),
                "mean_fold_wrong_down_rate": float(group["wrong_down_rate"].mean()),
                "mean_fold_correction": float(group["mean_correction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_financial_qrc_feature_transfer_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    config: FinancialQRCFeatureTransferConfig = FinancialQRCFeatureTransferConfig(),
    candidate_features: CandidateFeatureConfig = CandidateFeatureConfig(),
    reservoir: TemporalRydbergChainConfig | None = None,
    geometry: StaggeredLadderGeometryConfig | None = None,
    run_id: str | None = None,
) -> Path:
    """Test whether repaired QRC features provide signed L5 financial information."""

    config.validate()
    candidate_features.validate()
    fold_dir = Path(fold_dir)
    dataset = load_rolling_fold_dataset(fold_dir)
    level_channel = resolve_level_channel(
        dataset,
        name=config.level_channel_name,
        fallback=config.fallback_level_channel,
    )
    base_reservoir = reservoir or TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=config.incumbent_step_duration_us,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=config.selection_seed,
    )
    base_reservoir.validate()
    geometry_config = geometry or StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    geometry_config.validate()
    if base_reservoir.n_atoms != 6 or base_reservoir.shots is not None:
        raise ValueError("financial transfer assay requires exact six-atom simulation")

    run_dir = begin_run(
        Path(results_root),
        {
            "fold_dir": str(fold_dir),
            "config": config.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": base_reservoir.to_dict(),
            "geometry": geometry_config.to_dict(),
            "architectures": list(ARCHITECTURES),
            "financial_inputs": "incumbent level plus local instability",
            "financial_test_rows_used": 0,
            "fit_intercept": False,
            "lambda_calibration": False,
            "qlike_selection": False,
            "scientific_question": (
                "Do repaired QRC features provide sample-specific, correctly signed L5 HAR "
                "residual corrections beyond decompression and one dominant feature direction?"
            ),
        },
        run_id=run_id,
    )
    probability_dir = run_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=False)
    selection_dir = run_dir / "selection_candidates"

    cell_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, object]] = []
    label_gap_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    scaler_rows: list[dict[str, object]] = []
    retained_rows: list[pd.DataFrame] = []

    for fold in config.folds:
        selected = _select_rows_for_fold(
            dataset.manifest,
            fold=int(fold),
            leads=(int(config.lead),),
            max_per_class=int(config.max_per_class),
            excluded_ids=set(),
            seed=int(config.selection_seed),
        )
        tensor_rows = selected["_tensor_row"].to_numpy(dtype=int)
        level = extract_level_windows(
            dataset,
            tensor_rows,
            sequence_length=config.sequence_length,
            level_channel=level_channel,
        )
        usable = dataset.valid[tensor_rows] & np.isfinite(level).all(axis=1)
        frame = selected.loc[usable].reset_index(drop=True)
        level = level[usable]
        tensor_rows = tensor_rows[usable]
        if frame.empty or frame["fold_split"].eq("test").any():
            raise RuntimeError(f"fold {fold}: invalid development sample selection")
        train = frame["fold_split"].eq("train").to_numpy()
        validation = frame["fold_split"].eq("val").to_numpy()
        if not train.any() or not validation.any():
            raise RuntimeError(f"fold {fold}: empty train or validation split")

        raw_sequences = build_candidate_sequences(
            level,
            "level_instability",
            candidate_features,
        )
        channel_scaler = fit_channel_scaler(raw_sequences, train)
        encoded = transform_candidate_sequences(raw_sequences, channel_scaler)
        scaler_rows.append(
            {
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "channel_0_median": float(channel_scaler.medians[0]),
                "channel_1_median": float(channel_scaler.medians[1]),
                "channel_0_half_range": float(channel_scaler.half_ranges[0]),
                "channel_1_half_range": float(channel_scaler.half_ranges[1]),
            }
        )

        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = _fit_har(frame, y, train)
        residuals, residual_train_mask = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=config.prequential_blocks,
        )
        if residual_train_mask.sum() < 15:
            raise RuntimeError(
                f"fold {fold}: only {int(residual_train_mask.sum())} causal residual rows"
            )

        incumbent_reservoir = replace(
            base_reservoir,
            step_duration_us=config.incumbent_step_duration_us,
            shot_seed=config.selection_seed + int(fold),
        )
        incumbent_probabilities, incumbent_metadata = evolve_carrier_mask_probabilities(
            encoded,
            incumbent_reservoir,
            geometry_config,
            np.asarray(CARRIER_MASKS["identity"], dtype=float),
            mask_name="identity_incumbent",
            interaction_scale=config.interaction_scale,
            drive_phase_rad=0.0,
        )
        incumbent_banks = build_crossover_feature_banks(incumbent_probabilities)

        repaired_reservoir = replace(
            base_reservoir,
            step_duration_us=config.repaired_step_duration_us,
            shot_seed=config.selection_seed + 10_000 + int(fold),
        )
        repaired_probabilities, repaired_metadata = evolve_crossover_probabilities(
            encoded,
            repaired_reservoir,
            geometry_config,
            _schedule(config.palindrome_schedule),
            interaction_scale=config.interaction_scale,
            drive_phase_rad=0.0,
        )
        repaired_banks = build_crossover_feature_banks(repaired_probabilities)

        np.savez_compressed(
            probability_dir / f"fold_{fold}_financial_transfer.npz",
            encoded_sequences=encoded,
            incumbent_probabilities=incumbent_probabilities,
            repaired_probabilities=repaired_probabilities,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            fold_split=frame["fold_split"].astype(str).to_numpy(),
            label=frame["label"].to_numpy(dtype=int),
            target_path=y,
            har_prediction_path=har,
            prequential_residual_path=residuals,
            prequential_residual_valid=residual_train_mask,
        )

        representations = {
            "raw_input_linear": (encoded.reshape(len(encoded), -1), False),
            "incumbent_six_modes": (
                incumbent_banks["six_mode_density_curvature"],
                False,
            ),
            "incumbent_occupation_pair": (
                incumbent_banks["occupation_pair_raw"],
                False,
            ),
            "repaired_palindrome_occupation_pair": (
                repaired_banks["occupation_pair_raw"],
                False,
            ),
            "repaired_palindrome_pc1": (
                repaired_banks["occupation_pair_raw"],
                True,
            ),
        }

        for architecture, (matrix, pc1_only) in representations.items():
            correction, diagnostics, candidates = _fit_signed_residual_probe(
                matrix,
                y=y,
                har=har,
                residuals=residuals,
                residual_train_mask=residual_train_mask,
                origin_date=frame["origin_date"].astype(str).to_numpy(),
                config=config,
                pc1_only=pc1_only,
            )
            candidates.insert(0, "fold", int(fold))
            candidates.insert(1, "architecture", architecture)
            candidates.to_csv(
                selection_dir / f"fold_{fold}__{architecture}.csv",
                index=False,
            )
            fit_rows.append(
                {
                    "fold": int(fold),
                    "architecture": architecture,
                    **diagnostics,
                }
            )
            train_matrix = np.asarray(matrix[train], dtype=float)
            feature_rows.append(
                {
                    "fold": int(fold),
                    "architecture": architecture,
                    "rows": int(len(matrix)),
                    "features": int(matrix.shape[1]),
                    "effective_rank_train": float(effective_rank(train_matrix)),
                    "numerical_rank_train": int(
                        np.linalg.matrix_rank(
                            train_matrix - train_matrix.mean(axis=0, keepdims=True)
                        )
                    ),
                    "pc1_only_readout": bool(pc1_only),
                }
            )
            cells = _cell_frame(
                frame,
                y,
                har,
                correction,
                validation,
                fold=int(fold),
                architecture=architecture,
                split_horizon=config.split_horizon,
            )
            cell_frames.append(cells)
            fold_metric_rows.extend(_fold_metric_rows(cells))
            label_gap_rows.extend(_label_gap_rows(cells))

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
        (run_dir / f"fold_{fold}_simulation_metadata.json").write_text(
            json.dumps(
                {
                    "incumbent": incumbent_metadata,
                    "repaired": repaired_metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    cells = pd.concat(cell_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    label_gaps = pd.DataFrame(label_gap_rows)
    architecture_summary = _aggregate_fold_metrics(fold_metrics)
    label_gap_summary = (
        label_gaps.groupby(["architecture", "segment"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            positive_gap_folds=(
                "transition_minus_control_correction",
                lambda values: int((values > 0.0).sum()),
            ),
            mean_fold_transition_minus_control=(
                "transition_minus_control_correction",
                "mean",
            ),
            minimum_fold_transition_minus_control=(
                "transition_minus_control_correction",
                "min",
            ),
        )
        .sort_values(["architecture", "segment"])
    )

    cells.to_csv(run_dir / "validation_cells.csv.gz", index=False, compression="gzip")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    architecture_summary.to_csv(run_dir / "architecture_summary.csv", index=False)
    label_gaps.to_csv(run_dir / "label_gap_by_fold.csv", index=False)
    label_gap_summary.to_csv(run_dir / "label_gap_summary.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(run_dir / "readout_fits.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(run_dir / "feature_diagnostics.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(run_dir / "channel_scalers.csv", index=False)
    pd.concat(retained_rows, ignore_index=True).to_csv(
        run_dir / "selected_samples.csv", index=False
    )

    primary = architecture_summary.loc[
        architecture_summary["population"].eq("all")
        & architecture_summary["segment"].eq("all")
    ].copy()
    primary = primary.sort_values(
        [
            "positive_residual_sign_gap_folds",
            "positive_correlation_folds",
            "rmse_fold_wins",
            "qlike_fold_wins",
            "mean_fold_residual_sign_gap",
        ],
        ascending=[False, False, False, False, False],
    )
    summary = {
        "status": "financial_qrc_feature_transfer_complete",
        "test_rows_used": 0,
        "folds": [int(value) for value in config.folds],
        "lead": int(config.lead),
        "architectures": list(ARCHITECTURES),
        "primary_directional_ranking": primary["architecture"].tolist(),
        "promotion_rule": (
            "A repaired QRC representation is promising only if its correction-residual "
            "correlation and residual-sign gap are positive in most folds, it improves over "
            "incumbent dynamics with the same 63-feature bank, and any metric gain is not "
            "purchased by a high wrong-up rate or control-only uplift."
        ),
        "files": {
            "validation_cells": "validation_cells.csv.gz",
            "fold_metrics": "fold_metrics.csv",
            "architecture_summary": "architecture_summary.csv",
            "label_gap_by_fold": "label_gap_by_fold.csv",
            "label_gap_summary": "label_gap_summary.csv",
            "readout_fits": "readout_fits.csv",
            "feature_diagnostics": "feature_diagnostics.csv",
            "channel_scalers": "channel_scalers.csv",
            "selected_samples": "selected_samples.csv",
        },
        "config": config.to_dict(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
