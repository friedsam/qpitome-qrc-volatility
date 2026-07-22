from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    _fit_correction,
    choose_calibration,
    chronological_inner_split,
)
from transition_forecasting.qrc.instability_mechanism_tools import (
    SUPPORTED_CONTROLS,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
)
from transition_forecasting.qrc.temporal_rydberg_chain import effective_rank


@dataclass(frozen=True)
class ArchitectureCase:
    name: str
    architecture: str
    control: str
    interaction_scale: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LadderChallengerConfig:
    folds: tuple[int, ...] = (1, 2, 3)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    sequence_length: int = 40
    prequential_blocks: int = 5
    inner_holdout_fraction: float = 0.25
    global_lambdas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
    )
    pca_components: int = 4
    ridge_alpha: float = 100.0
    ladder_interaction_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    control_interaction_scale: float = 1.0
    controls: tuple[str, ...] = (
        "reset",
        "shuffled",
        "reversed",
        "block_shuffled",
    )
    block_size: int = 5
    seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or not self.leads:
            raise ValueError("folds and leads cannot be empty")
        if self.max_per_class < 1 or self.sequence_length < 2:
            raise ValueError("max_per_class and sequence_length must be positive")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.global_lambdas or any(x < 0 for x in self.global_lambdas):
            raise ValueError("global_lambdas must be nonnegative")
        if self.pca_components < 1 or self.ridge_alpha <= 0:
            raise ValueError("PCA components and ridge alpha must be positive")
        if not self.ladder_interaction_scales or any(
            x <= 0 for x in self.ladder_interaction_scales
        ):
            raise ValueError("ladder interaction scales must be positive")
        if self.control_interaction_scale <= 0:
            raise ValueError("control_interaction_scale must be positive")
        if set(self.controls).difference(SUPPORTED_CONTROLS):
            raise ValueError("unsupported ladder control")
        if "ordered" in self.controls:
            raise ValueError("ordered is already part of the ladder sweep")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _scale_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def architecture_cases(
    config: LadderChallengerConfig,
) -> tuple[ArchitectureCase, ...]:
    config.validate()
    cases = [
        ArchitectureCase(
            name="chain_interaction_off",
            architecture="chain",
            control="ordered",
            interaction_scale=0.0,
        ),
        ArchitectureCase(
            name="chain_interacting_1p00",
            architecture="chain",
            control="ordered",
            interaction_scale=1.0,
        ),
        ArchitectureCase(
            name="ladder_interaction_off",
            architecture="ladder",
            control="ordered",
            interaction_scale=0.0,
        ),
    ]
    cases.extend(
        ArchitectureCase(
            name=f"ladder_ordered_{_scale_token(scale)}",
            architecture="ladder",
            control="ordered",
            interaction_scale=float(scale),
        )
        for scale in config.ladder_interaction_scales
    )
    cases.extend(
        ArchitectureCase(
            name=(
                f"ladder_{control}_"
                f"{_scale_token(config.control_interaction_scale)}"
            ),
            architecture="ladder",
            control=str(control),
            interaction_scale=float(config.control_interaction_scale),
        )
        for control in config.controls
    )
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("architecture case names are not unique")
    return tuple(cases)


def _calibrated_readout(
    features: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    origin_date: np.ndarray,
    config: LadderChallengerConfig,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < config.pca_components:
        raise ValueError("feature matrix is too narrow for the frozen readout")
    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        residual_train_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )
    spec = {
        "model_kind": "pca",
        "feature_spec": f"prefix_{config.pca_components}",
        "indices": tuple(range(config.pca_components)),
        "components": config.pca_components,
        "alpha": config.ridge_alpha,
    }
    inner_correction = _fit_correction(
        spec,
        features=matrix,
        residuals=residuals,
        fit_mask=inner_fit,
    )
    _, lambda_values = choose_calibration(
        inner_correction,
        y=y,
        har=har,
        tune_mask=inner_tune,
        gate=np.ones(len(y), dtype=float),
        mode="global",
        lambdas=config.global_lambdas,
        policy="qlike",
        rmse_weight=0.0,
    )
    selected_lambda = float(lambda_values[0])
    full_correction = _fit_correction(
        spec,
        features=matrix,
        residuals=residuals,
        fit_mask=residual_train_mask,
    )
    uncalibrated = har + full_correction
    calibrated = har + selected_lambda * full_correction
    diagnostics = {
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
        "inner_har_qlike": float(_metric_payload(y, har, inner_tune)["qlike"]),
        "inner_calibrated_qlike": float(
            _metric_payload(
                y,
                har + selected_lambda * inner_correction,
                inner_tune,
            )["qlike"]
        ),
    }
    return calibrated, uncalibrated, selected_lambda, diagnostics


def _prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    case: str,
    architecture: str,
    control: str,
    interaction_scale: float,
    variant: str,
    selected_lambda: float,
) -> pd.DataFrame:
    horizons = y.shape[1]
    return pd.DataFrame(
        {
            "fold": np.repeat(frame["fold"].to_numpy(), horizons),
            "sample_id": np.repeat(
                frame["sample_id"].astype(str).to_numpy(), horizons
            ),
            "fold_split": np.repeat(
                frame["fold_split"].astype(str).to_numpy(), horizons
            ),
            "lead": np.repeat(frame["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(frame["label"].to_numpy(dtype=int), horizons),
            "episode_id": np.repeat(
                frame["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                frame["origin_date"].astype(str).to_numpy(), horizons
            ),
            "case": np.repeat(case, len(frame) * horizons),
            "architecture": np.repeat(
                architecture, len(frame) * horizons
            ),
            "control": np.repeat(control, len(frame) * horizons),
            "interaction_scale": np.repeat(
                interaction_scale, len(frame) * horizons
            ),
            "readout_variant": np.repeat(variant, len(frame) * horizons),
            "selected_lambda": np.repeat(
                selected_lambda, len(frame) * horizons
            ),
            "horizon": np.tile(np.arange(1, horizons + 1), len(frame)),
            "y_true": y.reshape(-1),
            "y_pred": prediction.reshape(-1),
        }
    )


def _group_metric_rows(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    fold: int,
    case: ArchitectureCase,
    variant: str,
    selected_lambda: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lead in sorted(frame["lead"].unique()):
        for label in sorted(frame["label"].unique()):
            local = (
                frame["lead"].eq(lead).to_numpy()
                & frame["label"].eq(label).to_numpy()
            )
            if not local.any():
                continue
            rows.append(
                {
                    "fold": int(fold),
                    "case": case.name,
                    "architecture": case.architecture,
                    "control": case.control,
                    "interaction_scale": case.interaction_scale,
                    "readout_variant": variant,
                    "selected_lambda": selected_lambda,
                    "lead": int(lead),
                    "label": int(label),
                    "samples": int(local.sum()),
                    **_metric_payload(
                        y[local],
                        prediction[local],
                        np.ones(int(local.sum()), dtype=bool),
                    ),
                }
            )
    return rows


def _dimension_count(matrix: np.ndarray, threshold: float) -> int:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    if variance.sum() <= 0:
        return 0
    cumulative = np.cumsum(variance / variance.sum())
    return int(np.searchsorted(cumulative, threshold) + 1)


def _feature_diagnostic_row(
    *,
    features: np.ndarray,
    occupation_features: np.ndarray,
    train_mask: np.ndarray,
    fold: int,
    case: ArchitectureCase,
) -> dict[str, object]:
    train_all = np.asarray(features[train_mask], dtype=float)
    train_occupation = np.asarray(
        occupation_features[train_mask], dtype=float
    )
    return {
        "fold": int(fold),
        "case": case.name,
        "architecture": case.architecture,
        "control": case.control,
        "interaction_scale": case.interaction_scale,
        "all_feature_width": int(features.shape[1]),
        "occupation_feature_width": int(occupation_features.shape[1]),
        "all_effective_rank": effective_rank(train_all),
        "occupation_effective_rank": effective_rank(train_occupation),
        "all_dimensions_95pct": _dimension_count(train_all, 0.95),
        "occupation_dimensions_95pct": _dimension_count(
            train_occupation, 0.95
        ),
        "occupation_near_constant": int(
            (train_occupation.std(axis=0) < 1e-8).sum()
        ),
    }


def _ladder_mode_rows(
    *,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    probe_steps: tuple[int, ...],
    residuals: np.ndarray,
    residual_mask: np.ndarray,
    fold: int,
    case: ArchitectureCase,
) -> list[dict[str, object]]:
    if case.architecture != "ladder":
        return []
    rows: list[dict[str, object]] = []
    for probe in probe_steps:
        indices = {
            site: feature_names.index(
                f"probe_{probe}_occupation_site_{site}"
            )
            for site in range(6)
        }
        occupation = np.column_stack(
            [features[:, indices[site]] for site in range(6)]
        )
        modes = {
            "top_density": occupation[:, :3].mean(axis=1),
            "bottom_density": occupation[:, 3:].mean(axis=1),
            "row_imbalance": (
                occupation[:, :3].mean(axis=1)
                - occupation[:, 3:].mean(axis=1)
            ),
            "top_gradient": occupation[:, 2] - occupation[:, 0],
            "bottom_gradient": occupation[:, 5] - occupation[:, 3],
        }
        for name, values in modes.items():
            correlations = []
            for horizon in range(residuals.shape[1]):
                left = values[residual_mask]
                right = residuals[residual_mask, horizon]
                if np.std(left) <= 0 or np.std(right) <= 0:
                    correlations.append(np.nan)
                else:
                    correlations.append(
                        float(np.corrcoef(left, right)[0, 1])
                    )
            rows.append(
                {
                    "fold": int(fold),
                    "case": case.name,
                    "control": case.control,
                    "interaction_scale": case.interaction_scale,
                    "probe_step": int(probe),
                    "mode": name,
                    "train_std": float(np.std(values[residual_mask])),
                    "mean_abs_residual_correlation": float(
                        np.nanmean(np.abs(correlations))
                    ),
                    "max_abs_residual_correlation": float(
                        np.nanmax(np.abs(correlations))
                    ),
                }
            )
    return rows


def _write_case_archive(
    path: Path,
    blocks: list[dict[str, object]],
    feature_names: tuple[str, ...],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "feature_matrix": np.concatenate(
            [np.asarray(block["features"], dtype=float) for block in blocks]
        ),
        "feature_names": np.asarray(feature_names, dtype=str),
        "source_level": np.concatenate(
            [np.asarray(block["source_level"], dtype=float) for block in blocks]
        ),
        "controlled_level": np.concatenate(
            [
                np.asarray(block["controlled_level"], dtype=float)
                for block in blocks
            ]
        ),
        "encoded_sequence": np.concatenate(
            [
                np.asarray(block["encoded_sequence"], dtype=float)
                for block in blocks
            ]
        ),
        "target_path": np.concatenate(
            [np.asarray(block["targets"], dtype=float) for block in blocks]
        ),
        "har_prediction_path": np.concatenate(
            [np.asarray(block["har"], dtype=float) for block in blocks]
        ),
        "prequential_residual_path": np.concatenate(
            [np.asarray(block["residuals"], dtype=float) for block in blocks]
        ),
        "prequential_residual_valid": np.concatenate(
            [
                np.asarray(block["residual_valid"], dtype=bool)
                for block in blocks
            ]
        ),
    }
    for key in (
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
    ):
        values = np.concatenate([np.asarray(block[key]) for block in blocks])
        arrays[key] = (
            values.astype(int)
            if key in {"fold", "lead", "label"}
            else values.astype(str)
        )
    for key in ("case", "architecture", "control", "interaction_scale"):
        arrays[key] = np.concatenate(
            [
                np.repeat(block[key], len(block["sample_id"]))
                for block in blocks
            ]
        )
    np.savez_compressed(path, **arrays)
