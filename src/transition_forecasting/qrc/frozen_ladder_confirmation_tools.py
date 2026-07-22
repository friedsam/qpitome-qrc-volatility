from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    chronological_inner_split,
)
from transition_forecasting.qrc.ladder_mode_readout_tools import (
    _fit_correction,
    choose_global_lambda,
    ladder_mode_weights,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
)


@dataclass(frozen=True)
class FrozenLadderConfirmationConfig:
    """Locked development-confirmation protocol for folds 4--8.

    Folds 1--3 were used to select the ladder geometry/readout. This assay
    evaluates that single frozen specification on the remaining non-test
    validation folds. It never selects geometry, interaction scale, readout
    family, PCA width, or ridge alpha.
    """

    folds: tuple[int, ...] = (4, 5, 6, 7, 8)
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
    ladder_interaction_scale: float = 1.25
    seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds:
            raise ValueError("folds cannot be empty")
        if any(int(fold) <= 3 for fold in self.folds):
            raise ValueError(
                "confirmation folds must exclude development folds 1--3"
            )
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be unique")
        if not self.leads:
            raise ValueError("leads cannot be empty")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.global_lambdas or any(
            value < 0 for value in self.global_lambdas
        ):
            raise ValueError("global_lambdas must be nonnegative")
        if self.pca_components < 1:
            raise ValueError("pca_components must be positive")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.ladder_interaction_scale <= 0:
            raise ValueError("ladder_interaction_scale must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def occupation_matrix(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    probe_steps: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    values = np.asarray(features, dtype=float)
    for probe in probe_steps:
        indices = []
        for site in range(6):
            name = f"probe_{int(probe)}_occupation_site_{site}"
            try:
                indices.append(feature_names.index(name))
            except ValueError as exc:
                raise ValueError(f"missing occupation feature {name!r}") from exc
            names.append(name)
        blocks.append(values[:, indices])
    matrix = np.concatenate(blocks, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("occupation matrix contains non-finite values")
    return matrix, tuple(names)


def symmetric_ladder_mode_matrix(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    probe_steps: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the three frozen row-symmetric spatial modes per probe."""

    occupations, _ = occupation_matrix(features, feature_names, probe_steps)
    weights = ladder_mode_weights()[:, :3]
    blocks = []
    names = []
    for probe_index, probe in enumerate(probe_steps):
        start = 6 * probe_index
        transformed = occupations[:, start : start + 6] @ weights
        blocks.append(transformed)
        names.extend(
            (
                f"probe_{int(probe)}_symmetric_constant",
                f"probe_{int(probe)}_symmetric_gradient",
                f"probe_{int(probe)}_symmetric_curvature",
            )
        )
    return np.concatenate(blocks, axis=1), tuple(names)


def fit_frozen_calibrated_prediction(
    matrix: np.ndarray,
    *,
    transform: str,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    residual_train_mask: np.ndarray,
    origin_date: np.ndarray,
    config: FrozenLadderConfirmationConfig,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Fit the frozen residual head and select only its allowed global lambda."""

    inner_fit, inner_tune = chronological_inner_split(
        origin_date,
        residual_train_mask,
        holdout_fraction=config.inner_holdout_fraction,
    )
    inner_correction = _fit_correction(
        matrix,
        residuals,
        inner_fit,
        transform=transform,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    selected_lambda, inner_payload = choose_global_lambda(
        inner_correction,
        y=y,
        har=har,
        tune_mask=inner_tune,
        lambdas=config.global_lambdas,
    )
    full_correction = _fit_correction(
        matrix,
        residuals,
        residual_train_mask,
        transform=transform,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    prediction = har + float(selected_lambda) * full_correction
    diagnostics = {
        "selected_lambda": float(selected_lambda),
        "inner_fit_rows": float(inner_fit.sum()),
        "inner_tune_rows": float(inner_tune.sum()),
        **{key: float(value) for key, value in inner_payload.items()},
    }
    return prediction, float(selected_lambda), diagnostics


def prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    har: np.ndarray,
    *,
    mask: np.ndarray,
    model_name: str,
    selected_lambda: float,
    readout_family: str,
) -> pd.DataFrame:
    selected = frame.loc[mask].reset_index(drop=True)
    horizons = int(y.shape[1])
    return pd.DataFrame(
        {
            "fold": np.repeat(selected["fold"].to_numpy(dtype=int), horizons),
            "sample_id": np.repeat(
                selected["sample_id"].astype(str).to_numpy(), horizons
            ),
            "lead": np.repeat(selected["lead"].to_numpy(dtype=int), horizons),
            "label": np.repeat(selected["label"].to_numpy(dtype=int), horizons),
            "episode_id": np.repeat(
                selected["episode_id"].astype(str).to_numpy(), horizons
            ),
            "origin_date": np.repeat(
                selected["origin_date"].astype(str).to_numpy(), horizons
            ),
            "evaluation_split": np.repeat("val", int(mask.sum()) * horizons),
            "model_name": np.repeat(model_name, int(mask.sum()) * horizons),
            "readout_family": np.repeat(
                readout_family, int(mask.sum()) * horizons
            ),
            "selected_lambda": np.repeat(
                float(selected_lambda), int(mask.sum()) * horizons
            ),
            "horizon": np.tile(np.arange(1, horizons + 1), int(mask.sum())),
            "y_true": np.asarray(y)[mask].reshape(-1),
            "y_pred": np.asarray(prediction)[mask].reshape(-1),
            "har_pred": np.asarray(har)[mask].reshape(-1),
        }
    )


def metric_row(
    y: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    fold: int,
    model_name: str,
    readout_family: str,
    selected_lambda: float,
    diagnostics: dict[str, float] | None = None,
) -> dict[str, object]:
    payload = _metric_payload(y, prediction, mask)
    return {
        "fold": int(fold),
        "model_name": model_name,
        "readout_family": readout_family,
        "selected_lambda": float(selected_lambda),
        "validation_rows": int(np.asarray(mask, dtype=bool).sum()),
        **({} if diagnostics is None else diagnostics),
        **{key: float(value) for key, value in payload.items()},
    }


def grouped_metric_rows(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    fold: int,
    model_name: str,
    readout_family: str,
    selected_lambda: float,
) -> list[dict[str, object]]:
    selected = frame.loc[mask].reset_index(drop=True)
    y_selected = np.asarray(y)[mask]
    prediction_selected = np.asarray(prediction)[mask]
    rows: list[dict[str, object]] = []
    for lead in sorted(selected["lead"].unique()):
        for label in sorted(selected["label"].unique()):
            local = (
                selected["lead"].eq(lead).to_numpy()
                & selected["label"].eq(label).to_numpy()
            )
            if not local.any():
                continue
            payload = _metric_payload(
                y_selected[local],
                prediction_selected[local],
                np.ones(int(local.sum()), dtype=bool),
            )
            rows.append(
                {
                    "fold": int(fold),
                    "model_name": model_name,
                    "readout_family": readout_family,
                    "selected_lambda": float(selected_lambda),
                    "lead": int(lead),
                    "label": int(label),
                    "samples": int(local.sum()),
                    **{key: float(value) for key, value in payload.items()},
                }
            )
    return rows


def confirmation_gate(
    pooled_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Predeclared gate for deciding whether to expose the common test block."""

    ladder_name = "frozen_ladder_symmetric_1p25"
    har_name = "har"
    ladder = pooled_metrics.loc[pooled_metrics["model_name"].eq(ladder_name)]
    har = pooled_metrics.loc[pooled_metrics["model_name"].eq(har_name)]
    if len(ladder) != 1 or len(har) != 1:
        raise ValueError("confirmation gate requires one ladder and one HAR row")
    ladder_row = ladder.iloc[0]
    har_row = har.iloc[0]
    paired = fold_metrics.pivot(
        index="fold", columns="model_name", values="qlike"
    )
    required = {ladder_name, har_name}
    if not required.issubset(paired.columns):
        raise ValueError("fold metrics do not contain both gate models")
    fold_improvements = int((paired[ladder_name] <= paired[har_name]).sum())
    criteria = {
        "pooled_qlike_below_har": bool(
            float(ladder_row["qlike"]) < float(har_row["qlike"])
        ),
        "pooled_rmse_within_0p01_of_har": bool(
            float(ladder_row["rmse"]) <= float(har_row["rmse"]) + 0.01
        ),
        "qlike_no_worse_in_at_least_3_of_5_folds": bool(
            fold_improvements >= 3
        ),
    }
    return {
        "gate_version": 1,
        "purpose": "decide whether to run the common untouched test block",
        "folds_improving_or_tying_har": fold_improvements,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
    }
