from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
    _parse_mixed_utc,
)

SelectionPolicy = Literal["qlike", "balanced"]
CalibrationMode = Literal["global", "horizon"]


@dataclass(frozen=True)
class FrozenChainReadoutTuningConfig:
    case: str = "ordered_interaction_1p00"
    observable_family: str = "occupation"
    inner_holdout_fraction: float = 0.25
    ridge_alphas: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0, 1000.0)
    global_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    pca_prefixes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 12)
    pc_bands: tuple[str, ...] = ("5", "2-5", "4-8", "5-8", "9-12")
    pls_components: tuple[int, ...] = (1, 2, 3, 4)
    gate_types: tuple[str, ...] = (
        "none",
        "instability_high_65",
        "instability_ramp_50_90",
        "har_slope_ramp_50_90",
        "joint_ramp_50_90",
    )
    calibration_modes: tuple[CalibrationMode, ...] = ("global", "horizon")
    selection_policies: tuple[SelectionPolicy, ...] = ("qlike", "balanced")
    balanced_rmse_weight: float = 1.0
    fixed_reference_components: int = 4
    fixed_reference_alpha: float = 100.0

    def validate(self) -> None:
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if not self.global_lambdas or any(value < 0 for value in self.global_lambdas):
            raise ValueError("global_lambdas must be nonnegative")
        if not self.pca_prefixes or any(value < 1 for value in self.pca_prefixes):
            raise ValueError("pca_prefixes must be positive")
        if any(value < 1 for value in self.pls_components):
            raise ValueError("pls_components must be positive")
        if set(self.calibration_modes).difference({"global", "horizon"}):
            raise ValueError("unsupported calibration mode")
        if set(self.selection_policies).difference({"qlike", "balanced"}):
            raise ValueError("unsupported selection policy")
        if self.balanced_rmse_weight < 0:
            raise ValueError("balanced_rmse_weight must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenChainArchive:
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    encoded_sequence: np.ndarray
    target_path: np.ndarray
    har_prediction_path: np.ndarray
    prequential_residual_path: np.ndarray
    prequential_residual_valid: np.ndarray
    fold: np.ndarray
    sample_id: np.ndarray
    fold_split: np.ndarray
    lead: np.ndarray
    label: np.ndarray
    episode_id: np.ndarray
    origin_date: np.ndarray
    case: np.ndarray


def load_frozen_chain_archive(path: Path, *, case: str) -> FrozenChainArchive:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "feature_matrix",
            "feature_names",
            "encoded_sequence",
            "target_path",
            "har_prediction_path",
            "prequential_residual_path",
            "prequential_residual_valid",
            "fold",
            "sample_id",
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
            "case",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"archive is missing arrays: {sorted(missing)}")
        case_values = np.asarray(bundle["case"]).astype(str)
        mask = case_values == str(case)
        if not mask.any():
            raise ValueError(f"case {case!r} does not occur in {path}")

        def take(name: str, dtype: object | None = None) -> np.ndarray:
            values = np.asarray(bundle[name])[mask]
            return values.astype(dtype) if dtype is not None else values

        feature_names = tuple(str(value) for value in np.asarray(bundle["feature_names"]))
        archive = FrozenChainArchive(
            feature_matrix=take("feature_matrix", float),
            feature_names=feature_names,
            encoded_sequence=take("encoded_sequence", float),
            target_path=take("target_path", float),
            har_prediction_path=take("har_prediction_path", float),
            prequential_residual_path=take("prequential_residual_path", float),
            prequential_residual_valid=take("prequential_residual_valid", bool),
            fold=take("fold", int),
            sample_id=take("sample_id", str),
            fold_split=take("fold_split", str),
            lead=take("lead", int),
            label=take("label", int),
            episode_id=take("episode_id", str),
            origin_date=take("origin_date", str),
            case=take("case", str),
        )
    n = len(archive.fold)
    arrays = {
        "feature_matrix": len(archive.feature_matrix),
        "encoded_sequence": len(archive.encoded_sequence),
        "target_path": len(archive.target_path),
        "har_prediction_path": len(archive.har_prediction_path),
        "prequential_residual_path": len(archive.prequential_residual_path),
        "prequential_residual_valid": len(archive.prequential_residual_valid),
        "sample_id": len(archive.sample_id),
        "fold_split": len(archive.fold_split),
        "lead": len(archive.lead),
        "label": len(archive.label),
        "episode_id": len(archive.episode_id),
        "origin_date": len(archive.origin_date),
    }
    if any(length != n for length in arrays.values()):
        raise ValueError(f"archive arrays are not aligned: rows={n}, arrays={arrays}")
    if archive.feature_matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match feature_matrix width")
    if not np.isfinite(archive.feature_matrix).all():
        raise ValueError("feature_matrix contains non-finite values")
    return archive


def observable_indices(names: tuple[str, ...], family: str) -> np.ndarray:
    if family == "all":
        selected = list(range(len(names)))
    elif family == "occupation":
        selected = [index for index, name in enumerate(names) if "occupation_site_" in name]
    elif family == "connected":
        selected = [index for index, name in enumerate(names) if "_connected_" in name]
    elif family == "pair":
        selected = [
            index
            for index, name in enumerate(names)
            if "nearest_pair_" in name or "long_pair_" in name
        ]
    else:
        raise ValueError(f"unsupported observable family: {family}")
    if not selected:
        raise ValueError(f"observable family {family!r} selected no features")
    return np.asarray(selected, dtype=int)


def parse_pc_band(specification: str, width: int) -> tuple[int, ...]:
    text = str(specification).strip()
    if "-" in text:
        left_text, right_text = text.split("-", maxsplit=1)
        left, right = int(left_text), int(right_text)
    else:
        left = right = int(text)
    if left < 1 or right < left:
        raise ValueError(f"invalid PC band: {specification!r}")
    indices = tuple(index - 1 for index in range(left, right + 1) if index <= width)
    if not indices:
        raise ValueError(f"PC band {specification!r} exceeds available width {width}")
    return indices


def chronological_inner_split(
    origin_date: np.ndarray,
    eligible: np.ndarray,
    *,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"origin_date": np.asarray(origin_date).astype(str)})
    dates = _parse_mixed_utc(frame["origin_date"])
    mask = np.asarray(eligible, dtype=bool)
    unique_dates = np.asarray(sorted(dates[mask].unique()))
    if len(unique_dates) < 4:
        raise ValueError("not enough unique training dates for an inner split")
    holdout_dates = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    holdout_dates = min(holdout_dates, len(unique_dates) - 2)
    cutoff = unique_dates[-holdout_dates]
    inner_fit = mask & dates.lt(cutoff).to_numpy()
    inner_tune = mask & dates.ge(cutoff).to_numpy()
    if inner_fit.sum() < 10 or inner_tune.sum() < 5:
        raise ValueError(
            "inner split is too small: "
            f"fit={int(inner_fit.sum())}, tune={int(inner_tune.sum())}"
        )
    return inner_fit, inner_tune


def _ramp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("gate thresholds must be finite")
    if high <= low + 1e-12:
        return (np.asarray(values, dtype=float) >= high).astype(float)
    return np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)


def gate_values(
    gate_type: str,
    *,
    encoded_sequence: np.ndarray,
    har: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    instability = np.asarray(encoded_sequence, dtype=float)[:, -5:, 1].mean(axis=1)
    har_slope = np.abs(np.asarray(har, dtype=float)[:, -1] - np.asarray(har, dtype=float)[:, 0])
    fit = np.asarray(fit_mask, dtype=bool)
    if gate_type == "none":
        return np.ones(len(instability), dtype=float)
    if gate_type == "instability_high_65":
        threshold = float(np.quantile(instability[fit], 0.65))
        return (instability >= threshold).astype(float)
    instability_gate = _ramp(
        instability,
        float(np.quantile(instability[fit], 0.50)),
        float(np.quantile(instability[fit], 0.90)),
    )
    if gate_type == "instability_ramp_50_90":
        return instability_gate
    har_gate = _ramp(
        har_slope,
        float(np.quantile(har_slope[fit], 0.50)),
        float(np.quantile(har_slope[fit], 0.90)),
    )
    if gate_type == "har_slope_ramp_50_90":
        return har_gate
    if gate_type == "joint_ramp_50_90":
        return np.sqrt(instability_gate * har_gate)
    raise ValueError(f"unsupported gate type: {gate_type}")


def _qlike_single_horizon(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    local_y = np.asarray(y, dtype=float)[:, None]
    local_prediction = np.asarray(prediction, dtype=float)[:, None]
    return float(_metric_payload(local_y, local_prediction, mask)["qlike"])


def choose_calibration(
    correction: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    tune_mask: np.ndarray,
    gate: np.ndarray,
    mode: CalibrationMode,
    lambdas: tuple[float, ...],
    policy: SelectionPolicy,
    rmse_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    gated = np.asarray(correction, dtype=float) * np.asarray(gate, dtype=float)[:, None]
    har_payload = _metric_payload(y, har, tune_mask)
    if mode == "global":
        best_lambda = min(
            lambdas,
            key=lambda value: _candidate_score(
                _metric_payload(
                    y,
                    har + float(value) * gated,
                    tune_mask,
                ),
                har_payload,
                policy,
                rmse_weight=rmse_weight,
            ),
        )
        values = np.repeat(float(best_lambda), y.shape[1])
    elif mode == "horizon":
        selected = []
        for horizon in range(y.shape[1]):
            local_har_payload = _metric_payload(
                y[:, [horizon]],
                har[:, [horizon]],
                tune_mask,
            )
            selected.append(
                min(
                    lambdas,
                    key=lambda value: _candidate_score(
                        _metric_payload(
                            y[:, [horizon]],
                            (
                                har[:, horizon]
                                + float(value) * gated[:, horizon]
                            )[:, None],
                            tune_mask,
                        ),
                        local_har_payload,
                        policy,
                        rmse_weight=rmse_weight,
                    ),
                )
            )
        values = np.asarray(selected, dtype=float)
    else:
        raise ValueError(f"unsupported calibration mode: {mode}")
    prediction = har + gated * values[None, :]
    return prediction, values


def _fit_pca_ridge_correction(
    features: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    indices: tuple[int, ...],
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(features[fit_mask])
    scaled = scaler.transform(features)
    component_count = max(indices) + 1
    maximum = min(int(fit_mask.sum()), features.shape[1])
    if component_count > maximum:
        raise ValueError("requested PC indices exceed fit rank")
    pca = PCA(n_components=component_count, svd_solver="full").fit(scaled[fit_mask])
    scores = pca.transform(scaled)[:, indices]
    model = Ridge(alpha=float(alpha)).fit(scores[fit_mask], residuals[fit_mask])
    return np.asarray(model.predict(scores), dtype=float)


def _fit_pls_correction(
    features: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    components: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(features[fit_mask])
    scaled = scaler.transform(features)
    maximum = min(int(fit_mask.sum()) - 1, features.shape[1])
    if components > maximum:
        raise ValueError("PLS components exceed fit rank")
    model = PLSRegression(
        n_components=int(components),
        scale=False,
        max_iter=1000,
    ).fit(scaled[fit_mask], residuals[fit_mask])
    return np.asarray(model.predict(scaled), dtype=float)


def _candidate_category(model_kind: str, feature_spec: str, gate_type: str) -> tuple[str, ...]:
    categories = ["full"]
    if model_kind == "pca" and feature_spec == "prefix_4":
        categories.append("gating")
        if gate_type == "none":
            categories.append("shrinkage")
    if gate_type == "none":
        categories.append("component")
    return tuple(dict.fromkeys(categories))


def _candidate_score(
    payload: dict[str, float],
    har_payload: dict[str, float],
    policy: SelectionPolicy,
    *,
    rmse_weight: float,
) -> float:
    if policy == "qlike":
        return float(payload["qlike"])
    qlike_ratio = float(payload["qlike"]) / max(float(har_payload["qlike"]), 1e-12)
    rmse_ratio = float(payload["rmse"]) / max(float(har_payload["rmse"]), 1e-12)
    return qlike_ratio + float(rmse_weight) * rmse_ratio


def _candidate_specs(config: FrozenChainReadoutTuningConfig, width: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for count in config.pca_prefixes:
        if count <= width:
            for alpha in config.ridge_alphas:
                specs.append({
                    "model_kind": "pca",
                    "feature_spec": f"prefix_{count}",
                    "indices": tuple(range(count)),
                    "components": count,
                    "alpha": float(alpha),
                })
    for band in config.pc_bands:
        try:
            indices = parse_pc_band(band, width)
        except ValueError:
            continue
        for alpha in config.ridge_alphas:
            specs.append({
                "model_kind": "pca",
                "feature_spec": f"band_{band}",
                "indices": indices,
                "components": len(indices),
                "alpha": float(alpha),
            })
    for count in config.pls_components:
        if count <= width:
            specs.append({
                "model_kind": "pls",
                "feature_spec": f"pls_{count}",
                "indices": tuple(),
                "components": count,
                "alpha": np.nan,
            })
    return specs


def _fit_correction(
    spec: dict[str, object],
    *,
    features: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    if spec["model_kind"] == "pca":
        return _fit_pca_ridge_correction(
            features,
            residuals,
            fit_mask,
            indices=tuple(int(value) for value in spec["indices"]),
            alpha=float(spec["alpha"]),
        )
    return _fit_pls_correction(
        features,
        residuals,
        fit_mask,
        components=int(spec["components"]),
    )


def _metadata_frame(archive: FrozenChainArchive, mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "fold": archive.fold[mask],
        "sample_id": archive.sample_id[mask],
        "fold_split": archive.fold_split[mask],
        "lead": archive.lead[mask],
        "label": archive.label[mask],
        "episode_id": archive.episode_id[mask],
        "origin_date": archive.origin_date[mask],
    }).reset_index(drop=True)


def _prediction_rows(
    metadata: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    model_name: str,
    policy: str,
    category: str,
) -> pd.DataFrame:
    horizons = y.shape[1]
    payload = {
        column: np.repeat(metadata[column].to_numpy(), horizons)
        for column in metadata.columns
    }
    payload.update({
        "model_name": np.repeat(model_name, len(metadata) * horizons),
        "selection_policy": np.repeat(policy, len(metadata) * horizons),
        "category": np.repeat(category, len(metadata) * horizons),
        "horizon": np.tile(np.arange(1, horizons + 1), len(metadata)),
        "y_true": y.reshape(-1),
        "y_pred": prediction.reshape(-1),
    })
    return pd.DataFrame(payload)


def _group_metrics(
    metadata: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    fold: int,
    model_name: str,
    policy: str,
    category: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lead in sorted(metadata["lead"].unique()):
        for label in sorted(metadata["label"].unique()):
            local = metadata["lead"].eq(lead).to_numpy() & metadata["label"].eq(label).to_numpy()
            if not local.any():
                continue
            rows.append({
                "fold": int(fold),
                "model_name": model_name,
                "selection_policy": policy,
                "category": category,
                "lead": int(lead),
                "label": int(label),
                "samples": int(local.sum()),
                **_metric_payload(y[local], prediction[local], np.ones(int(local.sum()), dtype=bool)),
            })
    return rows


def _select_candidate(
    candidates: pd.DataFrame,
    *,
    category: str,
    policy: SelectionPolicy,
) -> pd.Series:
    subset = candidates.loc[
        candidates["categories"].str.split("|").apply(
            lambda values: category in values
        )
        & candidates["calibration_policy"].eq(policy)
    ].copy()
    if subset.empty:
        raise ValueError(f"no candidates belong to category {category!r}")
    return subset.sort_values([f"score_{policy}", "inner_qlike", "inner_rmse"]).iloc[0]
