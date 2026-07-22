from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.qrc.representation_screen_analysis import _metric_payload
from transition_forecasting.qrc.temporal_rydberg_chain import effective_rank

OrderControl = Literal["ordered", "reset", "shuffled", "reversed", "block_shuffled"]
ReadoutMode = Literal["direct", "prequential_har_residual"]
SUPPORTED_CONTROLS: tuple[OrderControl, ...] = (
    "ordered", "reset", "shuffled", "reversed", "block_shuffled"
)
OBSERVABLE_FAMILIES = ("all", "occupation", "pair", "connected", "summary")


@dataclass(frozen=True)
class InstabilityMechanismAssayConfig:
    folds: tuple[int, ...] = (1, 2, 3)
    leads: tuple[int, ...] = (1, 5, 10)
    max_per_class: int = 12
    interaction_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    control_scales: tuple[float, ...] = (0.5, 1.0)
    controls: tuple[OrderControl, ...] = ("reset", "shuffled", "reversed", "block_shuffled")
    block_size: int = 5
    alphas: tuple[float, ...] = (100.0, 1000.0, 10000.0)
    pca_components: tuple[int, ...] = (1, 2, 4, 8, 16, 0)
    pls_components: tuple[int, ...] = (1, 2, 4, 8)
    individual_pc_count: int = 16
    readout_modes: tuple[ReadoutMode, ...] = ("direct", "prequential_har_residual")
    prequential_blocks: int = 5
    sequence_length: int = 40
    seed: int = 20260721
    level_channel_name: str = "log_volatility_level"
    fallback_level_channel: int = 0

    def validate(self) -> None:
        if not self.folds or not self.leads:
            raise ValueError("folds and leads cannot be empty")
        if self.max_per_class < 1 or self.sequence_length < 2:
            raise ValueError("max_per_class and sequence_length must be positive")
        if not self.interaction_scales or any(x < 0 for x in self.interaction_scales):
            raise ValueError("interaction scales must be nonnegative")
        if any(x not in self.interaction_scales for x in self.control_scales):
            raise ValueError("control scales must also appear in interaction_scales")
        if set(self.controls).difference(SUPPORTED_CONTROLS) or "ordered" in self.controls:
            raise ValueError("controls must be non-ordered supported controls")
        if self.block_size < 1 or self.individual_pc_count < 1:
            raise ValueError("block size and PC count must be positive")
        if any(x <= 0 for x in self.alphas + self.pls_components):
            raise ValueError("alphas and PLS components must be positive")
        if any(x < 0 for x in self.pca_components):
            raise ValueError("PCA components must be nonnegative; zero means full")
        if set(self.readout_modes).difference({"direct", "prequential_har_residual"}):
            raise ValueError("unsupported readout mode")
        if self.prequential_blocks < 2:
            raise ValueError("prequential_blocks must be at least two")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def case_specs(config: InstabilityMechanismAssayConfig) -> tuple[tuple[str, float], ...]:
    config.validate()
    values = [("ordered", float(x)) for x in config.interaction_scales]
    values += [(str(c), float(x)) for x in config.control_scales for c in config.controls]
    return tuple(values)


def case_name(control: str, scale: float) -> str:
    return f"{control}_interaction_{scale:.2f}".replace(".", "p")


def reorder_level(level: np.ndarray, control: OrderControl, *, seed: int, block_size: int) -> np.ndarray:
    level = np.asarray(level, dtype=float)
    if level.ndim != 2 or not np.isfinite(level).all():
        raise ValueError("level must be finite with shape (samples, time)")
    if control in {"ordered", "reset"}:
        return level.copy()
    if control == "reversed":
        return level[:, ::-1].copy()
    rng = np.random.default_rng(seed)
    out = level.copy()
    if control == "shuffled":
        for row in range(len(out)):
            out[row] = level[row, rng.permutation(level.shape[1])]
        return out
    if control == "block_shuffled":
        blocks = [np.arange(i, min(i + block_size, level.shape[1])) for i in range(0, level.shape[1], block_size)]
        for row in range(len(out)):
            out[row] = level[row, np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])]
        return out
    raise ValueError(f"unsupported control: {control}")


def family_indices(names: tuple[str, ...], family: str) -> np.ndarray:
    if family == "all":
        return np.arange(len(names), dtype=int)
    if family.startswith("probe_"):
        selected = [i for i, name in enumerate(names) if name.startswith(f"{family}_")]
    elif family == "occupation":
        selected = [i for i, name in enumerate(names) if "occupation_site_" in name]
    elif family == "pair":
        selected = [i for i, name in enumerate(names) if "nearest_pair_" in name or "long_pair_" in name]
    elif family == "connected":
        selected = [i for i, name in enumerate(names) if "_connected_" in name]
    elif family == "summary":
        selected = [i for i, name in enumerate(names) if "excitation_density" in name or "mean_domain_wall" in name]
    else:
        raise ValueError(f"unsupported family: {family}")
    if not selected:
        raise ValueError(f"family {family!r} selected no features")
    return np.asarray(selected, dtype=int)


def target_context(
    readout_mode: ReadoutMode,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    train_mask: np.ndarray,
    residual_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if readout_mode == "direct":
        return train_mask, y, np.zeros_like(y)
    return residual_mask, residuals, har


def evaluate_individual_pcs(
    features: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    train_mask: np.ndarray,
    residual_mask: np.ndarray,
    val_mask: np.ndarray,
    fold: int,
    case: str,
    control: str,
    scale: float,
    alphas: tuple[float, ...],
    readout_modes: tuple[ReadoutMode, ...],
    pc_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scaler = StandardScaler().fit(features[train_mask])
    scaled = scaler.transform(features)
    count = min(pc_count, int(train_mask.sum()), features.shape[1])
    pca = PCA(n_components=count, svd_solver="full").fit(scaled[train_mask])
    scores = pca.transform(scaled)
    metrics: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for pc in range(count):
        x = scores[:, [pc]]
        correlations = []
        for horizon in range(y.shape[1]):
            a, b = x[train_mask, 0], y[train_mask, horizon]
            correlations.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan)
        diagnostics.append({
            "fold": fold, "case": case, "control": control, "interaction_scale": scale,
            "pc": pc + 1, "explained_variance": float(pca.explained_variance_ratio_[pc]),
            "mean_abs_train_horizon_correlation": float(np.nanmean(np.abs(correlations))),
            "max_abs_train_horizon_correlation": float(np.nanmax(np.abs(correlations))),
        })
        for mode in readout_modes:
            fit_mask, fit_y, baseline = target_context(
                mode, y=y, har=har, residuals=residuals, train_mask=train_mask, residual_mask=residual_mask
            )
            for alpha in alphas:
                model = Ridge(alpha=float(alpha)).fit(x[fit_mask], fit_y[fit_mask])
                prediction = baseline + model.predict(x)
                metrics.append({
                    "fold": fold, "case": case, "control": control, "interaction_scale": scale,
                    "analysis": "individual_pc", "observable_family": "all", "readout_mode": mode,
                    "components": pc + 1, "alpha": float(alpha),
                    **{f"val_{k}": v for k, v in _metric_payload(y, prediction, val_mask).items()},
                })
    return metrics, diagnostics


def evaluate_pls(
    features: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    train_mask: np.ndarray,
    residual_mask: np.ndarray,
    val_mask: np.ndarray,
    fold: int,
    case: str,
    control: str,
    scale: float,
    components: tuple[int, ...],
    readout_modes: tuple[ReadoutMode, ...],
) -> list[dict[str, object]]:
    scaled = StandardScaler().fit(features[train_mask]).transform(features)
    rows = []
    for mode in readout_modes:
        fit_mask, fit_y, baseline = target_context(
            mode, y=y, har=har, residuals=residuals, train_mask=train_mask, residual_mask=residual_mask
        )
        maximum = min(int(fit_mask.sum()) - 1, features.shape[1])
        for count in sorted({min(int(x), maximum) for x in components if maximum >= 1}):
            model = PLSRegression(n_components=count, scale=False, max_iter=1000).fit(scaled[fit_mask], fit_y[fit_mask])
            prediction = baseline + model.predict(scaled)
            rows.append({
                "fold": fold, "case": case, "control": control, "interaction_scale": scale,
                "analysis": "pls", "observable_family": "all", "readout_mode": mode,
                "components": count, "alpha": np.nan,
                **{f"val_{k}": v for k, v in _metric_payload(y, prediction, val_mask).items()},
            })
    return rows


def family_diagnostics(
    features: np.ndarray,
    names: tuple[str, ...],
    *,
    train_mask: np.ndarray,
    fold: int,
    case: str,
    control: str,
    scale: float,
    probes: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for family in OBSERVABLE_FAMILIES + tuple(f"probe_{x}" for x in probes):
        matrix = features[train_mask][:, family_indices(names, family)]
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        variance = singular**2
        fraction = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
        cumulative = np.cumsum(fraction)
        dim = lambda q: int(np.searchsorted(cumulative, q) + 1) if cumulative.size and cumulative[-1] > 0 else 0
        rows.append({
            "fold": fold, "case": case, "control": control, "interaction_scale": scale,
            "observable_family": family, "features": matrix.shape[1],
            "effective_rank_train": effective_rank(matrix),
            "numerical_rank_train": int(np.linalg.matrix_rank(centered)),
            "dimensions_90pct_variance": dim(0.90), "dimensions_95pct_variance": dim(0.95),
            "near_constant_features": int((matrix.std(axis=0) < 1e-8).sum()),
        })
    return rows
