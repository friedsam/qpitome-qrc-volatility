from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

Representation = Literal[
    "level_only",
    "level_rate",
    "level_instability",
    "level_positive_shock_energy",
    "level_slope_disagreement",
]

REPRESENTATIONS: tuple[Representation, ...] = (
    "level_only",
    "level_rate",
    "level_instability",
    "level_positive_shock_energy",
    "level_slope_disagreement",
)

SECOND_CHANNEL_NAMES: dict[Representation, str] = {
    "level_only": "zero_control",
    "level_rate": "raw_rate",
    "level_instability": "local_instability_5",
    "level_positive_shock_energy": "positive_shock_energy_5",
    "level_slope_disagreement": "slope_5_minus_20",
}


@dataclass(frozen=True)
class CandidateFeatureConfig:
    instability_window: int = 5
    shock_window: int = 5
    short_slope_window: int = 5
    long_slope_window: int = 20
    q_low: float = 0.01
    q_high: float = 0.99

    def validate(self) -> None:
        if self.instability_window < 1:
            raise ValueError("instability_window must be positive")
        if self.shock_window < 1:
            raise ValueError("shock_window must be positive")
        if self.short_slope_window < 2:
            raise ValueError("short_slope_window must be at least 2")
        if self.long_slope_window < self.short_slope_window:
            raise ValueError("long_slope_window must be at least short_slope_window")
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelScaler:
    medians: tuple[float, float]
    half_ranges: tuple[float, float]
    q_low: float
    q_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def raw_rate(level_windows: np.ndarray) -> np.ndarray:
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2:
        raise ValueError("level_windows must have shape (samples, time)")
    if not np.isfinite(level).all():
        raise ValueError("level_windows must be finite")
    return np.diff(level, axis=1, prepend=level[:, :1])


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError("values must have shape (samples, time)")
    if window < 1:
        raise ValueError("window must be positive")
    out = np.empty_like(data)
    cumulative = np.cumsum(data, axis=1)
    for end in range(data.shape[1]):
        start = max(0, end - window + 1)
        total = cumulative[:, end].copy()
        if start > 0:
            total -= cumulative[:, start - 1]
        out[:, end] = total / float(end - start + 1)
    return out


def local_instability(level_windows: np.ndarray, window: int = 5) -> np.ndarray:
    rate = raw_rate(level_windows)
    return np.sqrt(np.maximum(_trailing_mean(rate**2, window), 0.0))


def positive_shock_energy(level_windows: np.ndarray, window: int = 5) -> np.ndarray:
    rate = raw_rate(level_windows)
    positive = np.maximum(rate, 0.0)
    return _trailing_mean(positive**2, window)


def trailing_slope(level_windows: np.ndarray, window: int) -> np.ndarray:
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2 or not np.isfinite(level).all():
        raise ValueError("level_windows must be finite with shape (samples, time)")
    if window < 2:
        raise ValueError("window must be at least 2")
    out = np.zeros_like(level)
    for end in range(level.shape[1]):
        start = max(0, end - window + 1)
        width = end - start + 1
        if width < 2:
            continue
        x = np.arange(width, dtype=float)
        x -= x.mean()
        denominator = float(x @ x)
        y = level[:, start : end + 1]
        out[:, end] = (y @ x) / denominator
    return out


def slope_disagreement(
    level_windows: np.ndarray,
    short_window: int = 5,
    long_window: int = 20,
) -> np.ndarray:
    return trailing_slope(level_windows, short_window) - trailing_slope(
        level_windows, long_window
    )


def build_candidate_sequences(
    level_windows: np.ndarray,
    representation: Representation,
    config: CandidateFeatureConfig | None = None,
) -> np.ndarray:
    cfg = config or CandidateFeatureConfig()
    cfg.validate()
    level = np.asarray(level_windows, dtype=float)
    if level.ndim != 2 or not np.isfinite(level).all():
        raise ValueError("level_windows must be finite with shape (samples, time)")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")

    if representation == "level_only":
        second = np.zeros_like(level)
    elif representation == "level_rate":
        second = raw_rate(level)
    elif representation == "level_instability":
        second = local_instability(level, cfg.instability_window)
    elif representation == "level_positive_shock_energy":
        second = positive_shock_energy(level, cfg.shock_window)
    else:
        second = slope_disagreement(
            level,
            cfg.short_slope_window,
            cfg.long_slope_window,
        )
    return np.stack([level, second], axis=-1)


def _safe_center_scale(
    values: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot scale an empty or non-finite channel")
    median = float(np.median(finite))
    low = float(np.quantile(finite, q_low))
    high = float(np.quantile(finite, q_high))
    half = 0.5 * (high - low)
    if not np.isfinite(half) or half <= 1e-12:
        half = float(np.std(finite))
    if not np.isfinite(half) or half <= 1e-12:
        half = 1.0
    return median, half


def fit_channel_scaler(
    sequences: np.ndarray,
    train_mask: np.ndarray,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> ChannelScaler:
    values = np.asarray(sequences, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("sequences must have shape (samples, time, 2)")
    if mask.shape != (len(values),) or not mask.any():
        raise ValueError("train_mask must select at least one aligned sample")
    centers = [
        _safe_center_scale(values[mask, :, channel], q_low, q_high)
        for channel in range(2)
    ]
    return ChannelScaler(
        medians=(centers[0][0], centers[1][0]),
        half_ranges=(centers[0][1], centers[1][1]),
        q_low=float(q_low),
        q_high=float(q_high),
    )


def transform_candidate_sequences(
    sequences: np.ndarray,
    scaler: ChannelScaler,
) -> np.ndarray:
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("sequences must be finite with shape (samples, time, 2)")
    center = np.asarray(scaler.medians, dtype=float)[None, None, :]
    scale = np.asarray(scaler.half_ranges, dtype=float)[None, None, :]
    return np.clip((values - center) / scale, -1.0, 1.0)


def elementwise_quadratic_matrix(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("sequences must have shape (samples, time, 2)")
    level = values[:, :, 0]
    second = values[:, :, 1]
    basis = np.stack(
        [level, second, level**2, second**2, level * second],
        axis=-1,
    )
    return basis.reshape(len(values), -1)
