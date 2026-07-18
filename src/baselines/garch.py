"""GARCH volatility baseline mechanics for Phase 3 and transition forecasting.

This module owns only the econometric model mechanics. Dataset loading,
walk-forward geometry, artifact writing, and reporting belong to experiment
runners.

The implementation is adapted from the reset-branch monthly GARCH prototype,
but the monthly protocol and target-specific assumptions are intentionally not
carried into the canonical Phase 3 branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class GARCHConfig:
    """Configuration for the canonical univariate GARCH comparator."""

    p: int = 1
    q: int = 1
    mean: str = "Zero"
    distribution: str = "StudentsT"
    return_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.p < 1 or self.q < 1:
            raise ValueError("GARCH orders p and q must be positive")
        if self.return_scale <= 0:
            raise ValueError("return_scale must be positive")


@dataclass(frozen=True)
class GARCHForecast:
    """One fitted GARCH variance forecast path."""

    variance_path: np.ndarray
    converged: bool
    convergence_flag: int | None
    parameters: dict[str, float]
    note: str


def _require_arch() -> Any:
    """Import ``arch`` lazily so the core package stays usable without it."""

    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "GARCH baseline requires the optional dependency 'arch'. "
            "Install the baseline extras before running GARCH experiments."
        ) from exc
    return arch_model


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional array")
    if len(values) < 20:
        raise ValueError("Need at least 20 returns to fit GARCH")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns contain non-finite values")
    return values


def fit_garch_variance_path(
    returns: np.ndarray,
    *,
    horizon: int,
    config: GARCHConfig | None = None,
) -> GARCHForecast:
    """Fit GARCH and forecast the conditional-variance path."""

    if horizon < 1:
        raise ValueError("horizon must be positive")

    cfg = config or GARCHConfig()
    values = _validate_returns(returns)
    arch_model = _require_arch()

    try:
        model = arch_model(
            values * cfg.return_scale,
            mean=cfg.mean,
            vol="Garch",
            p=cfg.p,
            q=cfg.q,
            dist=cfg.distribution,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        parameters = {
            str(name): float(value)
            for name, value in result.params.items()
        }
        params_finite = bool(np.all(np.isfinite(list(parameters.values()))))
        convergence_flag = int(result.convergence_flag)
        converged = convergence_flag == 0 and params_finite

        forecast = result.forecast(
            horizon=horizon,
            reindex=False,
            method="analytic",
        )
        variance_path = forecast.variance.iloc[-1].to_numpy(dtype=float)
        if variance_path.shape != (horizon,):
            raise RuntimeError(
                "Unexpected GARCH forecast shape: "
                f"{variance_path.shape}; expected {(horizon,)}"
            )

        note = "converged" if converged else f"nonconvergence flag={convergence_flag}"
        return GARCHForecast(
            variance_path=variance_path,
            converged=converged,
            convergence_flag=convergence_flag,
            parameters=parameters,
            note=note,
        )
    except Exception as exc:
        return GARCHForecast(
            variance_path=np.full(horizon, np.nan),
            converged=False,
            convergence_flag=None,
            parameters={},
            note=f"exception:{type(exc).__name__}:{exc}",
        )


def variance_path_to_log_volatility_path(
    variance_path: np.ndarray,
    *,
    return_scale: float = 100.0,
    variance_floor: float = 1e-16,
) -> np.ndarray:
    """Convert scaled conditional variances to daily log-volatility.

    ``arch`` is fit to returns multiplied by ``return_scale``. Dividing the
    forecast variance by ``return_scale**2`` restores raw-return variance. The
    Stage D target is daily log Parkinson volatility, so the comparable GARCH
    quantity is ``log(sqrt(variance))`` at each forecast horizon, with no
    annualization or path aggregation.
    """

    path = np.asarray(variance_path, dtype=float)
    if path.ndim != 1 or len(path) == 0:
        raise ValueError("variance_path must be a non-empty one-dimensional array")
    if return_scale <= 0:
        raise ValueError("return_scale must be positive")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    if not np.all(np.isfinite(path)):
        return np.full(path.shape, np.nan, dtype=float)
    if np.any(path < 0):
        raise ValueError("variance_path contains negative values")
    raw_variance = np.maximum(path / (return_scale**2), variance_floor)
    return 0.5 * np.log(raw_variance)


def variance_path_to_realized_volatility(
    variance_path: np.ndarray,
    *,
    return_scale: float = 100.0,
    annualization_period: float | None = None,
) -> float:
    """Aggregate a variance path to realized-volatility units."""

    path = np.asarray(variance_path, dtype=float)
    if path.ndim != 1 or len(path) == 0:
        raise ValueError("variance_path must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(path)):
        return float("nan")
    if np.any(path < 0):
        raise ValueError("variance_path contains negative values")
    if return_scale <= 0:
        raise ValueError("return_scale must be positive")
    if annualization_period is not None and annualization_period <= 0:
        raise ValueError("annualization_period must be positive when provided")

    aggregate_variance = float(np.sum(path / (return_scale**2)))
    if annualization_period is not None:
        aggregate_variance *= annualization_period / len(path)
    return float(np.sqrt(aggregate_variance))


def config_to_dict(config: GARCHConfig) -> dict[str, Any]:
    """Return a JSON-safe configuration dictionary."""

    return asdict(config)
