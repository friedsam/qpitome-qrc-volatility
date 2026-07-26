"""GARCH volatility baseline mechanics with an ``arch``-compatible SciPy fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class GARCHConfig:
    p: int = 1
    q: int = 1
    mean: str = "Zero"
    distribution: str = "StudentsT"
    return_scale: float = 100.0
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.p != 1 or self.q != 1:
            raise ValueError("the canonical fallback supports GARCH(1,1) only")
        if self.return_scale <= 0:
            raise ValueError("return_scale must be positive")
        if self.backend not in {"auto", "arch", "scipy"}:
            raise ValueError("backend must be auto, arch, or scipy")


@dataclass(frozen=True)
class GARCHForecast:
    variance_path: np.ndarray
    converged: bool
    convergence_flag: int | None
    parameters: dict[str, float]
    note: str


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional array")
    if len(values) < 20:
        raise ValueError("Need at least 20 returns to fit GARCH")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns contain non-finite values")
    return values


def _fit_with_arch(values: np.ndarray, horizon: int, cfg: GARCHConfig) -> GARCHForecast:
    from arch import arch_model
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
    parameters = {str(name): float(value) for name, value in result.params.items()}
    params_finite = bool(np.all(np.isfinite(list(parameters.values()))))
    convergence_flag = int(result.convergence_flag)
    converged = convergence_flag == 0 and params_finite
    forecast = result.forecast(horizon=horizon, reindex=False, method="analytic")
    variance_path = forecast.variance.iloc[-1].to_numpy(dtype=float)
    if variance_path.shape != (horizon,):
        raise RuntimeError(f"Unexpected GARCH forecast shape: {variance_path.shape}")
    return GARCHForecast(
        variance_path=variance_path,
        converged=converged,
        convergence_flag=convergence_flag,
        parameters=parameters,
        note="converged:arch" if converged else f"nonconvergence:arch:{convergence_flag}",
    )


def _decode(theta: np.ndarray) -> tuple[float, float, float, float]:
    omega = float(np.exp(np.clip(theta[0], -30.0, 30.0)))
    a = float(np.exp(np.clip(theta[1], -20.0, 20.0)))
    b = float(np.exp(np.clip(theta[2], -20.0, 20.0)))
    total = 1.0 + a + b
    alpha = 0.999 * a / total
    beta = 0.999 * b / total
    nu = 2.05 + float(np.exp(np.clip(theta[3], -10.0, 10.0)))
    return omega, alpha, beta, nu


def _initial_theta(variance: float, alpha: float, beta: float, nu: float) -> np.ndarray:
    remaining = max(0.999 - alpha - beta, 1e-5)
    a = alpha / remaining
    b = beta / remaining
    omega = max(variance * max(1.0 - alpha - beta, 0.01), 1e-8)
    return np.log([omega, max(a, 1e-8), max(b, 1e-8), max(nu - 2.05, 1e-8)])


def _variance_recursion(values: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    from scipy.signal import lfilter
    initial = max(float(np.var(values, ddof=1)), 1e-8)
    innovations = np.empty_like(values)
    innovations[0] = initial
    innovations[1:] = omega + alpha * values[:-1] ** 2
    variance = lfilter([1.0], [1.0, -beta], innovations)
    return np.maximum(np.asarray(variance, dtype=float), 1e-12)


def _student_t_nll(theta: np.ndarray, values: np.ndarray) -> float:
    from scipy.special import gammaln
    omega, alpha, beta, nu = _decode(theta)
    variance = _variance_recursion(values, omega, alpha, beta)
    standardized_sq = values**2 / variance
    log_density = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * (nu - 2.0))
        - 0.5 * np.log(variance)
        - ((nu + 1.0) / 2.0) * np.log1p(standardized_sq / (nu - 2.0))
    )
    result = -float(np.sum(log_density))
    return result if np.isfinite(result) else 1e100


def _fit_with_scipy(values: np.ndarray, horizon: int, cfg: GARCHConfig) -> GARCHForecast:
    from scipy.optimize import minimize
    scaled = values * cfg.return_scale
    variance = max(float(np.var(scaled, ddof=1)), 1e-8)
    starts = (_initial_theta(variance, 0.05, 0.90, 8.0),)
    best = None
    for start in starts:
        result = minimize(
            _student_t_nll,
            start,
            args=(scaled,),
            method="L-BFGS-B",
            options={"maxiter": 80, "ftol": 1e-7, "maxls": 20},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    omega, alpha, beta, nu = _decode(best.x)
    conditional = _variance_recursion(scaled, omega, alpha, beta)
    first = omega + alpha * scaled[-1] ** 2 + beta * conditional[-1]
    forecast = np.empty(horizon, dtype=float)
    forecast[0] = max(first, 1e-12)
    persistence = alpha + beta
    for step in range(1, horizon):
        forecast[step] = max(omega + persistence * forecast[step - 1], 1e-12)
    finite = bool(np.all(np.isfinite(forecast)))
    converged = bool(best.success and finite)
    parameters = {"omega": omega, "alpha[1]": alpha, "beta[1]": beta, "nu": nu}
    return GARCHForecast(
        variance_path=forecast,
        converged=converged,
        convergence_flag=int(best.status),
        parameters=parameters,
        note=("converged:scipy" if converged else f"nonconvergence:scipy:{best.status}:{best.message}"),
    )


def fit_garch_variance_path(
    returns: np.ndarray,
    *,
    horizon: int,
    config: GARCHConfig | None = None,
) -> GARCHForecast:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    cfg = config or GARCHConfig()
    values = _validate_returns(returns)
    try:
        if cfg.backend in {"auto", "arch"}:
            try:
                return _fit_with_arch(values, horizon, cfg)
            except ImportError:
                if cfg.backend == "arch":
                    raise
        return _fit_with_scipy(values, horizon, cfg)
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
    path = np.asarray(variance_path, dtype=float)
    if path.ndim != 1 or len(path) == 0:
        raise ValueError("variance_path must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(path)):
        return np.full(path.shape, np.nan, dtype=float)
    if np.any(path < 0):
        raise ValueError("variance_path contains negative values")
    raw_variance = np.maximum(path / (return_scale**2), variance_floor)
    return 0.5 * np.log(raw_variance)


def config_to_dict(config: GARCHConfig) -> dict[str, Any]:
    return asdict(config)
