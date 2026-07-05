from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas_market_calendars as mcal
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GARCHConfig:
    p: int = 1
    q: int = 1
    mean: str = 'Zero'
    distribution: str = 'StudentsT'


@dataclass(frozen=True)
class GARCHForecast:
    variance_path: np.ndarray
    converged: bool
    note: str

    @property
    def monthly_variance_raw(self) -> float:
        if not self.converged or not np.all(np.isfinite(self.variance_path)):
            return float('nan')
        return max(float(self.variance_path.sum()) / 1e4, 1e-16)

    @property
    def log_rv_prediction(self) -> float:
        variance = self.monthly_variance_raw
        return 0.5 * float(np.log(variance)) if np.isfinite(variance) else float('nan')


def load_daily_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'date', 'close'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f'Missing required daily columns: {missing}')

    daily = frame[['date', 'close']].dropna(subset=['close']).copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='raise')
    daily = daily.sort_values('date').reset_index(drop=True)

    if daily['date'].duplicated().any():
        raise ValueError('Daily source contains duplicate dates')
    if (daily['close'] <= 0).any():
        raise ValueError('Daily source contains non-positive close values')

    daily['log_return'] = np.log(daily['close']).diff()
    return daily.dropna(subset=['log_return'])[['date', 'log_return']].reset_index(drop=True)


def count_trading_days(forecast_date: pd.Timestamp, calendar: str = 'NYSE') -> int:
    period = pd.Timestamp(forecast_date).to_period('M')
    schedule = mcal.get_calendar(calendar).schedule(
        start_date=period.start_time,
        end_date=period.end_time,
    )
    n_days = int(len(schedule))
    if n_days < 1:
        raise ValueError(f'No {calendar} trading days found for {period}')
    return n_days


def fit_garch_variance_path(
    returns: np.ndarray,
    horizon: int,
    config: GARCHConfig = GARCHConfig(),
) -> GARCHForecast:
    returns_pct = np.asarray(returns, dtype=float) * 100.0
    try:
        model = arch_model(
            returns_pct,
            mean=config.mean,
            vol='Garch',
            p=config.p,
            q=config.q,
            dist=config.distribution,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(disp='off', show_warning=False)

        params_finite = bool(np.all(np.isfinite(result.params.to_numpy())))
        converged = result.convergence_flag == 0 and params_finite
        variance_path = result.forecast(
            horizon=horizon,
            reindex=False,
            method='analytic',
        ).variance.iloc[-1].to_numpy(dtype=float)
        note = 'converged' if converged else f'nonconvergence flag={result.convergence_flag}'
        return GARCHForecast(variance_path=variance_path, converged=converged, note=note)
    except Exception as exc:
        return GARCHForecast(
            variance_path=np.full(horizon, np.nan),
            converged=False,
            note=f'exception:{type(exc).__name__}:{exc}',
        )
