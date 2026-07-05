#!/usr/bin/env python3
"""Run the canonical GARCH baseline on the frozen monthly target.

The model is GARCH(1,1) with Student-t innovations on daily S&P 500 log returns.
Each fold uses daily returns from the protocol training-window start through the
prediction origin, forecasts the full next-month conditional-variance path,
and maps the aggregate variance into the target's log-RV units.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from arch import arch_model

TARGET = 'target_log_rv_t_plus_1'
MODEL_NAME = 'garch_1_1_t'
NYSE = mcal.get_calendar('NYSE')


def score(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    yv, pv = np.exp(2 * y), np.exp(2 * p)
    ratio = yv / pv
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    X = np.column_stack([np.ones(len(p)), p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ssr = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        'rmse_log_rv': rmse,
        'mae_log_rv': mae,
        'qlike_variance': qlike,
        'mz_intercept': float(beta[0]),
        'mz_slope': float(beta[1]),
        'mz_r2': float(1.0 - ssr / sst),
    }


def load_daily_returns(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    required = {'date', 'close'}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f'Missing required daily columns: {missing}')

    daily = raw[['date', 'close']].dropna(subset=['close']).copy()
    daily['date'] = pd.to_datetime(daily['date'], errors='raise')
    daily = daily.sort_values('date').reset_index(drop=True)
    if daily['date'].duplicated().any():
        raise ValueError('Daily source contains duplicate dates')
    if (daily['close'] <= 0).any():
        raise ValueError('Daily source contains non-positive close values')

    daily['log_return'] = np.log(daily['close']).diff()
    return daily.dropna(subset=['log_return'])[['date', 'log_return']].reset_index(drop=True)


def trading_days_in_month(forecast_date: pd.Timestamp) -> int:
    period = forecast_date.to_period('M')
    schedule = NYSE.schedule(start_date=period.start_time, end_date=period.end_time)
    n_days = int(len(schedule))
    if n_days < 1:
        raise ValueError(f'No NYSE trading days found for {period}')
    return n_days


def fit_and_forecast_variance_path(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, bool, str]:
    returns_pct = np.asarray(returns, dtype=float) * 100.0
    try:
        model = arch_model(
            returns_pct,
            mean='Zero',
            vol='Garch',
            p=1,
            q=1,
            dist='StudentsT',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(disp='off', show_warning=False)

        params_finite = bool(np.all(np.isfinite(result.params.to_numpy())))
        converged = result.convergence_flag == 0 and params_finite
        forecast = result.forecast(horizon=horizon, reindex=False, method='analytic')
        variance_path = forecast.variance.iloc[-1].to_numpy(dtype=float)
        note = 'converged' if converged else f'nonconvergence flag={result.convergence_flag}'
        return variance_path, converged, note
    except Exception as exc:
        note = f'exception:{type(exc).__name__}:{exc}'
        return np.full(horizon, np.nan), False, note


def main() -> None:
    p = argparse.ArgumentParser(description='GARCH(1,1)-t monthly volatility baseline')
    p.add_argument(
        '--features',
        type=Path,
        default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'),
    )
    p.add_argument(
        '--folds',
        type=Path,
        default=Path('results/paper_monthly/protocol/paper_rolling_one_step/folds.csv'),
    )
    p.add_argument(
        '--daily-source',
        type=Path,
        default=Path('data/interim/paper_monthly/daily_source_prepared.parquet'),
    )
    p.add_argument(
        '--outdir',
        type=Path,
        default=Path('results/challenge_primary/models/garch'),
    )
    args = p.parse_args()

    monthly = pd.read_parquet(args.features)
    monthly['date'] = pd.to_datetime(monthly['date'], errors='raise')
    folds = pd.read_csv(
        args.folds,
        parse_dates=['forecast_date', 'prediction_origin', 'train_calendar_start', 'train_calendar_end'],
    )
    daily = load_daily_returns(args.daily_source)

    rows: list[dict] = []
    for _, fold in folds.iterrows():
        train_mask = (
            (daily['date'] >= fold['train_calendar_start']) &
            (daily['date'] <= fold['prediction_origin'])
        )
        train_returns = daily.loc[train_mask, 'log_return'].to_numpy(dtype=float)
        n_days = trading_days_in_month(fold['forecast_date'])
        variance_path_pct2, converged, note = fit_and_forecast_variance_path(train_returns, n_days)

        if converged and np.all(np.isfinite(variance_path_pct2)):
            monthly_variance = max(float(variance_path_pct2.sum()) / 1e4, 1e-16)
            prediction = 0.5 * float(np.log(monthly_variance))
        else:
            monthly_variance = np.nan
            prediction = np.nan

        target_row = monthly.loc[monthly['date'].eq(fold['prediction_origin'])]
        if len(target_row) != 1:
            raise ValueError(f'Target row not found exactly once for {fold["prediction_origin"]}')

        rows.append({
            'fold_id': int(fold['fold_id']),
            'forecast_date': fold['forecast_date'],
            'prediction_origin': fold['prediction_origin'],
            'model': MODEL_NAME,
            'y_true_log_rv': float(target_row.iloc[0][TARGET]),
            'y_pred_log_rv': prediction,
            'variance_monthly_forecast_raw': monthly_variance,
            'n_trading_days_nyse': n_days,
            'train_daily_rows': int(len(train_returns)),
            'converged': bool(converged),
            'convergence_note': note,
        })

    predictions = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / 'predictions.csv', index=False)

    valid = predictions.loc[predictions['converged'] & predictions['y_pred_log_rv'].notna()].copy()
    if valid.empty:
        raise RuntimeError('No converged GARCH folds')

    metrics = {
        'model': MODEL_NAME,
        'n_forecasts': int(len(valid)),
        'n_failed_dropped': int(len(predictions) - len(valid)),
        **score(valid['y_true_log_rv'].to_numpy(), valid['y_pred_log_rv'].to_numpy()),
    }
    pd.DataFrame([metrics]).to_csv(args.outdir / 'metrics.csv', index=False)

    failed_ids = predictions.loc[~predictions['converged'], 'fold_id'].astype(int).tolist()
    manifest = {
        'protocol': 'paper_rolling_one_step development backbone',
        'model': MODEL_NAME,
        'specification': 'GARCH(1,1), zero mean, Student-t innovations, refit each fold',
        'daily_source': str(args.daily_source),
        'information_boundary': 'train_calendar_start <= daily date <= prediction_origin',
        'history_window': 'protocol train start through prediction origin; includes the origin month used to condition the forecast',
        'forecast_aggregation': 'sum analytic multi-step daily conditional variances over forecast month',
        'target_mapping': '0.5 * log(monthly conditional variance), in the same units as log realized volatility',
        'trading_day_count': 'NYSE calendar via pandas_market_calendars',
        'n_folds_total': int(len(predictions)),
        'n_converged': int(len(valid)),
        'n_failed': int(len(predictions) - len(valid)),
        'failed_fold_ids': failed_ids,
        'metrics_computed_on': 'converged folds only',
        'known_limitations': [
            'Conditional-variance forecasts depend on the fitted GARCH data-generating assumption.',
            'No leverage extension; asymmetry is deferred unless it becomes a specific research question.',
            'Parameters are refit independently for every fold.',
        ],
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f'\nFolds: total={len(predictions)} converged={len(valid)} failed={len(failed_ids)}')
    if failed_ids:
        print(f'Failed fold ids: {failed_ids}')


if __name__ == '__main__':
    main()
