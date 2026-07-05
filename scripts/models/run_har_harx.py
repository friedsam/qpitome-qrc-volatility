#!/usr/bin/env python3
"""Run HAR and currently available public HARX references on the shared protocol.

HAR uses 1-, 3-, and 12-month log-RV state. HARX adds only resolved paper-grounded
public exogenous inputs. DP and EP are not included until their exact definitions are resolved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET = 'target_log_rv_t_plus_1'
HAR_FEATURES = ['vol_state_1m', 'vol_state_3m_mean', 'vol_state_12m_mean']
HARX_EXOGENOUS = [
    'market_mkt_excess',
    'market_smb',
    'market_hml',
    'market_str',
    'credit_tb3ms_level',
    'credit_default_spread_baa_minus_aaa_level',
    'macro_inflation_growth_lag1',
    'macro_ip_growth_lag1',
]


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'))
    p.add_argument('--folds', type=Path, default=Path('results/paper_monthly/protocol/paper_rolling_one_step/folds.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/paper_monthly/models/har_harx'))
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df['date'] = pd.to_datetime(df['date'])
    folds = pd.read_csv(args.folds, parse_dates=['forecast_date','prediction_origin','train_calendar_start','train_calendar_end'])

    model_features = {
        'har': HAR_FEATURES,
        'harx_public_available': HAR_FEATURES + HARX_EXOGENOUS,
    }
    rows = []

    for _, fold in folds.iterrows():
        for model_name, cols in model_features.items():
            train_mask = (
                (df['date'] >= fold['train_calendar_start']) &
                (df['date'] <= fold['train_calendar_end']) &
                df[cols].notna().all(axis=1) &
                df[TARGET].notna()
            )
            train = df.loc[train_mask]
            pred_row = df.loc[df['date'].eq(fold['prediction_origin'])].iloc[0]
            if pred_row[cols].isna().any():
                raise ValueError(f'Incomplete prediction row for {model_name} at {fold["prediction_origin"]}')

            reg = LinearRegression().fit(train[cols], train[TARGET])
            pred = float(reg.predict(pd.DataFrame([pred_row[cols].to_dict()]))[0])
            rows.append({
                'fold_id': int(fold['fold_id']),
                'forecast_date': fold['forecast_date'],
                'model': model_name,
                'y_true_log_rv': float(pred_row[TARGET]),
                'y_pred_log_rv': pred,
                'train_rows': int(len(train)),
            })

    pred = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.outdir / 'predictions.csv', index=False)

    metrics = []
    for model, g in pred.groupby('model'):
        metrics.append({'model': model, 'n_forecasts': len(g), **score(g['y_true_log_rv'].to_numpy(), g['y_pred_log_rv'].to_numpy())})
    metrics = pd.DataFrame(metrics).sort_values('rmse_log_rv')
    metrics.to_csv(args.outdir / 'metrics.csv', index=False)

    manifest = {
        'protocol': 'paper_rolling_one_step',
        'models': {
            'har': HAR_FEATURES,
            'harx_public_available': HAR_FEATURES + HARX_EXOGENOUS,
        },
        'estimator': 'ordinary least squares, refit every fold',
        'selection': 'none',
        'limitation': 'HARX excludes DP and EP until exact paper definitions are resolved',
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))
    print('\n' + json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
