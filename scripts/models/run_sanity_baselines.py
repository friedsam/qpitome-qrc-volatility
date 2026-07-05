#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

TARGET = 'target_log_rv_t_plus_1'
COMPACT7 = [
    'vol_state_1m',
    'market_mkt_excess',
    'market_str',
    'vol_state_3m_mean',
    'credit_default_spread_baa_minus_aaa_level',
    'macro_ip_growth_lag1',
    'macro_inflation_growth_lag1',
]


def score(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    yv, pv = np.exp(2 * y), np.exp(2 * p)
    r = yv / pv
    qlike = float(np.mean(r - np.log(r) - 1))
    X = np.column_stack([np.ones(len(p)), p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    ssr = float(np.sum((y - fit) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        'rmse_log_rv': rmse,
        'mae_log_rv': mae,
        'qlike_variance': qlike,
        'mz_intercept': float(beta[0]),
        'mz_slope': float(beta[1]),
        'mz_r2': float(1 - ssr / sst),
    }


def ridge_prediction(train: pd.DataFrame, pred_row: pd.Series, cols: list[str], y: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[cols])
    Xp = scaler.transform(pd.DataFrame([pred_row[cols].to_dict()]))
    return float(Ridge(alpha=1.0).fit(Xtr, y).predict(Xp)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'))
    p.add_argument('--catalog', type=Path, default=Path('data/processed/paper_monthly/features/feature_catalog.csv'))
    p.add_argument('--folds', type=Path, default=Path('results/paper_monthly/protocol/paper_rolling_one_step/folds.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/paper_monthly/models/sanity_baselines'))
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df['date'] = pd.to_datetime(df['date'])
    folds = pd.read_csv(args.folds, parse_dates=['forecast_date','prediction_origin','train_calendar_start','train_calendar_end'])
    all24 = pd.read_csv(args.catalog)['feature'].tolist()
    for col in COMPACT7:
        if col not in all24:
            raise KeyError(f'Compact feature missing from catalog: {col}')

    rows = []
    for _, f in folds.iterrows():
        train_mask = (
            (df['date'] >= f['train_calendar_start']) &
            (df['date'] <= f['train_calendar_end']) &
            df[all24].notna().all(axis=1) &
            df[TARGET].notna()
        )
        train = df.loc[train_mask]
        pred_row = df.loc[df['date'].eq(f['prediction_origin'])].iloc[0]
        y_train = train[TARGET].to_numpy(float)
        y_true = float(pred_row[TARGET])

        preds = {
            'historical_mean': float(y_train.mean()),
            'persistence': float(pred_row['vol_state_1m']),
        }
        ar1 = LinearRegression().fit(train[['vol_state_1m']], y_train)
        preds['linear_ar1'] = float(ar1.predict(pd.DataFrame({'vol_state_1m':[pred_row['vol_state_1m']]}))[0])
        preds['ridge_compact7_alpha1'] = ridge_prediction(train, pred_row, COMPACT7, y_train)
        preds['ridge_all24_alpha1'] = ridge_prediction(train, pred_row, all24, y_train)

        for model, pred in preds.items():
            rows.append({
                'fold_id': int(f['fold_id']),
                'forecast_date': f['forecast_date'],
                'model': model,
                'y_true_log_rv': y_true,
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
        'models': ['historical_mean','persistence','linear_ar1','ridge_compact7_alpha1','ridge_all24_alpha1'],
        'compact7_features': COMPACT7,
        'compact7_role': 'paper-informed sanity-check proxy, not exact paper QR1 or QR2 and not the challenge target definition',
        'selection': 'none; fixed baselines only',
        'ridge_alpha': 1.0,
        'forecast_count_per_model': int(len(folds)),
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
