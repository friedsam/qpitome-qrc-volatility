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
    features = pd.read_csv(args.catalog)['feature'].tolist()
    rows = []

    for _, f in folds.iterrows():
        train_mask = (
            (df['date'] >= f['train_calendar_start']) &
            (df['date'] <= f['train_calendar_end']) &
            df[features].notna().all(axis=1) &
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

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[features])
        Xp = scaler.transform(pd.DataFrame([pred_row[features].to_dict()]))
        ridge = Ridge(alpha=1.0).fit(Xtr, y_train)
        preds['ridge_all24_alpha1'] = float(ridge.predict(Xp)[0])

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
        'models': ['historical_mean','persistence','linear_ar1','ridge_all24_alpha1'],
        'selection': 'none; fixed baselines only',
        'ridge_alpha': 1.0,
        'forecast_count_per_model': int(len(folds)),
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
