#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FIRST_FORECAST = pd.Timestamp('1997-08-31')
LAST_FORECAST = pd.Timestamp('2017-12-31')
WINDOW_ORIGINS = 569
EXPECTED_FORECASTS = 245
TARGET_COLUMN = 'target_log_rv_t_plus_1'


def shift_month_end(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(months)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'))
    p.add_argument('--catalog', type=Path, default=Path('data/processed/paper_monthly/features/feature_catalog.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/paper_monthly/protocol/paper_rolling_one_step'))
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    catalog = pd.read_csv(args.catalog)
    feature_cols = catalog['feature'].tolist()
    required = {'date', TARGET_COLUMN, *feature_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].duplicated().any() or not df['date'].is_monotonic_increasing:
        raise ValueError('Input dates must be unique and increasing')

    forecast_dates = pd.date_range(FIRST_FORECAST, LAST_FORECAST, freq='ME')
    if len(forecast_dates) != EXPECTED_FORECASTS:
        raise ValueError(f'Expected {EXPECTED_FORECASTS} forecasts, got {len(forecast_dates)}')

    complete_features = df[feature_cols].notna().all(axis=1)
    eligible_train = complete_features & df[TARGET_COLUMN].notna()

    rows = []
    for fold_id, forecast_date in enumerate(forecast_dates):
        prediction_origin = shift_month_end(forecast_date, -1)
        train_end = shift_month_end(prediction_origin, -1)
        train_start = shift_month_end(train_end, -(WINDOW_ORIGINS - 1))

        calendar_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_mask = calendar_mask & eligible_train
        prediction_mask = df['date'].eq(prediction_origin)

        if int(calendar_mask.sum()) != WINDOW_ORIGINS:
            raise ValueError(f'Fold {fold_id}: wrong calendar window length')
        if int(prediction_mask.sum()) != 1:
            raise ValueError(f'Prediction origin not found exactly once: {prediction_origin.date()}')
        if not bool(complete_features.loc[prediction_mask].iloc[0]):
            raise ValueError(f'Prediction origin has incomplete features: {prediction_origin.date()}')

        train_dates = df.loc[train_mask, 'date']
        rows.append({
            'fold_id': fold_id,
            'forecast_date': forecast_date,
            'prediction_origin': prediction_origin,
            'train_calendar_start': train_start,
            'train_calendar_end': train_end,
            'train_calendar_origins': int(calendar_mask.sum()),
            'train_eligible_rows': int(train_mask.sum()),
            'train_first_eligible_origin': train_dates.min(),
            'train_last_eligible_origin': train_dates.max(),
        })

    folds = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.outdir / 'folds.csv', index=False)

    manifest = {
        'protocol': 'paper_rolling_one_step',
        'input': str(args.input),
        'target_column': TARGET_COLUMN,
        'feature_count': int(len(feature_cols)),
        'forecast_count': int(len(folds)),
        'first_forecast': str(folds['forecast_date'].min().date()),
        'last_forecast': str(folds['forecast_date'].max().date()),
        'calendar_window_origins': WINDOW_ORIGINS,
        'first_fold_train_calendar_start': str(folds.iloc[0]['train_calendar_start'].date()),
        'first_fold_train_calendar_end': str(folds.iloc[0]['train_calendar_end'].date()),
        'first_fold_train_eligible_rows': int(folds.iloc[0]['train_eligible_rows']),
        'last_fold_train_eligible_rows': int(folds.iloc[-1]['train_eligible_rows']),
        'rules': [
            'identical forecast dates for all model families',
            'identical rolling calendar windows for all model families',
            'common complete-case eligibility across all 24 preliminary predictors',
            'no protocol-level imputation',
            'preprocessing fitted within each training fold only',
            'the 245 out-of-sample targets are not used for hyperparameter selection',
            'primary task is one-step open-loop forecasting',
        ],
        'paper_alignment_note': (
            'The paper reports February 1950-June 1997 as approximately 571 months; '
            'the inclusive calendar count is 569. It reports 245 forecasts from August 1997 '
            'through December 2017. This protocol uses the explicit calendar count and causal '
            'origin-to-next-month target mapping.'
        ),
    }
    (args.outdir / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print('\nFirst fold:')
    print(folds.head(1).to_string(index=False))
    print('\nLast fold:')
    print(folds.tail(1).to_string(index=False))


if __name__ == '__main__':
    main()
