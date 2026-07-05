#!/usr/bin/env python3
"""Build preliminary, leakage-aware feature families for the paper-monthly dataset.

This script does not fit scalers, PCA, or supervised selectors. It creates only deterministic
features available at month t for forecasting future monthly log realized volatility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = 'log_rv_close'


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def rolling_prior_robust_z(s: pd.Series, window: int) -> pd.Series:
    """Current value relative to trailing median/MAD using prior months only."""
    prior = s.shift(1)
    med = prior.rolling(window, min_periods=window).median()
    mad = prior.rolling(window, min_periods=window).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )
    denom = 1.4826 * mad
    return (s - med) / denom.where(denom > 0)


def add_feature(
    out: pd.DataFrame,
    catalog: list[dict],
    name: str,
    values: pd.Series,
    family: str,
    role: str,
    formula: str,
    timing: str = 'available by end of month t',
) -> None:
    out[name] = values
    catalog.append({
        'feature': name,
        'family': family,
        'role': role,
        'formula': formula,
        'timing': timing,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--input', type=Path,
        default=Path('data/processed/paper_monthly/paper_parity_1950_02_to_2017_12.parquet'),
    )
    p.add_argument(
        '--output', type=Path,
        default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'),
    )
    p.add_argument(
        '--catalog', type=Path,
        default=Path('data/processed/paper_monthly/features/feature_catalog.csv'),
    )
    p.add_argument(
        '--manifest', type=Path,
        default=Path('data/processed/paper_monthly/features/feature_manifest.json'),
    )
    args = p.parse_args()

    df = read_table(args.input)
    required = {
        'date', TARGET, 'mkt_excess', 'smb', 'hml', 'str',
        'tb3ms', 'aaa_yield', 'baa_yield',
        'default_spread_baa_minus_aaa__candidate',
        'inflation_log_change__avail_lag1', 'ip_log_growth__avail_lag1',
    }
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise KeyError(f'Missing required columns: {missing_required}')

    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].duplicated().any() or not df['date'].is_monotonic_increasing:
        raise ValueError('Input dates must be unique and increasing')

    y = pd.to_numeric(df[TARGET], errors='coerce')
    out = pd.DataFrame({'date': df['date']})
    catalog: list[dict] = []

    # 1. Volatility state: direct and multiscale memory.
    add_feature(out, catalog, 'vol_state_1m', y, 'volatility_state', 'current state', TARGET)
    add_feature(out, catalog, 'vol_state_3m_mean', y.rolling(3, min_periods=3).mean(),
                'volatility_state', 'short memory', 'mean(log_rv[t-2:t])')
    add_feature(out, catalog, 'vol_state_12m_mean', y.rolling(12, min_periods=12).mean(),
                'volatility_state', 'long memory', 'mean(log_rv[t-11:t])')

    # 2. Dynamics: direction, curvature, and separation of timescales.
    d1 = y.diff(1)
    add_feature(out, catalog, 'vol_change_1m', d1, 'volatility_dynamics', 'rate of change', 'log_rv[t] - log_rv[t-1]')
    add_feature(out, catalog, 'vol_change_3m', y - y.shift(3), 'volatility_dynamics', 'medium change', 'log_rv[t] - log_rv[t-3]')
    add_feature(out, catalog, 'vol_acceleration_1m', d1.diff(1), 'volatility_dynamics', 'acceleration', 'delta1[t] - delta1[t-1]')
    add_feature(out, catalog, 'vol_short_minus_long', out['vol_state_3m_mean'] - out['vol_state_12m_mean'],
                'volatility_dynamics', 'timescale separation', 'mean3(log_rv) - mean12(log_rv)')
    add_feature(out, catalog, 'vol_prior_12m_robust_z', rolling_prior_robust_z(y, 12),
                'volatility_dynamics', 'robust surprise', 'current log_rv relative to prior-12-month median/MAD')

    # 3. Market drivers: monthly realized factor shocks are already changes, not levels.
    for col in ['mkt_excess', 'smb', 'hml', 'str']:
        add_feature(out, catalog, f'market_{col}', pd.to_numeric(df[col], errors='coerce'),
                    'market_drivers', 'market shock', col)

    # 4. Credit/rates state and movement.
    for col in ['tb3ms', 'aaa_yield', 'baa_yield', 'default_spread_baa_minus_aaa__candidate']:
        s = pd.to_numeric(df[col], errors='coerce')
        base = col.replace('__candidate', '')
        add_feature(out, catalog, f'credit_{base}_level', s,
                    'credit_rates', 'financial conditions state', col)
        add_feature(out, catalog, f'credit_{base}_change_1m', s.diff(1),
                    'credit_rates', 'financial conditions change', f'{col}[t] - {col}[t-1]')

    # 5. Macro dynamics: use only explicitly lagged availability variants at this stage.
    macro_specs = {
        'inflation_log_change__avail_lag1': 'macro_inflation_growth_lag1',
        'ip_log_growth__avail_lag1': 'macro_ip_growth_lag1',
    }
    for source, name in macro_specs.items():
        s = pd.to_numeric(df[source], errors='coerce')
        add_feature(out, catalog, name, s, 'macro_dynamics', 'lagged macro growth', source,
                    timing='one-month availability lag enforced')
        add_feature(out, catalog, f'{name}_change_1m', s.diff(1), 'macro_dynamics', 'macro acceleration',
                    f'{source}[t] - {source}[t-1]', timing='one-month availability lag enforced')

    # Targets are retained for downstream protocol construction, never as candidate features.
    out['target_log_rv_t_plus_1'] = y.shift(-1)
    out['target_log_rv_t_plus_5'] = y.shift(-5)

    feature_cols = [c for c in out.columns if c not in {'date', 'target_log_rv_t_plus_1', 'target_log_rv_t_plus_5'}]
    exact_duplicate_pairs = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            if out[a].equals(out[b]):
                exact_duplicate_pairs.append({'a': a, 'b': b})
    if exact_duplicate_pairs:
        raise ValueError(f'Exact duplicate engineered features found: {exact_duplicate_pairs}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_csv(args.catalog, index=False)

    missingness = {
        c: int(out[c].isna().sum()) for c in out.columns if out[c].isna().any()
    }
    manifest = {
        'input': str(args.input),
        'output': str(args.output),
        'rows': int(len(out)),
        'date_min': str(out['date'].min()),
        'date_max': str(out['date'].max()),
        'feature_count': int(len(feature_cols)),
        'families': sorted(pd.DataFrame(catalog)['family'].unique().tolist()),
        'targets': ['target_log_rv_t_plus_1', 'target_log_rv_t_plus_5'],
        'missingness': missingness,
        'rules': [
            'no scaler, PCA, imputation, or supervised selection is fitted here',
            'all engineered predictors use information at or before month t',
            'macro features use explicit one-month availability-lag variants',
            'frequency-domain features are deferred pending null-tested design',
            'feature promotion requires out-of-sample validation',
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
