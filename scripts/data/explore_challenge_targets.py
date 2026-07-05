#!/usr/bin/env python3
"""Compare candidate forward-volatility horizons for the challenge task.

This is target-design EDA only. It does not fit forecasting models.
It compares overlapping daily origins with non-overlapping anchor views and
summarizes persistence, transition frequency, and stakeholder-relevant event counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)


def forward_rv_sq(log_returns: pd.Series, horizon: int) -> pd.Series:
    sq = log_returns.pow(2)
    return sq.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def trailing_rv_sq(log_returns: pd.Series, horizon: int) -> pd.Series:
    return log_returns.pow(2).rolling(horizon).sum()


def safe_log_rv(rv_sq: pd.Series) -> pd.Series:
    return 0.5 * np.log(rv_sq.where(rv_sq > 0))


def acf(series: pd.Series, lag: int) -> float:
    x = series.dropna()
    if len(x) <= lag:
        return float('nan')
    return float(x.autocorr(lag=lag))


def regime_labels(values: pd.Series, q_low: float, q_high: float) -> pd.Series:
    out = pd.Series(index=values.index, dtype='object')
    out.loc[values <= q_low] = 'calm'
    out.loc[(values > q_low) & (values < q_high)] = 'middle'
    out.loc[values >= q_high] = 'turbulent'
    return out


def transition_type(current: pd.Series, future: pd.Series) -> pd.Series:
    return current.astype(str) + '_to_' + future.astype(str)


def summarize_view(df: pd.DataFrame, horizon: int, view: str) -> tuple[dict, pd.DataFrame]:
    y = df[f'forward_log_rv_{horizon}d']
    current = df[f'trailing_log_rv_{horizon}d']
    valid = y.notna() & current.notna()
    x = df.loc[valid].copy()
    yv = x[f'forward_log_rv_{horizon}d']
    cv = x[f'trailing_log_rv_{horizon}d']

    q_low = float(yv.quantile(1/3))
    q_high = float(yv.quantile(2/3))
    x['current_regime'] = regime_labels(cv, q_low, q_high)
    x['future_regime'] = regime_labels(yv, q_low, q_high)
    x['transition'] = transition_type(x['current_regime'], x['future_regime'])

    summary = {
        'horizon_days': horizon,
        'view': view,
        'rows': int(len(x)),
        'date_min': str(x['date'].min().date()),
        'date_max': str(x['date'].max().date()),
        'target_mean': float(yv.mean()),
        'target_std': float(yv.std()),
        'target_acf_lag1': acf(yv, 1),
        'target_acf_lag_horizon': acf(yv, horizon if view == 'overlapping_daily' else 1),
        'current_future_spearman': float(cv.corr(yv, method='spearman')),
        'q33_target': q_low,
        'q67_target': q_high,
        'calm_to_turbulent': int((x['transition'] == 'calm_to_turbulent').sum()),
        'turbulent_to_calm': int((x['transition'] == 'turbulent_to_calm').sum()),
        'all_regime_changes': int((x['current_regime'] != x['future_regime']).sum()),
        'direct_extreme_transitions': int(x['transition'].isin(['calm_to_turbulent','turbulent_to_calm']).sum()),
        'transition_fraction': float((x['current_regime'] != x['future_regime']).mean()),
    }

    transitions = (
        x.groupby(['current_regime', 'future_regime'], dropna=False)
        .size()
        .rename('count')
        .reset_index()
    )
    transitions.insert(0, 'view', view)
    transitions.insert(0, 'horizon_days', horizon)
    return summary, transitions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--input', type=Path,
        default=Path('data/interim/paper_monthly/daily_source_prepared.parquet'),
    )
    p.add_argument(
        '--outdir', type=Path,
        default=Path('results/challenge_primary/target_eda'),
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    required = {'date', 'close'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    df = df[['date', 'close']].dropna().copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].duplicated().any() or not df['date'].is_monotonic_increasing:
        raise ValueError('Daily dates must be unique and increasing')
    if (df['close'] <= 0).any():
        raise ValueError('Non-positive close values found')

    df['log_return'] = np.log(df['close']).diff()

    for h in HORIZONS:
        df[f'trailing_log_rv_{h}d'] = safe_log_rv(trailing_rv_sq(df['log_return'], h))
        df[f'forward_log_rv_{h}d'] = safe_log_rv(forward_rv_sq(df['log_return'], h))

    summaries = []
    transition_tables = []
    for h in HORIZONS:
        overlapping = df[['date', f'trailing_log_rv_{h}d', f'forward_log_rv_{h}d']].copy()
        s, t = summarize_view(overlapping, h, 'overlapping_daily')
        summaries.append(s)
        transition_tables.append(t)

        anchor_idx = np.arange(0, len(df), h)
        anchored = df.iloc[anchor_idx][['date', f'trailing_log_rv_{h}d', f'forward_log_rv_{h}d']].copy()
        s, t = summarize_view(anchored, h, 'nonoverlapping_anchor')
        summaries.append(s)
        transition_tables.append(t)

    summary_df = pd.DataFrame(summaries)
    transitions_df = pd.concat(transition_tables, ignore_index=True)

    # Event table for stakeholder inspection: strongest future volatility and largest upward changes.
    event_rows = []
    for h in HORIZONS:
        x = df[['date', f'trailing_log_rv_{h}d', f'forward_log_rv_{h}d']].dropna().copy()
        x['forward_minus_current'] = x[f'forward_log_rv_{h}d'] - x[f'trailing_log_rv_{h}d']
        top_future = x.nlargest(20, f'forward_log_rv_{h}d').assign(event_type='highest_future_volatility')
        top_jump = x.nlargest(20, 'forward_minus_current').assign(event_type='largest_upward_transition')
        for part in [top_future, top_jump]:
            part = part.assign(horizon_days=h)
            event_rows.append(part)
    events_df = pd.concat(event_rows, ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.outdir / 'horizon_summary.csv', index=False)
    transitions_df.to_csv(args.outdir / 'transition_counts.csv', index=False)
    events_df.to_csv(args.outdir / 'extreme_and_transition_events.csv', index=False)

    manifest = {
        'input': str(args.input),
        'horizons_days': list(HORIZONS),
        'views': ['overlapping_daily', 'nonoverlapping_anchor'],
        'target_definition': 'log(sqrt(sum of squared daily log returns over the next h trading days))',
        'current_state_definition': 'same realized-volatility formula over the trailing h trading days',
        'regime_definition_for_eda_only': 'calm/middle/turbulent by full-sample target terciles within each horizon/view',
        'warning': 'Full-sample terciles are descriptive only and must not be used as model-training thresholds.',
        'selection_rule': (
            'Choose a challenge target only after comparing persistence, overlap, effective sample size, '
            'transition counts, stakeholder interpretability, and later baseline difficulty.'
        ),
    }
    (args.outdir / 'target_eda_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(summary_df.to_string(index=False))
    print('\n' + json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
