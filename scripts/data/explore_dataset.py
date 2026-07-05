#!/usr/bin/env python3
"""Generate a compact, reproducible EDA package for one prepared dataset.

Exploration can be restricted to a predeclared date cutoff so feature decisions are not informed by final test data.
The report separates target-state persistence from exogenous relationships and removes exact numeric duplicates from rankings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    z = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 5 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return None, int(len(z))
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method='spearman')), int(len(z))


def feature_family(name: str) -> str:
    if name.startswith(('rv_', 'log_rv_', 'log_of_rv_', 'n_daily_returns_')):
        return 'target_state'
    if name in {'mkt_excess', 'smb', 'hml', 'rf', 'str'}:
        return 'market_factors'
    if name in {'tb3ms', 'aaa_yield', 'baa_yield', 'default_spread_baa_minus_aaa__candidate'}:
        return 'rates_credit'
    if name in {'cpi_index', 'ip_index', 'inflation_log_change', 'ip_log_growth',
                'inflation_log_change__avail_lag1', 'ip_log_growth__avail_lag1'}:
        return 'macro'
    return 'other'


def canonical_duplicate_groups(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[dict]]:
    """Return canonical numeric columns and exact-duplicate mapping."""
    ordered = sorted(columns, key=lambda c: ('adjclose' in c, c))
    kept: list[str] = []
    duplicate_rows: list[dict] = []
    for col in ordered:
        s = pd.to_numeric(df[col], errors='coerce')
        duplicate_of = None
        for prior in kept:
            p = pd.to_numeric(df[prior], errors='coerce')
            if s.equals(p):
                duplicate_of = prior
                break
        if duplicate_of is None:
            kept.append(col)
        else:
            duplicate_rows.append({'duplicate_column': col, 'canonical_column': duplicate_of})
    return kept, duplicate_rows


def remove_target_duplicates(
    df: pd.DataFrame,
    columns: list[str],
    target_name: str,
    duplicate_rows: list[dict],
) -> list[str]:
    """Remove candidate columns exactly equal to the target itself."""
    target = pd.to_numeric(df[target_name], errors='coerce')
    kept: list[str] = []
    for col in columns:
        s = pd.to_numeric(df[col], errors='coerce')
        if s.equals(target):
            duplicate_rows.append({'duplicate_column': col, 'canonical_column': target_name})
        else:
            kept.append(col)
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--dataset-name', required=True)
    p.add_argument('--date-column', default='date')
    p.add_argument('--target', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--exploration-end', default=None, help='Inclusive YYYY-MM-DD cutoff; final test data remain unseen')
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = read_table(args.input)
    if args.date_column not in df or args.target not in df:
        raise KeyError(f'Required columns missing: date={args.date_column}, target={args.target}')
    df[args.date_column] = pd.to_datetime(df[args.date_column], errors='raise')
    if df[args.date_column].duplicated().any():
        raise ValueError('Duplicate dates found; EDA refuses to continue')
    if not df[args.date_column].is_monotonic_increasing:
        raise ValueError('Dates are not increasing; EDA refuses to continue')

    full_rows = len(df)
    if args.exploration_end:
        cutoff = pd.Timestamp(args.exploration_end)
        df = df[df[args.date_column] <= cutoff].copy()
    else:
        cutoff = None

    target = pd.to_numeric(df[args.target], errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid_target = target.dropna()
    if len(valid_target) < 20:
        raise ValueError(f'Only {len(valid_target)} valid target values')

    quantiles = valid_target.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).rename('value').reset_index().rename(columns={'index': 'quantile'})
    quantiles.to_csv(args.outdir / 'target_quantiles.csv', index=False)

    med = float(valid_target.median())
    mad = float((valid_target - med).abs().median())
    robust_z = pd.Series(np.nan, index=df.index, dtype=float)
    if mad > 0:
        robust_z = 0.67448975 * (target - med) / mad
    events = df[[args.date_column]].copy()
    events['target'] = target
    events['robust_z'] = robust_z
    events['abs_robust_z'] = robust_z.abs()
    events = events.nlargest(min(20, len(events)), 'abs_robust_z')
    events.to_csv(args.outdir / 'extreme_events.csv', index=False)

    acf_rows = []
    for lag in [1, 2, 3, 6, 12, 24, 36]:
        pair = pd.concat([target, target.shift(lag)], axis=1).dropna()
        acf_rows.append({'lag': lag, 'pearson_acf': None if len(pair) < 5 else float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 'n': int(len(pair))})
    pd.DataFrame(acf_rows).to_csv(args.outdir / 'target_acf.csv', index=False)

    excluded_feature_columns = {c for c in df.columns if c == args.target or c.startswith('target_')}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded_feature_columns
    ]
    canonical_cols, duplicate_rows = canonical_duplicate_groups(df, numeric_cols)
    canonical_cols = remove_target_duplicates(df, canonical_cols, args.target, duplicate_rows)
    duplicate_rows = sorted(duplicate_rows, key=lambda x: (x['canonical_column'], x['duplicate_column']))
    pd.DataFrame(duplicate_rows, columns=['duplicate_column', 'canonical_column']).to_csv(
        args.outdir / 'duplicate_numeric_columns.csv', index=False
    )

    corr_rows = []
    for col in canonical_cols:
        rho0, n0 = safe_spearman(pd.to_numeric(df[col], errors='coerce'), target)
        rho1, n1 = safe_spearman(pd.to_numeric(df[col], errors='coerce'), target.shift(-1))
        corr_rows.append({
            'feature': col,
            'family': feature_family(col),
            'rho_same_time': rho0,
            'n_same_time': n0,
            'rho_to_next_target': rho1,
            'n_next_target': n1,
        })
    corr = pd.DataFrame(corr_rows)
    if len(corr):
        corr['abs_rho_to_next_target'] = corr['rho_to_next_target'].abs()
        corr = corr.sort_values('abs_rho_to_next_target', ascending=False)
    corr.to_csv(args.outdir / 'feature_correlations.csv', index=False)

    family_top = (
        corr.sort_values(['family', 'abs_rho_to_next_target'], ascending=[True, False])
        .groupby('family', as_index=False, group_keys=False)
        .head(8)
        if len(corr) else pd.DataFrame()
    )
    family_top.to_csv(args.outdir / 'feature_correlations_by_family.csv', index=False)

    missing = pd.DataFrame({'column': df.columns, 'missing_count': [int(df[c].isna().sum()) for c in df], 'missing_fraction': [float(df[c].isna().mean()) for c in df]})
    missing = missing.sort_values(['missing_fraction', 'column'], ascending=[False, True])
    missing.to_csv(args.outdir / 'missingness_snapshot.csv', index=False)

    summary = {
        'dataset': args.dataset_name,
        'input': str(args.input),
        'full_rows_before_exploration_cutoff': int(full_rows),
        'exploration_rows': int(len(df)),
        'exploration_end': None if cutoff is None else str(cutoff.date()),
        'date_min': str(df[args.date_column].min().date()),
        'date_max': str(df[args.date_column].max().date()),
        'target': args.target,
        'target_valid_n': int(len(valid_target)),
        'target_mean': float(valid_target.mean()),
        'target_std': float(valid_target.std()),
        'target_median': med,
        'target_mad': mad,
        'numeric_feature_count_raw': int(len(numeric_cols)),
        'numeric_feature_count_canonical': int(len(canonical_cols)),
        'exact_duplicate_numeric_columns': int(len(duplicate_rows)),
        'excluded_future_target_columns': sorted(excluded_feature_columns - {args.target}),
    }
    (args.outdir / 'exploration_manifest.json').write_text(json.dumps(summary, indent=2))

    lines = [
        f'# Exploration summary: {args.dataset_name}', '',
        '## Scope', '',
        f'- Input: `{args.input}`',
        f'- Exploration rows: {len(df)} of {full_rows}',
        f'- Date range: {summary["date_min"]} to {summary["date_max"]}',
        f'- Target: `{args.target}`; valid n={len(valid_target)}',
        f'- Exploration cutoff: {summary["exploration_end"] or "none - review before using for feature decisions"}',
        f'- Numeric candidates: {len(numeric_cols)} raw, {len(canonical_cols)} after removing {len(duplicate_rows)} exact duplicates, including target duplicates', '',
        '## Target', '',
        f'- Mean: {valid_target.mean():.6g}',
        f'- Standard deviation: {valid_target.std():.6g}',
        f'- Median: {med:.6g}',
        f'- MAD: {mad:.6g}', '',
        'Extreme dates are listed in `extreme_events.csv`; they are observations to inspect, not rows to delete automatically.', '',
        '## Dependence', '',
    ]
    for row in acf_rows:
        lines.append(f'- Lag {row["lag"]}: ACF={row["pearson_acf"] if row["pearson_acf"] is not None else "NA"} (n={row["n"]})')

    lines += ['', '## Preliminary feature relationships by family', '',
              'These are exploratory Spearman correlations only. Exact numeric duplicates, target duplicates, and future-target columns are excluded. Target-state variables are reported separately from exogenous variables because persistence is not evidence that exogenous information is useless.', '']
    if len(family_top):
        for family in ['target_state', 'market_factors', 'rates_credit', 'macro', 'other']:
            block = family_top[family_top['family'] == family]
            if block.empty:
                continue
            lines += [f'### {family}', '']
            for _, row in block.iterrows():
                lines.append(f'- `{row.feature}`: rho to next target={row.rho_to_next_target:.3f} (n={int(row.n_next_target)})')
            lines.append('')
    else:
        lines.append('- No numeric candidate features available.')

    lines += ['## Data-quality link', '', 'This exploration does not replace the formal schema/missingness audit. Any unexpected missingness, duplicate dates, or impossible values blocks modeling.']
    (args.outdir / 'exploration_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
