#!/usr/bin/env python3
"""Prepare the primary paper-style monthly volatility dataset.

One active preparation script. It preserves definitional alternatives until audit resolves them;
it does not silently deduplicate dates, select a price field, or guess unresolved paper fields.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def latest_snapshot(root: Path) -> Path:
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f'No snapshots under {root}')
    return candidates[-1]


def parse_yahoo_chart(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    chart = payload.get('chart', {})
    if chart.get('error') is not None:
        raise ValueError(f"Yahoo chart error: {chart['error']}")
    results = chart.get('result') or []
    if len(results) != 1:
        raise ValueError(f'Expected one Yahoo result, found {len(results)}')

    result = results[0]
    timestamps = result.get('timestamp') or []
    quote = (result.get('indicators', {}).get('quote') or [{}])[0]
    adj = (result.get('indicators', {}).get('adjclose') or [{}])[0]
    n = len(timestamps)
    if n == 0:
        raise ValueError('Yahoo result contains no timestamps')

    def aligned(name: str, values: list | None, required: bool = False) -> list:
        values = values or []
        if required and len(values) != n:
            raise ValueError(f'Yahoo {name} length {len(values)} != timestamp length {n}')
        if values and len(values) != n:
            raise ValueError(f'Yahoo {name} length {len(values)} != timestamp length {n}')
        return values if values else [None] * n

    df = pd.DataFrame({
        'date': pd.to_datetime(timestamps, unit='s', utc=True).tz_convert(None).normalize(),
        'open': aligned('open', quote.get('open')),
        'high': aligned('high', quote.get('high')),
        'low': aligned('low', quote.get('low')),
        'close': aligned('close', quote.get('close'), required=True),
        'volume': aligned('volume', quote.get('volume')),
        'adjclose': aligned('adjclose', adj.get('adjclose')),
    })
    for col in ['open', 'high', 'low', 'close', 'volume', 'adjclose']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df['date'].duplicated().any():
        dup = df.loc[df['date'].duplicated(keep=False), 'date'].astype(str).tolist()[:20]
        raise ValueError(f'Duplicate Yahoo trading dates; no silent deduplication: {dup}')
    if not df['date'].is_monotonic_increasing:
        raise ValueError('Yahoo dates are not strictly ordered')
    for col in ['close', 'adjclose']:
        finite = df[col].dropna()
        if len(finite) and (finite <= 0).any():
            raise ValueError(f'Non-positive {col} values found')
    return df


def build_monthly_target(daily: pd.DataFrame, price_field: str) -> pd.DataFrame:
    x = daily[['date', price_field]].dropna().copy()
    if x.empty:
        raise ValueError(f'No usable {price_field} observations')
    ret_col = f'log_return_{price_field}'
    x[ret_col] = np.log(x[price_field]).diff()
    x['month'] = x['date'].dt.to_period('M')

    grouped = x.groupby('month', sort=True)[ret_col]
    rv = grouped.apply(lambda s: float(np.sqrt(np.square(s.dropna()).sum())))
    n_returns = grouped.count()
    out = pd.DataFrame({
        'date': rv.index.to_timestamp('M'),
        f'n_daily_returns_{price_field}': n_returns.values,
        f'rv_{price_field}': rv.values,
    })
    out[f'log_rv_{price_field}'] = np.log(out[f'rv_{price_field}'].where(out[f'rv_{price_field}'] > 0))

    # Preserve both plausible interpretations of the paper's quarterly/annual RV averages.
    out[f'rv_q3_mean_{price_field}'] = out[f'rv_{price_field}'].rolling(3, min_periods=3).mean()
    out[f'rv_a12_mean_{price_field}'] = out[f'rv_{price_field}'].rolling(12, min_periods=12).mean()
    out[f'log_of_rv_q3_mean_{price_field}'] = np.log(out[f'rv_q3_mean_{price_field}'])
    out[f'log_of_rv_a12_mean_{price_field}'] = np.log(out[f'rv_a12_mean_{price_field}'])
    out[f'log_rv_q3_mean_{price_field}'] = out[f'log_rv_{price_field}'].rolling(3, min_periods=3).mean()
    out[f'log_rv_a12_mean_{price_field}'] = out[f'log_rv_{price_field}'].rolling(12, min_periods=12).mean()
    return out


def read_fred(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f'FRED file has fewer than two columns: {path}')
    date_col, value_col = df.columns[:2]
    out = df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: value_name})
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
    if out['date'].isna().any():
        raise ValueError(f'Unparseable FRED dates in {path}')
    if out['date'].duplicated().any():
        raise ValueError(f'Duplicate monthly dates in {path}')
    return out


def extract_monthly_french_zip(path: Path, output_names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]
        if len(names) != 1:
            raise ValueError(f'Expected one data file in {path.name}, found {names}')
        text = zf.read(names[0]).decode('utf-8', errors='replace')

    rows = []
    for line in text.splitlines():
        if re.match(r'^\s*\d{6}\s*,', line):
            rows.append([p.strip() for p in line.split(',')])
    if not rows:
        raise ValueError(f'No YYYYMM monthly rows found in {path}')
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    if width - 1 < len(output_names):
        raise ValueError(f'Expected {len(output_names)} factors, found {width - 1}')

    raw = pd.DataFrame(rows)
    out = pd.DataFrame({'date': pd.to_datetime(raw[0], format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp('M')})
    for i, name in enumerate(output_names, start=1):
        out[name] = pd.to_numeric(raw[i], errors='coerce') / 100.0
    if out['date'].isna().any() or out['date'].duplicated().any():
        raise ValueError(f'Invalid or duplicate French monthly dates in {path}')
    return out


def merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    if right['date'].duplicated().any():
        raise ValueError(f'Duplicate dates in {label}')
    return left.merge(right, on='date', how='left', validate='one_to_one')


def write_table(df: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_path = parquet_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f'Parquet engine unavailable; wrote CSV fallback to {csv_path}')
        return csv_path


def classify_missingness(column: str) -> str:
    if column.startswith('target_'):
        return 'future_target_censoring'
    if '__avail_lag1' in column:
        return 'intentional_information_availability_lag_or_source_missingness'
    if any(token in column for token in ['q3_mean', 'a12_mean']):
        return 'rolling_warmup_or_source_missingness'
    return 'source_or_alignment_missingness_requires_review'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, default=Path('data/raw/paper_monthly'))
    p.add_argument('--snapshot', type=Path, default=None)
    p.add_argument('--outdir', type=Path, default=Path('data/interim/paper_monthly'))
    args = p.parse_args()

    snap = args.snapshot or latest_snapshot(args.raw_root)
    args.outdir.mkdir(parents=True, exist_ok=True)

    daily = parse_yahoo_chart(snap / 'gspc_daily_yahoo.json')
    monthly_close = build_monthly_target(daily, 'close')
    monthly_adj = build_monthly_target(daily, 'adjclose') if daily['adjclose'].notna().any() else None
    monthly = monthly_close if monthly_adj is None else monthly_close.merge(monthly_adj, on='date', how='outer', validate='one_to_one')

    fred_specs = {
        'fred_TB3MS_three_month_tbill.csv': 'tb3ms',
        'fred_CPIAUCSL_cpi.csv': 'cpi_index',
        'fred_INDPRO_industrial_production.csv': 'ip_index',
        'fred_AAA_aaa_corporate_yield.csv': 'aaa_yield',
        'fred_BAA_baa_corporate_yield.csv': 'baa_yield',
    }
    for filename, value_name in fred_specs.items():
        monthly = merge_one_to_one(monthly, read_fred(snap / filename, value_name), value_name)

    monthly = merge_one_to_one(
        monthly,
        extract_monthly_french_zip(snap / 'ff3_monthly.zip', ['mkt_excess', 'smb', 'hml', 'rf']),
        'ff3',
    )
    monthly = merge_one_to_one(
        monthly,
        extract_monthly_french_zip(snap / 'short_term_reversal_monthly.zip', ['str']),
        'str',
    )

    monthly['inflation_log_change'] = 100.0 * np.log(monthly['cpi_index']).diff()
    monthly['ip_log_growth'] = 100.0 * np.log(monthly['ip_index']).diff()
    monthly['default_spread_baa_minus_aaa__candidate'] = monthly['baa_yield'] - monthly['aaa_yield']
    for col in ['inflation_log_change', 'ip_log_growth']:
        monthly[f'{col}__avail_lag1'] = monthly[col].shift(1)

    for field in ['close', 'adjclose']:
        target = f'log_rv_{field}'
        if target in monthly:
            monthly[f'target_{target}_t_plus_1'] = monthly[target].shift(-1)
            monthly[f'target_{target}_t_plus_5'] = monthly[target].shift(-5)

    comparison = {}
    if {'log_rv_close', 'log_rv_adjclose'} <= set(monthly.columns):
        both = monthly[['log_rv_close', 'log_rv_adjclose']].dropna()
        diff = (both['log_rv_close'] - both['log_rv_adjclose']).abs()
        comparison = {
            'n_months_both': int(len(both)),
            'mean_abs_log_rv_difference': None if both.empty else float(diff.mean()),
            'max_abs_log_rv_difference': None if both.empty else float(diff.max()),
        }

    missing_rows = []
    for col in monthly.columns:
        n = int(monthly[col].isna().sum())
        if n:
            missing_rows.append({'column': col, 'missing_count': n, 'category': classify_missingness(col)})

    monthly_path = write_table(monthly, args.outdir / 'prepared_core.parquet')
    daily_path = write_table(daily, args.outdir / 'daily_source_prepared.parquet')
    pd.DataFrame(missing_rows).to_csv(args.outdir / 'missingness_categories.csv', index=False)

    metadata = {
        'dataset': 'paper_monthly',
        'raw_snapshot': str(snap),
        'monthly_output': str(monthly_path),
        'daily_output': str(daily_path),
        'target_formula': 'log(sqrt(sum(daily log return^2))) by calendar month',
        'price_fields_preserved': [c for c in ['close', 'adjclose'] if c in daily and daily[c].notna().any()],
        'close_vs_adjusted_comparison': comparison,
        'quarterly_annual_variants_preserved': ['mean RV level then log', 'mean monthly log RV'],
        'unresolved_before_exact_paper_parity': [
            'DP and EP exact source-field mapping',
            'exact default-spread definition versus transparent BAA-AAA candidate',
            'publication-lag treatment for macro variables',
            'which quarterly/annual log convention matches authors implementation',
        ],
        'rule': 'Unresolved definitions remain explicit; no guessed field is silently promoted.',
    }
    (args.outdir / 'preparation_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f'Prepared {len(monthly)} monthly rows from {monthly.date.min()} to {monthly.date.max()}')


if __name__ == '__main__':
    main()
