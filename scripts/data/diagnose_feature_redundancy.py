#!/usr/bin/env python3
"""Diagnose missingness and redundancy in engineered features without selecting features.

This script is descriptive only. It does not use targets, fit transforms, or promote/drop features.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--input', type=Path,
        default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'),
    )
    p.add_argument(
        '--catalog', type=Path,
        default=Path('data/processed/paper_monthly/features/feature_catalog.csv'),
    )
    p.add_argument(
        '--outdir', type=Path,
        default=Path('results/paper_monthly/features/preliminary_diagnostics'),
    )
    p.add_argument('--high-correlation-threshold', type=float, default=0.90)
    args = p.parse_args()

    df = read_table(args.input)
    catalog = pd.read_csv(args.catalog)
    feature_cols = catalog['feature'].tolist()

    missing = sorted(set(feature_cols) - set(df.columns))
    if missing:
        raise KeyError(f'Catalog features missing from table: {missing}')
    if df['date'].duplicated().any() or not pd.to_datetime(df['date']).is_monotonic_increasing:
        raise ValueError('Dates must be unique and increasing')

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    args.outdir.mkdir(parents=True, exist_ok=True)

    missingness = pd.DataFrame({
        'feature': feature_cols,
        'missing_count': [int(X[c].isna().sum()) for c in feature_cols],
        'missing_fraction': [float(X[c].isna().mean()) for c in feature_cols],
        'first_valid_date': [
            None if X[c].first_valid_index() is None else str(pd.to_datetime(df.loc[X[c].first_valid_index(), 'date']).date())
            for c in feature_cols
        ],
    }).merge(catalog[['feature', 'family', 'role']], on='feature', how='left', validate='one_to_one')
    missingness.to_csv(args.outdir / 'feature_missingness.csv', index=False)

    complete_mask = X.notna().all(axis=1)
    complete = df.loc[complete_mask, ['date'] + feature_cols].copy()
    complete.to_parquet(args.outdir / 'complete_case_features.parquet', index=False)

    corr = X.corr(method='spearman', min_periods=20)
    corr.to_csv(args.outdir / 'spearman_correlation_matrix.csv')

    pair_rows = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(float(rho)) >= args.high_correlation_threshold:
                pair_rows.append({
                    'feature_a': a,
                    'feature_b': b,
                    'spearman_rho': float(rho),
                    'abs_rho': abs(float(rho)),
                })
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs = pairs.sort_values('abs_rho', ascending=False)
    pairs.to_csv(args.outdir / 'high_correlation_pairs.csv', index=False)

    family_counts = catalog.groupby('family').size().rename('feature_count').reset_index()
    family_counts.to_csv(args.outdir / 'family_counts.csv', index=False)

    manifest = {
        'input': str(args.input),
        'feature_count': int(len(feature_cols)),
        'rows_total': int(len(df)),
        'rows_complete_all_features': int(complete_mask.sum()),
        'rows_lost_to_common_warmup_or_missingness': int((~complete_mask).sum()),
        'complete_date_min': None if complete.empty else str(pd.to_datetime(complete['date']).min().date()),
        'complete_date_max': None if complete.empty else str(pd.to_datetime(complete['date']).max().date()),
        'high_correlation_threshold': float(args.high_correlation_threshold),
        'high_correlation_pair_count': int(len(pairs)),
        'rule': 'descriptive only; no target use and no feature promotion/drop decision',
    }
    (args.outdir / 'diagnostic_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if len(pairs):
        print('\nHigh-correlation pairs:')
        print(pairs.to_string(index=False))


if __name__ == '__main__':
    main()
