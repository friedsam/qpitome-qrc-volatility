#!/usr/bin/env python3
"""Professional dataset audit with explicit missingness and integrity reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.csv', '.txt'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table format: {path.suffix}')


def contiguous_missing_runs(mask: pd.Series) -> tuple[int, int]:
    values = mask.fillna(False).to_numpy(bool)
    if not values.any():
        return 0, 0
    starts = np.where(values & ~np.r_[False, values[:-1]])[0]
    ends = np.where(values & ~np.r_[values[1:], False])[0]
    lengths = ends - starts + 1
    return int(len(lengths)), int(lengths.max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--date-column', default='date')
    p.add_argument('--dataset-name', required=True)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = read_table(args.input)

    if args.date_column in df.columns:
        parsed = pd.to_datetime(df[args.date_column], errors='coerce')
        bad_dates = int(parsed.isna().sum() - df[args.date_column].isna().sum())
        df[args.date_column] = parsed
    else:
        bad_dates = None

    schema_rows = []
    quality_rows = []
    for col in df.columns:
        s = df[col]
        miss = s.isna()
        n_runs, max_run = contiguous_missing_runs(miss)
        numeric = pd.to_numeric(s, errors='coerce') if not pd.api.types.is_datetime64_any_dtype(s) else None
        inf_count = 0 if numeric is None else int(np.isinf(numeric.to_numpy(float, na_value=np.nan)).sum())
        schema_rows.append({'column': col, 'dtype': str(s.dtype), 'n_unique': int(s.nunique(dropna=True))})
        quality_rows.append({
            'column': col,
            'n_rows': int(len(df)),
            'missing_count': int(miss.sum()),
            'missing_fraction': float(miss.mean()),
            'missing_runs': n_runs,
            'max_missing_run': max_run,
            'infinite_count': inf_count,
            'min': None if numeric is None or numeric.notna().sum() == 0 else float(numeric.min()),
            'max': None if numeric is None or numeric.notna().sum() == 0 else float(numeric.max()),
        })

    duplicate_rows = int(df.duplicated().sum())
    duplicate_dates = None
    monotonic_dates = None
    date_min = date_max = None
    if args.date_column in df.columns:
        duplicate_dates = int(df[args.date_column].duplicated().sum())
        monotonic_dates = bool(df[args.date_column].dropna().is_monotonic_increasing)
        date_min = str(df[args.date_column].min())
        date_max = str(df[args.date_column].max())

    manifest = {
        'dataset': args.dataset_name,
        'input_path': str(args.input),
        'input_sha256': sha256(args.input),
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'duplicate_rows': duplicate_rows,
        'duplicate_dates': duplicate_dates,
        'date_monotonic_increasing': monotonic_dates,
        'bad_date_parse_count': bad_dates,
        'date_min': date_min,
        'date_max': date_max,
        'rule': 'No silent row dropping. Missingness categories must be assigned before modeling.',
    }

    pd.DataFrame(schema_rows).to_csv(args.outdir / 'schema.csv', index=False)
    pd.DataFrame(quality_rows).to_csv(args.outdir / 'quality_report.csv', index=False)
    (args.outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print('\nColumns with missing or infinite values:')
    q = pd.DataFrame(quality_rows)
    flagged = q[(q.missing_count > 0) | (q.infinite_count > 0)]
    print(flagged.to_string(index=False) if len(flagged) else 'None')


if __name__ == '__main__':
    main()
