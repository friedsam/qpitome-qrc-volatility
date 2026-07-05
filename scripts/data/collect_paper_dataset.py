#!/usr/bin/env python3
"""Collect public raw inputs for the paper-style monthly volatility benchmark.

Raw files are immutable snapshots. Preparation is intentionally separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

YAHOO_SYMBOL = '^GSPC'
FRED_SERIES = {
    'TB3MS': 'three_month_tbill',
    'CPIAUCSL': 'cpi',
    'INDPRO': 'industrial_production',
    'AAA': 'aaa_corporate_yield',
    'BAA': 'baa_corporate_yield',
}
FRENCH_URLS = {
    'ff3_monthly': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip',
    'short_term_reversal_monthly': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_CSV.zip',
}


def fetch(url: str, path: Path) -> dict:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 qrc-volatility-research'})
    with urlopen(req, timeout=60) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        'url': url,
        'path': str(path),
        'bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=Path, default=Path('data/raw/paper_monthly'))
    p.add_argument('--start', default='1950-01-01')
    args = p.parse_args()

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snap = args.outdir / stamp
    records = []

    start_epoch = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int(time.time())
    yahoo_url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(YAHOO_SYMBOL)}'
        f'?period1={start_epoch}&period2={end_epoch}&interval=1d&events=history&includeAdjustedClose=true'
    )
    records.append(fetch(yahoo_url, snap / 'gspc_daily_yahoo.json'))

    for series_id, name in FRED_SERIES.items():
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        records.append(fetch(url, snap / f'fred_{series_id}_{name}.csv'))

    for name, url in FRENCH_URLS.items():
        records.append(fetch(url, snap / f'{name}.zip'))

    manifest = {
        'dataset': 'paper_monthly',
        'snapshot_utc': stamp,
        'paper_reference': 'Li et al., arXiv:2505.13933',
        'records': records,
        'notes': [
            'Shiller DP/EP data are collected separately after exact source-field verification.',
            'Default spread definition is not assumed; AAA and BAA raw series are both retained.',
            'Raw files are never overwritten.',
        ],
    }
    (snap / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
