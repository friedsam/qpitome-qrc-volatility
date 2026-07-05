#!/usr/bin/env python3
"""Collect public raw inputs for the paper-style monthly volatility benchmark.

Raw files are immutable snapshots. Preparation is intentionally separate.
Downloads are retried from scratch on transient network failures or incomplete reads.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
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

RETRYABLE = (
    http.client.IncompleteRead,
    HTTPError,
    URLError,
    TimeoutError,
    socket.timeout,
    ConnectionError,
)


def fetch(url: str, path: Path, *, attempts: int = 5, timeout: int = 120) -> dict:
    """Download one file atomically, retrying transient/incomplete responses.

    Every failed attempt is discarded. A raw file appears only after a complete
    response has been read successfully.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 qrc-volatility-research',
                'Accept-Encoding': 'identity',
                'Connection': 'close',
            },
        )
        try:
            chunks: list[bytes] = []
            with urlopen(req, timeout=timeout) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            content = b''.join(chunks)
            if not content:
                raise IOError(f'Empty response from {url}')

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_bytes(content)
            tmp.replace(path)
            return {
                'url': url,
                'path': str(path),
                'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest(),
                'download_attempts': attempt,
            }
        except RETRYABLE as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = 2 ** (attempt - 1)
            print(f'Download attempt {attempt}/{attempts} failed for {path.name}: {exc}')
            print(f'Retrying in {delay}s...')
            time.sleep(delay)

    raise RuntimeError(f'Failed to download {url} after {attempts} attempts') from last_error


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=Path, default=Path('data/raw/paper_monthly'))
    p.add_argument('--start', default='1950-01-01')
    p.add_argument('--attempts', type=int, default=5)
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
    print('Downloading S&P 500 daily history...')
    records.append(fetch(yahoo_url, snap / 'gspc_daily_yahoo.json', attempts=args.attempts))

    for series_id, name in FRED_SERIES.items():
        print(f'Downloading FRED {series_id}...')
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        records.append(fetch(url, snap / f'fred_{series_id}_{name}.csv', attempts=args.attempts))

    for name, url in FRENCH_URLS.items():
        print(f'Downloading Kenneth French {name}...')
        records.append(fetch(url, snap / f'{name}.zip', attempts=args.attempts))

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
