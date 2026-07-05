#!/usr/bin/env python3
"""Import the frozen legacy daily dataset as an isolated comparison dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('data/processed/phase3_spy_vix_volatility_extended.csv'))
    p.add_argument('--outdir', type=Path, default=Path('data/raw/legacy_daily'))
    args = p.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest_dir = args.outdir / stamp
    dest_dir.mkdir(parents=True, exist_ok=False)
    dest = dest_dir / args.input.name
    shutil.copy2(args.input, dest)
    manifest = {
        'dataset': 'legacy_daily',
        'snapshot_utc': stamp,
        'source_file': str(args.input.resolve()),
        'copied_file': str(dest),
        'sha256': sha256(dest),
        'role': 'comparison_only',
    }
    (dest_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
