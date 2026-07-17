from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.transition_data_audit import NAMES, load_ohlc


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    manifest_path: Path
    rows: int
    indices: int
    date_start: str
    date_end: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_transition_volatility_dataset(raw_root: Path, output_root: Path) -> BuildResult:
    output_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []

    for index_name in NAMES:
        path = raw_root / "global index etf return" / f"{index_name}.csv"
        frame = load_ohlc(path)
        parkinson = np.abs(np.log(frame["High"] / frame["Low"])) / np.sqrt(4 * np.log(2))
        result = pd.DataFrame(
            {
                "date": frame.index,
                "index": index_name,
                "parkinson_volatility": parkinson,
                "log_parkinson_volatility": np.log(parkinson.replace(0, np.nan)),
            }
        ).dropna()
        frames.append(result)
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(len(frame)),
                "date_start": str(frame.index.min().date()),
                "date_end": str(frame.index.max().date()),
            }
        )

    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "index"])
    output_path = output_root / "daily_parkinson_volatility.csv"
    combined.to_csv(output_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "eight_index_parkinson_volatility",
        "raw_root": str(raw_root),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": int(len(combined)),
        "indices": sorted(combined["index"].unique().tolist()),
        "date_start": str(pd.to_datetime(combined["date"]).min().date()),
        "date_end": str(pd.to_datetime(combined["date"]).max().date()),
        "estimator": "abs(log(high/low))/sqrt(4*log(2))",
        "transform": "natural log after dropping zero or invalid ranges",
        "inputs": inputs,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BuildResult(
        output_path=output_path,
        manifest_path=manifest_path,
        rows=int(len(combined)),
        indices=int(combined["index"].nunique()),
        date_start=manifest["date_start"],
        date_end=manifest["date_end"],
    )
