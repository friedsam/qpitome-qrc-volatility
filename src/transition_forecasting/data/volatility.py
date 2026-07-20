from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import _load_generic_ohlc
from transition_forecasting.catalogue.transition_events import log_parkinson


def consolidate_cleaned_ohlc(cleaned_root: Path, output_path: Path) -> pd.DataFrame:
    """Combine per-index cleaned OHLC files into one canonical ordered table."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(cleaned_root).glob("*.csv")):
        frame = pd.read_csv(path)
        frame = frame.dropna(axis=1, how="all")
        frame.insert(0, "index", path.stem)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No cleaned OHLC files found under {cleaned_root}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined = combined.sort_values(["index", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, compression="gzip", date_format="%Y-%m-%d")
    return combined


def build_daily_volatility(
    range_quality_path: Path,
    output_path: Path,
    *,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Build ordered daily log Parkinson volatility from frozen effective starts.

    ``range_quality_path`` remains the untouched-raw GPT-2 contract. When
    ``data_root`` is supplied, only the file locations are redirected to the
    correction-only modeling inputs; eligibility and effective starts are not
    recomputed from those transformed files.
    """
    quality = pd.read_csv(range_quality_path)
    required = {"path", "index", "recommended_effective_start"}
    missing = required.difference(quality.columns)
    if missing:
        raise ValueError(f"{range_quality_path}: missing columns {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for _, item in quality.iterrows():
        source_path = Path(str(item["path"]))
        path = Path(data_root) / source_path.name if data_root is not None else source_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing volatility input: {path}")
        index_name = str(item["index"])
        effective_start = pd.Timestamp(item["recommended_effective_start"])
        frame = _load_generic_ohlc(path)
        series = log_parkinson(frame[frame.index >= effective_start])
        if series.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "index": index_name,
                    "date": series.index,
                    "log_parkinson_volatility": series.to_numpy(dtype=float),
                    "effective_start": effective_start,
                }
            )
        )

    if not rows:
        raise RuntimeError("No eligible daily volatility series were produced")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["index", "date"]).reset_index(drop=True)
    if combined.duplicated(["index", "date"]).any():
        raise RuntimeError("Duplicate index/date rows in daily volatility output")
    if not np.isfinite(combined["log_parkinson_volatility"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite daily volatility values found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, compression="gzip", date_format="%Y-%m-%d")
    return combined
