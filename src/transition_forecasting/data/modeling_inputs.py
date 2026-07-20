from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def materialize_gpt2_modeling_inputs(
    individual_raw: Path,
    corrections_path: Path,
    output_root: Path,
    *,
    apply_structural_corrections: bool,
) -> Path:
    """Create modeling inputs that preserve GPT-2 raw semantics.

    Parity mode copies the exact raw files byte-for-byte. Corrected mode removes
    only the frozen structural-bad-print rows from otherwise unchanged tables.
    It does not apply the broader canonical-cleaning parser used for the public
    cleaned OHLC artifact.
    """
    individual_raw = Path(individual_raw)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    corrections = pd.read_csv(corrections_path)
    structural = corrections.loc[
        corrections.get("reason", pd.Series(dtype=str)).astype(str).eq("structural_bad_print")
    ].copy()
    if not structural.empty:
        structural["date"] = pd.to_datetime(structural["date"], errors="raise").dt.normalize()

    dates_by_index = {
        str(index): set(group["date"])
        for index, group in structural.groupby("index")
    }

    for source in sorted(individual_raw.glob("*.csv")):
        destination = output_root / source.name
        if not apply_structural_corrections:
            shutil.copy2(source, destination)
            continue

        flagged_dates = dates_by_index.get(source.stem, set())
        if not flagged_dates:
            shutil.copy2(source, destination)
            continue

        # Read all fields as strings so numeric values are not rounded or
        # normalized. Only date membership controls row removal.
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        normalized = {str(column).strip().lower(): column for column in frame.columns}
        if "date" not in normalized:
            raise ValueError(f"{source}: missing date column")
        parsed_dates = pd.to_datetime(frame[normalized["date"]], errors="coerce").dt.normalize()
        drop_mask = parsed_dates.isin(flagged_dates)
        dropped = int(drop_mask.sum())
        expected = len(flagged_dates)
        if dropped != expected:
            raise RuntimeError(
                f"{source}: correction-row mismatch; dropped={dropped} expected={expected}"
            )
        frame.loc[~drop_mask].to_csv(destination, index=False)

    return output_root
