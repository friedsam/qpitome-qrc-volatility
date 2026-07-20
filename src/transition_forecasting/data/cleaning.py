from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from transition_forecasting.quality.structural_bad_prints import (
    StructuralBadPrintPolicy,
    flag_structural_bad_prints,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ohlc(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one raw OHLC file and return canonical rows plus a removal ledger."""
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    work = raw.copy()
    work.insert(0, "source_row", range(len(work)))
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ("open", "high", "low", "close"):
        work[column] = pd.to_numeric(
            work[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    reasons = pd.Series("", index=work.index, dtype=object)
    reasons = reasons.mask(work["date"].isna(), "invalid_date")
    for column in ("open", "high", "low", "close"):
        reasons = reasons.mask(
            (reasons == "") & work[column].isna(),
            f"missing_or_nonnumeric_{column}",
        )
        reasons = reasons.mask(
            (reasons == "") & work[column].le(0),
            f"nonpositive_{column}",
        )
    reasons = reasons.mask((reasons == "") & work["high"].lt(work["low"]), "high_below_low")

    corrections = work.loc[reasons.ne("")].copy()
    corrections["reason"] = reasons.loc[reasons.ne("")]
    corrections["action"] = "drop"

    valid = work.loc[reasons.eq("")].sort_values(["date", "source_row"])
    duplicate_mask = valid.duplicated("date", keep="last")
    duplicates = valid.loc[duplicate_mask].copy()
    if len(duplicates):
        duplicates["reason"] = "duplicate_date_keep_last"
        duplicates["action"] = "drop"
        corrections = pd.concat([corrections, duplicates], ignore_index=True, sort=False)

    canonical = valid.loc[~duplicate_mask].copy().reset_index(drop=True)
    return canonical, corrections


def build_cleaned_ohlc(
    raw_root: Path,
    output_root: Path,
    *,
    expected_structural_flags: int = 12,
    expected_affected_indices: int = 2,
    force: bool = False,
    policy: StructuralBadPrintPolicy = StructuralBadPrintPolicy(),
) -> dict[str, object]:
    """Build cleaned individual-index OHLC files and complete correction provenance."""
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists")
        shutil.rmtree(output_root)

    source_files = sorted(raw_root.glob("*.csv"))
    if not source_files:
        raise RuntimeError(f"No CSV files found under {raw_root}")

    data_root = output_root / "individual_indices_data"
    data_root.mkdir(parents=True, exist_ok=False)

    correction_rows: list[pd.DataFrame] = []
    structural_rows: list[pd.DataFrame] = []
    file_records: list[dict[str, object]] = []

    for source in source_files:
        canonical, basic_corrections = parse_ohlc(source)
        basic_corrections.insert(0, "index", source.stem)
        basic_corrections.insert(1, "source_path", str(source))
        correction_rows.append(basic_corrections)

        audited = flag_structural_bad_prints(canonical, index_name=source.stem, policy=policy)
        flagged = audited.loc[audited["suspected_bad_print"]].copy()
        if len(flagged):
            flagged.insert(1, "source_path", str(source))
            flagged["reason"] = "structural_bad_print"
            flagged["action"] = "drop"
            structural_rows.append(flagged)

        flagged_dates = set(pd.to_datetime(flagged["date"])) if len(flagged) else set()
        cleaned = canonical.loc[~canonical["date"].isin(flagged_dates)].copy()
        cleaned = cleaned.drop(columns=["source_row"], errors="ignore")
        preferred = [
            column
            for column in ("date", "open", "high", "low", "close", "volume")
            if column in cleaned
        ]
        remaining = [column for column in cleaned.columns if column not in preferred]
        cleaned = cleaned[preferred + remaining]

        destination = data_root / source.name
        cleaned.to_csv(destination, index=False, date_format="%Y-%m-%d")

        file_records.append(
            {
                "index": source.stem,
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "output_path": str(destination),
                "output_sha256": sha256_file(destination),
                "source_rows": int(len(pd.read_csv(source))),
                "basic_removed_rows": int(len(basic_corrections)),
                "structural_removed_rows": int(len(flagged)),
                "final_rows": int(len(cleaned)),
            }
        )

    basic = (
        pd.concat(correction_rows, ignore_index=True, sort=False)
        if correction_rows
        else pd.DataFrame()
    )
    structural = (
        pd.concat(structural_rows, ignore_index=True, sort=False)
        if structural_rows
        else pd.DataFrame()
    )
    affected_indices = int(structural["index"].nunique()) if len(structural) else 0

    if (
        len(structural) != expected_structural_flags
        or affected_indices != expected_affected_indices
    ):
        raise RuntimeError(
            "Structural correction/source-lineage mismatch: "
            f"flagged_rows={len(structural)} expected={expected_structural_flags}; "
            f"affected_indices={affected_indices} expected={expected_affected_indices}"
        )

    all_corrections = pd.concat([basic, structural], ignore_index=True, sort=False)
    all_corrections.to_csv(output_root / "row_corrections.csv", index=False)
    pd.DataFrame(file_records).to_csv(output_root / "file_manifest.csv", index=False)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "policy": policy.to_dict(),
        "source_files": len(source_files),
        "basic_removed_rows": int(len(basic)),
        "structural_removed_rows": int(len(structural)),
        "affected_indices": affected_indices,
        "final_rows": int(sum(int(record["final_rows"]) for record in file_records)),
        "rules": [
            "parse dates and OHLC numerics",
            "drop invalid dates",
            "drop missing, nonnumeric, zero, or negative OHLC rows",
            "drop rows with high below low",
            "sort by date and keep the last duplicate-date row",
            "drop rows flagged by the frozen causal structural bad-print policy",
            "do not interpolate, forward fill, winsorize, clip, or synthesize dates",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
