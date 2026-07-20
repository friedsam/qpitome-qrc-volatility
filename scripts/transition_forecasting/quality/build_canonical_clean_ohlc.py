from __future__ import annotations

import argparse
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ohlc(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        reasons = reasons.mask((reasons == "") & work[column].isna(), f"missing_or_nonnumeric_{column}")
        reasons = reasons.mask((reasons == "") & work[column].le(0), f"nonpositive_{column}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the single canonical cleaned OHLC dataset used by all downstream transition workflows."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/transition_forecasting/canonical_ohlc"),
    )
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.force:
            raise SystemExit(f"{args.output_root} already exists; remove it or rerun with --force")
        shutil.rmtree(args.output_root)
    data_root = args.output_root / "individual_indices_data"
    data_root.mkdir(parents=True, exist_ok=False)

    policy = StructuralBadPrintPolicy()
    correction_rows: list[pd.DataFrame] = []
    structural_rows: list[pd.DataFrame] = []
    file_records: list[dict[str, object]] = []

    source_files = sorted(args.raw_root.glob("*.csv"))
    if not source_files:
        raise RuntimeError(f"No CSV files found under {args.raw_root}")

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
        preferred = [column for column in ("date", "open", "high", "low", "close", "volume") if column in cleaned]
        remaining = [column for column in cleaned.columns if column not in preferred]
        cleaned = cleaned[preferred + remaining]
        destination = data_root / source.name
        cleaned.to_csv(destination, index=False, date_format="%Y-%m-%d")

        file_records.append(
            {
                "index": source.stem,
                "source_path": str(source),
                "source_sha256": sha256(source),
                "output_path": str(destination),
                "output_sha256": sha256(destination),
                "source_rows": int(len(pd.read_csv(source))),
                "basic_removed_rows": int(len(basic_corrections)),
                "structural_removed_rows": int(len(flagged)),
                "final_rows": int(len(cleaned)),
            }
        )

    basic = pd.concat(correction_rows, ignore_index=True, sort=False) if correction_rows else pd.DataFrame()
    structural = pd.concat(structural_rows, ignore_index=True, sort=False) if structural_rows else pd.DataFrame()
    affected_indices = int(structural["index"].nunique()) if len(structural) else 0
    if len(structural) != args.expected_structural_flags or affected_indices != args.expected_affected_indices:
        raise RuntimeError(
            "Structural correction/source-lineage mismatch: "
            f"flagged_rows={len(structural)} expected={args.expected_structural_flags}; "
            f"affected_indices={affected_indices} expected={args.expected_affected_indices}"
        )

    all_corrections = pd.concat([basic, structural], ignore_index=True, sort=False)
    all_corrections.to_csv(args.output_root / "row_corrections.csv", index=False)
    pd.DataFrame(file_records).to_csv(args.output_root / "file_manifest.csv", index=False)

    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
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
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
