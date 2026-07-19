from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from transition_forecasting.quality.structural_bad_prints import (
    StructuralBadPrintPolicy,
    annotate_sample_manifest,
    audit_directory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.csv")):
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit structural OHLC bad prints. Optionally annotate an existing "
            "sample manifest without making the raw-data audit depend on it."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/transition_forecasting/quality/structural_bad_prints"),
    )
    parser.add_argument("--expected-flagged", type=int, default=12)
    parser.add_argument("--expected-indices", type=int, default=2)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)

    policy = StructuralBadPrintPolicy()
    inventory = source_inventory(args.raw_root)
    if not inventory:
        raise RuntimeError(f"No CSV files found under {args.raw_root}")

    full, flagged = audit_directory(args.raw_root, policy=policy)
    affected_indices = int(flagged["index"].nunique()) if len(flagged) else 0
    if len(flagged) != args.expected_flagged or affected_indices != args.expected_indices:
        raise RuntimeError(
            "Structural quality policy/source-lineage mismatch: "
            f"flagged_rows={len(flagged)} (expected {args.expected_flagged}), "
            f"affected_indices={affected_indices} (expected {args.expected_indices}). "
            "Inspect raw-data hashes and policy before continuing."
        )

    full.to_csv(outdir / "daily_ohlc_quality_audit.csv.gz", index=False)
    flagged.to_csv(outdir / "frozen_structural_bad_prints.csv", index=False)
    (outdir / "raw_source_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, object] = {
        "raw_root": str(args.raw_root),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "policy": policy.to_dict(),
        "expected_flagged": args.expected_flagged,
        "expected_indices": args.expected_indices,
        "flagged_rows": int(len(flagged)),
        "affected_indices": affected_indices,
        "raw_source_files": len(inventory),
        "quality_policy": (
            "flag daily rows using the frozen causal structural OHLC rule; "
            "when a manifest is supplied, exclude a sample if its input or target "
            "interval intersects a flagged date"
        ),
        "raw_data_modified": False,
        "test_rows_used": 0,
    }

    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        annotated = annotate_sample_manifest(manifest, flagged)
        contaminated = annotated.loc[annotated["bad_print_any"]].copy()
        clean_columns = [
            column
            for column in ("sample_id", "fold", "fold_split")
            if column in annotated.columns
        ]
        clean = annotated.loc[~annotated["bad_print_any"], clean_columns].copy()

        annotated.to_csv(outdir / "annotated_sample_manifest.csv.gz", index=False)
        contaminated.to_csv(outdir / "contaminated_samples.csv", index=False)
        clean.to_csv(outdir / "clean_sample_ids.csv", index=False)
        summary.update(
            {
                "manifest_sha256": sha256(args.manifest),
                "manifest_rows": int(len(annotated)),
                "contaminated_rows": int(annotated["bad_print_any"].sum()),
                "contaminated_sample_ids": int(contaminated["sample_id"].nunique()),
                "input_contaminated_rows": int(annotated["bad_print_in_input"].sum()),
                "target_contaminated_rows": int(annotated["bad_print_in_target"].sum()),
            }
        )

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (outdir / "params.json").write_text(
        json.dumps(
            {
                "policy": policy.to_dict(),
                "expected_flagged": args.expected_flagged,
                "expected_indices": args.expected_indices,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
