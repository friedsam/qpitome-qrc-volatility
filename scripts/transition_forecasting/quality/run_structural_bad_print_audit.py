from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from transition_forecasting.quality.structural_bad_prints import (
    StructuralBadPrintPolicy,
    annotate_sample_manifest,
    audit_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structural OHLC bad prints and annotate a sample manifest.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/transition_forecasting/quality/structural_bad_prints"))
    parser.add_argument("--expected-flagged", type=int, default=12)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)

    policy = StructuralBadPrintPolicy()
    full, flagged = audit_directory(args.raw_root, policy=policy)
    if len(flagged) != args.expected_flagged:
        raise RuntimeError(
            f"Structural quality policy produced {len(flagged)} flagged rows; "
            f"expected {args.expected_flagged}. Inspect raw-data lineage before continuing."
        )

    manifest = __import__("pandas").read_csv(args.manifest)
    annotated = annotate_sample_manifest(manifest, flagged)
    contaminated = annotated.loc[annotated["bad_print_any"]].copy()
    clean = annotated.loc[~annotated["bad_print_any"], ["sample_id", "fold", "fold_split"]].copy()

    full.to_csv(outdir / "daily_ohlc_quality_audit.csv.gz", index=False)
    flagged.to_csv(outdir / "frozen_structural_bad_prints.csv", index=False)
    annotated.to_csv(outdir / "annotated_sample_manifest.csv.gz", index=False)
    contaminated.to_csv(outdir / "contaminated_samples.csv", index=False)
    clean.to_csv(outdir / "clean_sample_ids.csv", index=False)

    summary = {
        "raw_root": str(args.raw_root),
        "manifest": str(args.manifest),
        "policy": policy.to_dict(),
        "expected_flagged": args.expected_flagged,
        "flagged_rows": int(len(flagged)),
        "affected_indices": int(flagged["index"].nunique()),
        "manifest_rows": int(len(annotated)),
        "contaminated_rows": int(annotated["bad_print_any"].sum()),
        "contaminated_sample_ids": int(contaminated["sample_id"].nunique()),
        "input_contaminated_rows": int(annotated["bad_print_in_input"].sum()),
        "target_contaminated_rows": int(annotated["bad_print_in_target"].sum()),
        "quality_policy": "exclude a sample if its input or target interval intersects a frozen structural bad-print date",
        "raw_data_modified": False,
        "test_rows_used": 0,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "params.json").write_text(json.dumps({"policy": policy.to_dict()}, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
