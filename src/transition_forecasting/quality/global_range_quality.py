from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from transition_forecasting.quality.ohlc_range_quality import annual_range_quality


def audit_global_range_quality(
    inventory_path: Path,
    minimum_nonzero_fraction: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, object]]:
    inventory = pd.read_csv(inventory_path)
    eligible = inventory[inventory["eligible"].astype(bool)].copy()
    eligible = eligible[~eligible["index"].isin(["all_indices_data", "^VIX_data"])]

    rows: list[dict[str, object]] = []
    for _, item in eligible.iterrows():
        path = Path(str(item["path"]))
        annual, summary = annual_range_quality(path, minimum_nonzero_fraction)
        rows.append(
            {
                "index": str(item["index"]),
                "path": str(path),
                "nominal_start": str(item["start"]),
                "nominal_end": str(item["end"]),
                "first_nonzero_range_date": summary["first_nonzero_range_date"],
                "first_qualifying_year": summary["first_qualifying_year"],
                "recommended_effective_start": summary["recommended_effective_start"],
                "total_rows": summary["total_rows"],
                "total_nonzero_range_rows": summary["total_nonzero_range_rows"],
                "total_zero_range_rows": summary["total_zero_range_rows"],
                "nonzero_range_fraction": (
                    summary["total_nonzero_range_rows"] / summary["total_rows"]
                    if summary["total_rows"]
                    else 0.0
                ),
                "longest_consecutive_zero_range_run": summary[
                    "longest_consecutive_zero_range_run"
                ],
                "annual_rows": int(len(annual)),
            }
        )

    report = pd.DataFrame(rows).sort_values(["recommended_effective_start", "index"])
    summary = {
        "indices_audited": int(len(report)),
        "minimum_nonzero_fraction": minimum_nonzero_fraction,
        "indices_with_delayed_effective_start": int(
            (
                pd.to_datetime(report["recommended_effective_start"], errors="coerce")
                > pd.to_datetime(report["nominal_start"], errors="coerce")
            ).sum()
        ),
        "indices_without_qualifying_year": int(report["first_qualifying_year"].isna().sum()),
        "earliest_effective_start": str(
            pd.to_datetime(report["recommended_effective_start"], errors="coerce").min().date()
        ),
        "latest_effective_start": str(
            pd.to_datetime(report["recommended_effective_start"], errors="coerce").max().date()
        ),
    }
    return report, summary


def write_global_range_quality(
    inventory_path: Path,
    run_dir: Path,
    minimum_nonzero_fraction: float = 0.95,
) -> dict[str, object]:
    report, summary = audit_global_range_quality(
        inventory_path,
        minimum_nonzero_fraction=minimum_nonzero_fraction,
    )
    report.to_csv(run_dir / "global_range_quality.csv", index=False)
    (run_dir / "global_range_quality_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
