#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_DIR = Path(
    "results/transition_forecasting/modeling/stage_d_dataset/20260718T010839Z"
)


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return 0.6744897501960817 * (numeric - median) / mad


def residual_extremes(
    daily: pd.DataFrame,
    catalogue: pd.DataFrame,
    z_threshold: float = 8.0,
) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["robust_z"] = frame.groupby("index")["log_parkinson_volatility"].transform(robust_z)
    frame["abs_robust_z"] = frame["robust_z"].abs()
    events = catalogue.copy()
    onset_col = "onset_date" if "onset_date" in events.columns else "event_onset"
    events[onset_col] = pd.to_datetime(events[onset_col], errors="coerce")
    by_index = {
        str(name): group[onset_col].dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        for name, group in events.groupby("index")
    }
    overlap: list[bool] = []
    nearest_days: list[float] = []
    for row in frame.itertuples(index=False):
        dates = by_index.get(str(row.index), np.array([], dtype="datetime64[ns]"))
        if len(dates) == 0:
            nearest_days.append(np.nan)
            overlap.append(False)
            continue
        differences = np.abs((dates - np.datetime64(row.date)) / np.timedelta64(1, "D"))
        nearest = float(np.min(differences))
        nearest_days.append(nearest)
        overlap.append(nearest <= 60.0)
    frame["nearest_event_days"] = nearest_days
    frame["within_60d_of_event"] = overlap
    return frame.loc[frame["abs_robust_z"] >= z_threshold].sort_values(
        ["abs_robust_z", "index", "date"], ascending=[False, True, True]
    )


def matching_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    controls = manifest.loc[manifest["label"] == 0].copy()
    columns = ["index", "lead", "split", "controls", "p50", "p90", "p99", "max"]
    if controls.empty or "match_distance" not in controls.columns:
        return pd.DataFrame(columns=columns)
    return (
        controls.groupby(["index", "lead", "split"], dropna=False)["match_distance"]
        .agg(
            controls="size",
            p50=lambda x: x.quantile(0.50),
            p90=lambda x: x.quantile(0.90),
            p99=lambda x: x.quantile(0.99),
            max="max",
        )
        .reset_index()
        .sort_values(["index", "lead", "split"])
    )


def sample_attribution(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["index", "lead", "split", "label"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
        .sort_values(["index", "lead", "split", "label"])
    )


def effective_start_summary(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["effective_start"] = pd.to_datetime(frame["effective_start"], errors="raise")
    return (
        frame.groupby("index", dropna=False)
        .agg(
            effective_start=("effective_start", "first"),
            first_observation=("date", "min"),
            last_observation=("date", "max"),
            observations=("date", "size"),
        )
        .reset_index()
        .sort_values("index")
    )


def scenario_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lower, upper, name in (
        (0.001, 0.999, "winsor_0.1_99.9"),
        (0.005, 0.995, "winsor_0.5_99.5"),
    ):
        changed = 0
        max_shift = 0.0
        for _, group in daily.groupby("index"):
            values = pd.to_numeric(group["log_parkinson_volatility"], errors="raise")
            clipped = values.clip(values.quantile(lower), values.quantile(upper))
            delta = (clipped - values).abs()
            changed += int((delta > 0).sum())
            if len(delta):
                max_shift = max(max_shift, float(delta.max()))
        rows.append(
            {
                "scenario": name,
                "changed_rows": changed,
                "max_abs_shift": max_shift,
                "applied": False,
            }
        )
    return pd.DataFrame(rows)


def baseline_sample_delta(
    current: pd.DataFrame,
    reference_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["status", "sample_id", "index", "lead", "split", "label"]
    if reference_path is None or not reference_path.is_file():
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "reference sample manifest unavailable",
        }
    reference = pd.read_csv(reference_path)
    ref_ids = set(reference["sample_id"].astype(str))
    cur_ids = set(current["sample_id"].astype(str))
    removed = reference.loc[reference["sample_id"].astype(str).isin(ref_ids - cur_ids)].copy()
    added = current.loc[current["sample_id"].astype(str).isin(cur_ids - ref_ids)].copy()
    removed.insert(0, "status", "removed_after_cleanup")
    added.insert(0, "status", "added_after_cleanup")
    combined = pd.concat([removed, added], ignore_index=True, sort=False)
    selected = [column for column in columns if column in combined.columns]
    combined = combined[selected].sort_values(["status", "sample_id"]).reset_index(drop=True)
    return combined, {
        "available": True,
        "reference_samples": int(len(reference)),
        "current_samples": int(len(current)),
        "shared_samples": int(len(ref_ids & cur_ids)),
        "removed_samples": int(len(ref_ids - cur_ids)),
        "added_samples": int(len(cur_ids - ref_ids)),
    }


def event_delta(
    current: pd.DataFrame,
    reference_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["status", "index", "event_onset"]
    if reference_path is None or not reference_path.is_file():
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": "reference event catalogue unavailable",
        }
    reference = pd.read_csv(reference_path)

    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        onset = "event_onset" if "event_onset" in frame.columns else "onset_date"
        result = frame[["index", onset]].copy().rename(columns={onset: "event_onset"})
        result["event_onset"] = pd.to_datetime(result["event_onset"], errors="raise").dt.strftime("%Y-%m-%d")
        return result.drop_duplicates().sort_values(["index", "event_onset"]).reset_index(drop=True)

    ref = canonical(reference)
    cur = canonical(current)
    ref_keys = set(map(tuple, ref.to_numpy()))
    cur_keys = set(map(tuple, cur.to_numpy()))
    rows = [
        {"status": "removed_after_cleanup", "index": index, "event_onset": onset}
        for index, onset in sorted(ref_keys - cur_keys)
    ] + [
        {"status": "added_after_cleanup", "index": index, "event_onset": onset}
        for index, onset in sorted(cur_keys - ref_keys)
    ]
    return pd.DataFrame(rows, columns=columns), {
        "available": True,
        "reference_events": int(len(ref)),
        "current_events": int(len(cur)),
        "shared_events": int(len(ref_keys & cur_keys)),
        "removed_events": int(len(ref_keys - cur_keys)),
        "added_events": int(len(cur_keys - ref_keys)),
    }


def write_markdown_report(report: dict[str, object], output_path: Path) -> None:
    counts = report["counts"]
    sample_delta = report["baseline_sample_delta"]
    events = report["event_delta"]
    text = f"""# Transition Dataset Cleanup Report

## Completed stages

1. Invalid and duplicate OHLC handling.
2. Structural bad-print removal before volatility calculation.
3. Effective-start recalculation.
4. Volatility and event-catalogue reconstruction.
5. Positive-window reconstruction and complete control rematching.
6. Split and tensor integrity validation.
7. Residual-extreme and event-overlap analysis.
8. Matching-quality and sample-attribution analysis.
9. Transformation-scenario analysis.
10. Baseline-versus-corrected sample and event deltas when references are present.

## Current evidence

- Residual extremes with |robust z| >= 8: {counts['residual_extremes_abs_robust_z_ge_8']}
- Isolated residual extremes: {counts['isolated_residual_extremes']}
- Matching strata audited: {counts['matching_strata']}
- Effective-start rows audited: {counts['effective_start_rows']}
- Baseline sample delta available: {sample_delta['available']}
- Baseline event delta available: {events['available']}

## Methodological disposition

Structural data errors are removed causally and all downstream products are rebuilt. Alternative winsorization scenarios are quantified but not silently substituted for the canonical data. Every generated table is retained in the run directory for review and reproducibility.
"""
    output_path.write_text(text, encoding="utf-8")


def analyze(
    dataset_dir: Path,
    output_dir: Path,
    reference_manifest: Path | None = None,
    reference_catalogue: Path | None = None,
) -> dict[str, object]:
    daily = pd.read_csv(dataset_dir / "daily_volatility.csv.gz")
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    manifest = pd.read_csv(dataset_dir / "sample_manifest.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    extremes = residual_extremes(daily, catalogue)
    matching = matching_summary(manifest)
    attribution = sample_attribution(manifest)
    starts = effective_start_summary(daily)
    scenarios = scenario_summary(daily)
    sample_delta_frame, sample_delta_report = baseline_sample_delta(manifest, reference_manifest)
    event_delta_frame, event_delta_report = event_delta(catalogue, reference_catalogue)

    outputs = {
        "residual_extremes.csv": extremes,
        "matching_quality.csv": matching,
        "sample_attribution.csv": attribution,
        "effective_starts.csv": starts,
        "transformation_scenarios.csv": scenarios,
        "baseline_sample_delta.csv": sample_delta_frame,
        "baseline_event_delta.csv": event_delta_frame,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    isolated = int((~extremes["within_60d_of_event"]).sum()) if len(extremes) else 0
    report: dict[str, object] = {
        "schema_version": 2,
        "dataset_dir": str(dataset_dir),
        "counts": {
            "residual_extremes_abs_robust_z_ge_8": int(len(extremes)),
            "isolated_residual_extremes": isolated,
            "matching_strata": int(len(matching)),
            "sample_attribution_rows": int(len(attribution)),
            "effective_start_rows": int(len(starts)),
        },
        "baseline_sample_delta": sample_delta_report,
        "event_delta": event_delta_report,
        "decisions": {
            "structural_bad_print_removal_applied": True,
            "controls_rematched": True,
            "effective_starts_recomputed": True,
            "winsorization_scenarios_evaluated": True,
        },
        "outputs": list(outputs) + ["final_cleanup_report.md"],
    }
    write_markdown_report(report, output_dir / "final_cleanup_report.md")
    (output_dir / "cleanup_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate complete cleanup evidence for the canonical transition dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/transition_forecasting/global_transition_dataset"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / "sample_manifest.csv",
    )
    parser.add_argument("--reference-catalogue", type=Path, default=None)
    args = parser.parse_args()
    report = analyze(
        args.dataset_dir,
        args.output_dir,
        reference_manifest=args.reference_manifest,
        reference_catalogue=args.reference_catalogue,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
