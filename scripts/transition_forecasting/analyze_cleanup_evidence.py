#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return 0.6744897501960817 * (numeric - median) / mad


def residual_extremes(daily: pd.DataFrame, catalogue: pd.DataFrame, z_threshold: float = 8.0) -> pd.DataFrame:
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
    event_overlap = []
    nearest_days = []
    for row in frame.itertuples(index=False):
        dates = by_index.get(str(row.index), np.array([], dtype="datetime64[ns]"))
        if len(dates) == 0:
            nearest_days.append(np.nan)
            event_overlap.append(False)
            continue
        differences = np.abs((dates - np.datetime64(row.date)) / np.timedelta64(1, "D"))
        nearest = float(np.min(differences))
        nearest_days.append(nearest)
        event_overlap.append(nearest <= 60.0)
    frame["nearest_event_days"] = nearest_days
    frame["within_60d_of_event"] = event_overlap
    return frame.loc[frame["abs_robust_z"] >= z_threshold].sort_values(
        ["abs_robust_z", "index", "date"], ascending=[False, True, True]
    )


def matching_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    controls = manifest.loc[manifest["label"] == 0].copy()
    if controls.empty or "match_distance" not in controls.columns:
        return pd.DataFrame(columns=["index", "lead", "split", "controls", "p50", "p90", "p99", "max"])
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
    rows = []
    for lower, upper, name in ((0.001, 0.999, "winsor_0.1_99.9"), (0.005, 0.995, "winsor_0.5_99.5")):
        changed = 0
        max_shift = 0.0
        for _, group in daily.groupby("index"):
            values = pd.to_numeric(group["log_parkinson_volatility"], errors="raise")
            lo = values.quantile(lower)
            hi = values.quantile(upper)
            clipped = values.clip(lo, hi)
            delta = (clipped - values).abs()
            changed += int((delta > 0).sum())
            if len(delta):
                max_shift = max(max_shift, float(delta.max()))
        rows.append({"scenario": name, "changed_rows": changed, "max_abs_shift": max_shift, "applied": False})
    return pd.DataFrame(rows)


def analyze(dataset_dir: Path, output_dir: Path) -> dict[str, object]:
    daily = pd.read_csv(dataset_dir / "daily_volatility.csv.gz")
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    manifest = pd.read_csv(dataset_dir / "sample_manifest.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    extremes = residual_extremes(daily, catalogue)
    matching = matching_summary(manifest)
    attribution = sample_attribution(manifest)
    starts = effective_start_summary(daily)
    scenarios = scenario_summary(daily)

    extremes.to_csv(output_dir / "residual_extremes.csv", index=False)
    matching.to_csv(output_dir / "matching_quality.csv", index=False)
    attribution.to_csv(output_dir / "sample_attribution.csv", index=False)
    starts.to_csv(output_dir / "effective_starts.csv", index=False)
    scenarios.to_csv(output_dir / "transformation_scenarios.csv", index=False)

    isolated_extremes = int((~extremes["within_60d_of_event"]).sum()) if len(extremes) else 0
    report = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "counts": {
            "residual_extremes_abs_robust_z_ge_8": int(len(extremes)),
            "isolated_residual_extremes": isolated_extremes,
            "matching_strata": int(len(matching)),
            "sample_attribution_rows": int(len(attribution)),
            "effective_start_rows": int(len(starts)),
        },
        "decisions": {
            "winsorization_applied": False,
            "clipping_applied": False,
            "additional_automatic_row_removal_applied": False,
            "reason": "Residual extremes are documented and scenario-tested; no non-causal transformation is applied automatically.",
        },
        "outputs": [
            "residual_extremes.csv",
            "matching_quality.csv",
            "sample_attribution.csv",
            "effective_starts.csv",
            "transformation_scenarios.csv",
        ],
    }
    (output_dir / "cleanup_evidence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate complete cleanup evidence for the canonical transition dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/transition_forecasting/global_transition_dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.dataset_dir, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
