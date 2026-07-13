#!/usr/bin/env python3
"""Cheap diagnostics for completed weekly regime-baseline predictions.

This script requires only ``one_step_predictions.csv`` from a completed run.
It decomposes predictive-density gains by decade, return-tail bucket, and a
small set of named crisis windows. No HMM refitting is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPARISONS = {
    "restricted_minus_unrestricted": (
        "hmm4_restricted_log_density",
        "hmm4_unrestricted_log_density",
    ),
    "restricted_minus_hmm2": (
        "hmm4_restricted_log_density",
        "hmm2_log_density",
    ),
    "restricted_minus_iid": (
        "hmm4_restricted_log_density",
        "iid_gaussian_log_density",
    ),
}

CRISIS_WINDOWS = {
    "1973_1974": ("1973-01-01", "1974-12-31"),
    "1987_crash": ("1987-08-01", "1988-03-31"),
    "dotcom_2000_2002": ("2000-03-01", "2002-12-31"),
    "gfc_2007_2009": ("2007-07-01", "2009-06-30"),
    "covid_2020": ("2020-02-01", "2020-08-31"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
}


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "return_pct"}
    for pair in COMPARISONS.values():
        required.update(pair)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    for name, (lhs, rhs) in COMPARISONS.items():
        frame[name] = frame[lhs].to_numpy(float) - frame[rhs].to_numpy(float)
    frame["decade"] = (frame["date"].dt.year // 10 * 10).astype(int)
    return frame


def summarize_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group, part in frame.groupby(group_col, sort=True, dropna=False):
        row: dict = {group_col: group, "n": int(len(part))}
        for comparison in COMPARISONS:
            values = part[comparison].to_numpy(float)
            row[f"{comparison}_mean"] = float(np.mean(values))
            row[f"{comparison}_total"] = float(np.sum(values))
            row[f"{comparison}_win_fraction"] = float(np.mean(values > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def tail_summary(frame: pd.DataFrame) -> pd.DataFrame:
    abs_return = frame["return_pct"].abs()
    quantiles = abs_return.quantile([0.50, 0.90, 0.95, 0.99]).to_dict()
    labels = pd.Series("central_50pct", index=frame.index, dtype=object)
    labels[abs_return > quantiles[0.50]] = "upper_50pct"
    labels[abs_return > quantiles[0.90]] = "top_10pct"
    labels[abs_return > quantiles[0.95]] = "top_5pct"
    labels[abs_return > quantiles[0.99]] = "top_1pct"
    work = frame.copy()
    work["tail_bucket"] = labels
    order = ["central_50pct", "upper_50pct", "top_10pct", "top_5pct", "top_1pct"]
    result = summarize_group(work, "tail_bucket")
    result["tail_bucket"] = pd.Categorical(result["tail_bucket"], categories=order, ordered=True)
    return result.sort_values("tail_bucket").reset_index(drop=True)


def crisis_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for name, (start, end) in CRISIS_WINDOWS.items():
        use = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        row: dict = {
            "window": name,
            "start": start,
            "end": end,
            "n": int(len(use)),
        }
        for comparison in COMPARISONS:
            values = use[comparison].to_numpy(float)
            row[f"{comparison}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{comparison}_total"] = float(np.sum(values)) if len(values) else np.nan
            row[f"{comparison}_win_fraction"] = float(np.mean(values > 0)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def concentration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison in COMPARISONS:
        values = frame[comparison].to_numpy(float)
        total = float(values.sum())
        ordered = np.sort(values)[::-1]
        n = len(ordered)
        for fraction in [0.01, 0.05, 0.10, 0.25, 0.50]:
            k = max(1, int(np.ceil(fraction * n)))
            contribution = float(ordered[:k].sum())
            rows.append({
                "comparison": comparison,
                "top_fraction": fraction,
                "n_weeks": k,
                "cumulative_gain": contribution,
                "fraction_of_total_gain": contribution / total if total != 0 else np.nan,
            })
    return pd.DataFrame(rows)


def sanity_checks(frame: pd.DataFrame) -> dict:
    checks: dict[str, object] = {
        "n_rows": int(len(frame)),
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
    }
    for comparison in COMPARISONS:
        values = frame[comparison].to_numpy(float)
        checks[f"{comparison}_total"] = float(values.sum())
        checks[f"{comparison}_mean"] = float(values.mean())
        checks[f"{comparison}_positive_weeks"] = int(np.sum(values > 0))
        checks[f"{comparison}_negative_weeks"] = int(np.sum(values < 0))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/modeling/weekly_regimes/weekly_regime_baselines__one_step_predictions.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/weekly_regimes"),
    )
    args = parser.parse_args()

    frame = load_predictions(args.predictions)
    decade = summarize_group(frame, "decade")
    tails = tail_summary(frame)
    crises = crisis_summary(frame)
    concentration = concentration_summary(frame)
    checks = sanity_checks(frame)

    args.outdir.mkdir(parents=True, exist_ok=True)
    decade.to_csv(
        args.outdir / "weekly_regime_baseline_diagnostics__gain_by_decade.csv",
        index=False,
    )
    tails.to_csv(
        args.outdir / "weekly_regime_baseline_diagnostics__gain_by_return_tail.csv",
        index=False,
    )
    crises.to_csv(
        args.outdir / "weekly_regime_baseline_diagnostics__gain_by_crisis_window.csv",
        index=False,
    )
    concentration.to_csv(
        args.outdir / "weekly_regime_baseline_diagnostics__gain_concentration.csv",
        index=False,
    )
    (
        args.outdir / "weekly_regime_baseline_diagnostics__sanity_checks.json"
    ).write_text(json.dumps(checks, indent=2))

    print("Sanity checks")
    print(json.dumps(checks, indent=2))
    print("\nRestricted minus unrestricted by decade")
    print(
        decade[[
            "decade",
            "n",
            "restricted_minus_unrestricted_mean",
            "restricted_minus_unrestricted_total",
            "restricted_minus_unrestricted_win_fraction",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print("\nRestricted minus unrestricted by return-tail bucket")
    print(
        tails[[
            "tail_bucket",
            "n",
            "restricted_minus_unrestricted_mean",
            "restricted_minus_unrestricted_total",
            "restricted_minus_unrestricted_win_fraction",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print("\nNamed crisis windows")
    print(
        crises[[
            "window",
            "n",
            "restricted_minus_unrestricted_total",
            "restricted_minus_hmm2_total",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )


if __name__ == "__main__":
    main()
