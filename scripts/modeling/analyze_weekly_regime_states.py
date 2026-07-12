#!/usr/bin/env python3
"""Interpret filtered weekly HMM states without refitting.

Uses the completed one-step prediction export to summarize each model's
causal filtered states by return sign, magnitude, volatility proxy, occupancy,
persistence, and dominant-state transitions.  This is an exploratory
interpretability check, not a test of parameter stability across refits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "hmm2": 2,
    "hmm4_unrestricted": 4,
    "hmm4_restricted": 4,
}


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    required = {"date", "return_pct"}
    for model, n_states in MODELS.items():
        required.update(f"{model}_filtered_state_{i}" for i in range(n_states))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["abs_return_pct"] = frame["return_pct"].abs()
    frame["rolling_vol_13w"] = frame["return_pct"].rolling(13, min_periods=6).std()
    return frame


def state_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, n_states in MODELS.items():
        probs = frame[[f"{model}_filtered_state_{i}" for i in range(n_states)]].to_numpy(float)
        dominant = probs.argmax(axis=1)
        for state in range(n_states):
            weight = probs[:, state]
            total = float(weight.sum())
            hard = dominant == state
            rows.append({
                "model": model,
                "state": state,
                "soft_occupancy": total / len(frame),
                "hard_occupancy": float(hard.mean()),
                "weighted_mean_return_pct": float(np.average(frame["return_pct"], weights=weight)),
                "weighted_positive_fraction": float(np.average((frame["return_pct"] > 0).astype(float), weights=weight)),
                "weighted_mean_abs_return_pct": float(np.average(frame["abs_return_pct"], weights=weight)),
                "weighted_mean_rolling_vol_13w": float(np.average(frame["rolling_vol_13w"].fillna(frame["rolling_vol_13w"].median()), weights=weight)),
                "hard_n": int(hard.sum()),
                "hard_mean_return_pct": float(frame.loc[hard, "return_pct"].mean()) if hard.any() else np.nan,
                "hard_return_std_pct": float(frame.loc[hard, "return_pct"].std()) if hard.sum() > 1 else np.nan,
                "hard_positive_fraction": float((frame.loc[hard, "return_pct"] > 0).mean()) if hard.any() else np.nan,
            })
    return pd.DataFrame(rows)


def transition_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model, n_states in MODELS.items():
        probs = frame[[f"{model}_filtered_state_{i}" for i in range(n_states)]].to_numpy(float)
        dominant = probs.argmax(axis=1)
        counts = np.zeros((n_states, n_states), dtype=int)
        for a, b in zip(dominant[:-1], dominant[1:]):
            counts[a, b] += 1
        for i in range(n_states):
            row_total = counts[i].sum()
            for j in range(n_states):
                count_rows.append({
                    "model": model,
                    "from_state": i,
                    "to_state": j,
                    "count": int(counts[i, j]),
                    "row_fraction": float(counts[i, j] / row_total) if row_total else np.nan,
                })
            runs = []
            current = 0
            for value in dominant:
                if value == i:
                    current += 1
                elif current:
                    runs.append(current)
                    current = 0
            if current:
                runs.append(current)
            summary_rows.append({
                "model": model,
                "state": i,
                "self_transition_fraction": float(counts[i, i] / row_total) if row_total else np.nan,
                "n_runs": int(len(runs)),
                "median_run_weeks": float(np.median(runs)) if runs else np.nan,
                "max_run_weeks": int(max(runs)) if runs else 0,
            })
    return pd.DataFrame(count_rows), pd.DataFrame(summary_rows)


def decade_occupancy(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["decade"] = (work["date"].dt.year // 10 * 10).astype(int)
    rows: list[dict] = []
    for model, n_states in MODELS.items():
        cols = [f"{model}_filtered_state_{i}" for i in range(n_states)]
        for decade, part in work.groupby("decade"):
            dominant = part[cols].to_numpy(float).argmax(axis=1)
            for state in range(n_states):
                rows.append({
                    "model": model,
                    "decade": int(decade),
                    "state": state,
                    "soft_occupancy": float(part[cols[state]].mean()),
                    "hard_occupancy": float(np.mean(dominant == state)),
                })
    return pd.DataFrame(rows)


def sanity_checks(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, n_states in MODELS.items():
        probs = frame[[f"{model}_filtered_state_{i}" for i in range(n_states)]].to_numpy(float)
        row_sums = probs.sum(axis=1)
        rows.append({
            "model": model,
            "n_rows": int(len(frame)),
            "minimum_probability": float(probs.min()),
            "maximum_probability": float(probs.max()),
            "maximum_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
            "mean_max_probability": float(np.mean(probs.max(axis=1))),
            "fraction_max_probability_above_0_8": float(np.mean(probs.max(axis=1) > 0.8)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("/tmp/weekly_regime_baselines/one_step_predictions.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/tmp/weekly_regime_state_diagnostics"),
    )
    args = parser.parse_args()

    frame = load_predictions(args.predictions)
    profiles = state_profiles(frame)
    transitions, persistence = transition_tables(frame)
    occupancy = decade_occupancy(frame)
    checks = sanity_checks(frame)

    args.outdir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.outdir / "state_profiles.csv", index=False)
    transitions.to_csv(args.outdir / "dominant_state_transitions.csv", index=False)
    persistence.to_csv(args.outdir / "state_persistence.csv", index=False)
    occupancy.to_csv(args.outdir / "state_occupancy_by_decade.csv", index=False)
    checks.to_csv(args.outdir / "sanity_checks.csv", index=False)

    print("Probability sanity checks")
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nState profiles")
    print(
        profiles[[
            "model",
            "state",
            "soft_occupancy",
            "weighted_mean_return_pct",
            "weighted_positive_fraction",
            "weighted_mean_abs_return_pct",
            "weighted_mean_rolling_vol_13w",
            "hard_mean_return_pct",
            "hard_return_std_pct",
        ]].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print("\nDominant-state persistence")
    print(persistence.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
