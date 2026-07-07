#!/usr/bin/env python3
"""Companion audit for state interpretability and temporal presence.

Run after run_regime_discovery_audit.py. Recomputes the preregistered state
labels exactly, then reports feature profiles and state occupancy by period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_regime_discovery_audit import (
    PERIODS,
    candidate_specs,
    fit_candidate,
    load_data,
    period_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/phase2_spy_vix_volatility.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/diagnostics/regime_discovery"),
    )
    p.add_argument("--seed", type=int, default=20260704)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)

    profile_rows: list[dict] = []
    occupancy_rows: list[dict] = []

    specs = candidate_specs()
    for i, spec in enumerate(specs):
        col = f"state__{spec.name}"
        df[col] = fit_candidate(df, spec, seed=args.seed + i)

        for state in sorted(int(x) for x in df[col].dropna().unique()):
            mask = df[col] == state
            row = {
                "candidate": col,
                "state": state,
                "n_days": int(mask.sum()),
            }
            for feat in spec.features:
                row[f"{feat}_mean"] = float(df.loc[mask, feat].mean())
                row[f"{feat}_median"] = float(df.loc[mask, feat].median())
            profile_rows.append(row)

        for period in PERIODS:
            if period == "full":
                continue
            pmask = period_mask(df, period)
            denom = int((pmask & df[col].notna()).sum())
            for state in sorted(int(x) for x in df[col].dropna().unique()):
                n = int((pmask & (df[col] == state)).sum())
                occupancy_rows.append(
                    {
                        "candidate": col,
                        "period": period,
                        "state": state,
                        "n_days": n,
                        "occupancy": float(n / denom) if denom else float("nan"),
                    }
                )

    profiles = pd.DataFrame(profile_rows)
    occupancy = pd.DataFrame(occupancy_rows)
    profiles.to_csv(args.outdir / "state_feature_profiles.csv", index=False)
    occupancy.to_csv(args.outdir / "period_state_occupancy.csv", index=False)

    presence = (
        occupancy.assign(present=lambda x: x["n_days"] > 0)
        .groupby("candidate", as_index=False)
        .agg(
            min_period_state_occupancy=("occupancy", "min"),
            all_states_present_all_periods=("present", "all"),
        )
    )
    presence.to_csv(args.outdir / "state_temporal_presence.csv", index=False)

    decisions_path = args.outdir / "candidate_decisions.csv"
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        decisions = decisions.merge(presence, on="candidate", how="left")
        missing = ~decisions["all_states_present_all_periods"].fillna(False)
        decisions.loc[missing, "passes_conservative_screen"] = False
        existing = decisions["rejection_reasons"].fillna("").astype(str)
        decisions.loc[missing, "rejection_reasons"] = existing[missing].map(
            lambda s: (s + ";state_absent_in_major_period").strip(";")
        )
        decisions.to_csv(decisions_path, index=False)

    print("\n=== State temporal presence ===")
    print(presence.to_string(index=False))
    print(f"\nWrote state profiles and temporal-presence outputs to {args.outdir}")


if __name__ == "__main__":
    main()
