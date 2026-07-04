#!/usr/bin/env python3
"""Falsification-first audit for economically meaningful market regimes.

This script does not optimize a predictive model. It asks whether the committed
SPY/VIX dataset contains persistent market states and sufficiently frequent,
economically distinct transitions to justify a later ESN/QRC forecasting task.

Outputs are written under results/diagnostics/regime_discovery/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


TRAIN_END = pd.Timestamp("2014-12-31")
PERIODS = {
    "1993_2004": (pd.Timestamp("1993-01-01"), pd.Timestamp("2004-12-31")),
    "2005_2014": (pd.Timestamp("2005-01-01"), pd.Timestamp("2014-12-31")),
    "2015_2019": (pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
    "2020_2024": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")),
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
}

OUTCOMES = [
    "fwd_return_5d",
    "fwd_return_10d",
    "fwd_return_20d",
    "fwd_max_drawdown_20d",
    "fwd_max_upside_20d",
    "future_rv_20d",
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    features: tuple[str, ...]
    k: int | None = None


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
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--transition-cooldown", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260704)
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {
        "date",
        "spy_adj_close",
        "rv_20d",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "spy_drawdown_20d",
        "vix_close",
        "vix_log_change",
        "spy_log_hl_range",
        "spy_log_volume_change",
        "future_rv_20d",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return add_forward_outcomes(df)


def add_forward_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = out["spy_adj_close"].to_numpy(dtype=float)
    n = len(out)

    for h in (5, 10, 20):
        arr = np.full(n, np.nan)
        arr[: n - h] = px[h:] / px[: n - h] - 1.0
        out[f"fwd_return_{h}d"] = arr

    fwd_dd = np.full(n, np.nan)
    fwd_up = np.full(n, np.nan)
    for i in range(n - 20):
        path = px[i + 1 : i + 21] / px[i] - 1.0
        fwd_dd[i] = float(np.min(path))
        fwd_up[i] = float(np.max(path))
    out["fwd_max_drawdown_20d"] = fwd_dd
    out["fwd_max_upside_20d"] = fwd_up
    return out


def candidate_specs() -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            name="vol_level_x_acceleration_grid",
            family="quantile_grid",
            features=("rv_20d", "rv_ratio_5_20"),
        )
    ]
    compact = ("rv_20d", "vix_close", "spy_drawdown_20d", "rv_ratio_5_20")
    rich = (
        "rv_20d",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "spy_drawdown_20d",
        "vix_close",
        "vix_log_change",
        "spy_log_hl_range",
        "spy_log_volume_change",
    )
    for k in (2, 3, 4, 5):
        specs.append(CandidateSpec(f"risk4_k{k}", "kmeans", compact, k))
        specs.append(CandidateSpec(f"state8_k{k}", "kmeans", rich, k))
    return specs


def fit_candidate(df: pd.DataFrame, spec: CandidateSpec, seed: int) -> pd.Series:
    train = df["date"] <= TRAIN_END
    X = df.loc[:, list(spec.features)].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    labels = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if spec.family == "quantile_grid":
        q_level = df.loc[train & valid, "rv_20d"].quantile([1 / 3, 2 / 3]).to_numpy()
        q_accel = df.loc[train & valid, "rv_ratio_5_20"].quantile([1 / 3, 2 / 3]).to_numpy()
        level_bin = np.digitize(df.loc[valid, "rv_20d"], q_level)
        accel_bin = np.digitize(df.loc[valid, "rv_ratio_5_20"], q_accel)
        labels.loc[valid] = (level_bin * 3 + accel_bin).astype(int)
        return labels

    scaler = StandardScaler().fit(X.loc[train & valid])
    Xs_train = scaler.transform(X.loc[train & valid])
    km = KMeans(n_clusters=int(spec.k), random_state=seed, n_init=50)
    km.fit(Xs_train)
    labels.loc[valid] = km.predict(scaler.transform(X.loc[valid])).astype(int)
    return labels


def run_lengths(labels: pd.Series) -> pd.DataFrame:
    arr = labels.astype("Int64")
    rows: list[dict] = []
    start = 0
    vals = arr.to_numpy()
    n = len(vals)
    while start < n:
        if pd.isna(vals[start]):
            start += 1
            continue
        state = int(vals[start])
        end = start + 1
        while end < n and not pd.isna(vals[end]) and int(vals[end]) == state:
            end += 1
        rows.append({"state": state, "run_length": end - start})
        start = end
    return pd.DataFrame(rows)


def persistence_table(df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    s = df[candidate]
    runs = run_lengths(s)
    rows: list[dict] = []
    nonmissing = s.notna().sum()
    for state in sorted(int(x) for x in s.dropna().unique()):
        mask = s == state
        prev_valid = s.shift(1).notna()
        denom = int((mask & prev_valid).sum())
        self_trans = float(((mask) & (s.shift(1) == state)).sum() / denom) if denom else np.nan
        rr = runs.loc[runs["state"] == state, "run_length"]
        rows.append(
            {
                "candidate": candidate,
                "state": state,
                "n_days": int(mask.sum()),
                "occupancy": float(mask.sum() / nonmissing) if nonmissing else np.nan,
                "self_transition_probability": self_trans,
                "median_run_length": float(rr.median()) if len(rr) else np.nan,
                "p90_run_length": float(rr.quantile(0.9)) if len(rr) else np.nan,
                "n_runs": int(len(rr)),
            }
        )
    return pd.DataFrame(rows)


def between_state_stat(values: np.ndarray, labels: np.ndarray) -> float:
    ok = np.isfinite(values)
    values = values[ok]
    labels = labels[ok]
    if len(values) < 2:
        return np.nan
    grand = float(np.mean(values))
    total = float(np.sum((values - grand) ** 2))
    if total <= 0:
        return 0.0
    between = 0.0
    for state in np.unique(labels):
        v = values[labels == state]
        if len(v):
            between += len(v) * (float(np.mean(v)) - grand) ** 2
    return between / total


def block_permutation_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    n_perm: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    ok = np.isfinite(values) & pd.notna(labels)
    values = values[ok]
    labels = labels[ok].astype(int)
    observed = between_state_stat(values, labels)
    if not np.isfinite(observed) or len(values) < block_size * 3:
        return observed, np.nan

    n = len(labels)
    starts = list(range(0, n, block_size))
    blocks = [labels[i : min(i + block_size, n)] for i in starts]
    ge = 0
    for _ in range(n_perm):
        order = rng.permutation(len(blocks))
        perm = np.concatenate([blocks[i] for i in order])[:n]
        stat = between_state_stat(values, perm)
        ge += int(stat >= observed)
    return observed, (ge + 1.0) / (n_perm + 1.0)


def bh_fdr(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    ranked = valid.to_numpy() * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out.loc[valid.index] = np.minimum(ranked, 1.0)
    return out


def period_mask(df: pd.DataFrame, period: str) -> pd.Series:
    lo, hi = PERIODS[period]
    return (df["date"] >= lo) & (df["date"] <= hi)


def economic_separation(
    df: pd.DataFrame,
    candidate: str,
    n_perm: int,
    block_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests: list[dict] = []
    summaries: list[dict] = []
    base_rng = np.random.default_rng(seed)

    for period in PERIODS:
        pmask = period_mask(df, period)
        for outcome in OUTCOMES:
            sub = df.loc[pmask, [candidate, outcome]].dropna()
            if len(sub) < 100:
                continue
            labels = sub[candidate].to_numpy(dtype=int)
            values = sub[outcome].to_numpy(dtype=float)
            obs, p = block_permutation_pvalue(
                values,
                labels,
                n_perm=n_perm,
                block_size=block_size,
                rng=np.random.default_rng(base_rng.integers(0, 2**32 - 1)),
            )
            tests.append(
                {
                    "candidate": candidate,
                    "period": period,
                    "outcome": outcome,
                    "n": len(sub),
                    "n_states": int(sub[candidate].nunique()),
                    "effect_eta2": obs,
                    "block_permutation_p": p,
                }
            )
            for state, g in sub.groupby(candidate):
                summaries.append(
                    {
                        "candidate": candidate,
                        "period": period,
                        "outcome": outcome,
                        "state": int(state),
                        "n": len(g),
                        "mean": float(g[outcome].mean()),
                        "median": float(g[outcome].median()),
                        "std": float(g[outcome].std(ddof=1)),
                    }
                )
    return pd.DataFrame(tests), pd.DataFrame(summaries)


def transition_events(labels: pd.Series, cooldown: int) -> list[tuple[int, int, int]]:
    vals = labels.astype("Int64").to_numpy()
    raw: list[tuple[int, int, int]] = []
    for i in range(1, len(vals)):
        if pd.isna(vals[i - 1]) or pd.isna(vals[i]):
            continue
        a, b = int(vals[i - 1]), int(vals[i])
        if a != b:
            raw.append((i, a, b))

    kept: list[tuple[int, int, int]] = []
    last_by_pair: dict[tuple[int, int], int] = {}
    for i, a, b in raw:
        pair = (a, b)
        if i - last_by_pair.get(pair, -10**9) > cooldown:
            kept.append((i, a, b))
            last_by_pair[pair] = i
    return kept


def transition_feasibility(df: pd.DataFrame, candidate: str, cooldown: int) -> pd.DataFrame:
    events = transition_events(df[candidate], cooldown)
    rows: list[dict] = []
    if not events:
        return pd.DataFrame()

    event_df = pd.DataFrame(events, columns=["idx", "from_state", "to_state"])
    event_df["date"] = df.loc[event_df["idx"], "date"].to_numpy()
    for outcome in OUTCOMES:
        event_df[outcome] = df.loc[event_df["idx"], outcome].to_numpy()

    raw_changes = (
        df[candidate].notna()
        & df[candidate].shift(1).notna()
        & (df[candidate] != df[candidate].shift(1))
    )
    raw_pairs = pd.DataFrame(
        {
            "from_state": df[candidate].shift(1)[raw_changes].astype(int),
            "to_state": df[candidate][raw_changes].astype(int),
        }
    )
    raw_counts = raw_pairs.value_counts().to_dict()

    for (a, b), g in event_df.groupby(["from_state", "to_state"]):
        row = {
            "candidate": candidate,
            "from_state": int(a),
            "to_state": int(b),
            "raw_transition_days": int(raw_counts.get((a, b), 0)),
            "episode_count": int(len(g)),
        }
        for period in PERIODS:
            if period == "full":
                continue
            lo, hi = PERIODS[period]
            row[f"episodes_{period}"] = int(((g["date"] >= lo) & (g["date"] <= hi)).sum())
        for outcome in OUTCOMES:
            row[f"{outcome}_mean"] = float(g[outcome].mean())
            row[f"{outcome}_median"] = float(g[outcome].median())
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_decision_table(
    persistence: pd.DataFrame,
    tests: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in sorted(persistence["candidate"].unique()):
        p = persistence[persistence["candidate"] == candidate]
        t = tests[tests["candidate"] == candidate]
        tr = transitions[transitions["candidate"] == candidate]

        min_occ = float(p["occupancy"].min())
        min_self = float(p["self_transition_probability"].min())
        max_episode = int(tr["episode_count"].max()) if len(tr) else 0

        oos = t[t["period"].isin(["2015_2019", "2020_2024"])]
        gain_loss = oos[oos["outcome"].isin([
            "fwd_return_5d",
            "fwd_return_10d",
            "fwd_return_20d",
            "fwd_max_drawdown_20d",
            "fwd_max_upside_20d",
        ])]
        n_sig_oos = int((gain_loss["q_value"] < 0.05).sum())
        periods_sig = int(gain_loss.loc[gain_loss["q_value"] < 0.05, "period"].nunique())

        reasons: list[str] = []
        if min_occ < 0.05:
            reasons.append("tiny_state")
        if min_self < 0.50:
            reasons.append("weak_persistence")
        if n_sig_oos == 0:
            reasons.append("no_oos_gain_loss_separation")
        if periods_sig < 2:
            reasons.append("not_stable_across_both_oos_periods")
        if max_episode < 50:
            reasons.append("too_few_transition_episodes")

        rows.append(
            {
                "candidate": candidate,
                "min_occupancy": min_occ,
                "min_self_transition_probability": min_self,
                "n_significant_oos_gain_loss_tests": n_sig_oos,
                "n_oos_periods_with_significance": periods_sig,
                "max_directed_transition_episode_count": max_episode,
                "passes_conservative_screen": len(reasons) == 0,
                "rejection_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    decisions: pd.DataFrame,
    persistence: pd.DataFrame,
    tests: pd.DataFrame,
    transitions: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Regime Discovery Audit Report",
        "",
        "This report is generated by `scripts/run_regime_discovery_audit.py`.",
        "",
        "## Audit configuration",
        "",
        f"- permutations: {args.permutations}",
        f"- block size: {args.block_size} trading days",
        f"- transition cooldown: {args.transition_cooldown} trading days",
        f"- seed: {args.seed}",
        "- unsupervised fit period ends: 2014-12-31",
        "",
        "## Conservative candidate decisions",
        "",
        decisions.to_markdown(index=False),
        "",
    ]

    passed = decisions[decisions["passes_conservative_screen"]]
    if passed.empty:
        lines += [
            "## Primary conclusion",
            "",
            "No candidate state system passed the conservative screen. The dataset does not yet support a defensible regime-transition target under the preregistered criteria.",
            "",
        ]
    else:
        lines += [
            "## Primary conclusion",
            "",
            "At least one candidate passed the conservative screen. This is only evidence that a transition target may be feasible; it is not evidence that the transition is forecastable or that QRC has an advantage.",
            "",
            "Passing candidates:",
            "",
            passed.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Strongest out-of-sample economic-separation tests",
        "",
    ]
    oos = tests[tests["period"].isin(["2015_2019", "2020_2024"])].sort_values(
        ["q_value", "effect_eta2"], ascending=[True, False]
    ).head(20)
    lines += [oos.to_markdown(index=False), ""]

    lines += [
        "## Largest directed transition samples",
        "",
    ]
    top_tr = transitions.sort_values("episode_count", ascending=False).head(20)
    lines += [top_tr.to_markdown(index=False), ""]

    lines += [
        "## Minimum state persistence by candidate",
        "",
        persistence.groupby("candidate", as_index=False)
        .agg(
            min_occupancy=("occupancy", "min"),
            min_self_transition=("self_transition_probability", "min"),
            min_median_run=("median_run_length", "min"),
        )
        .to_markdown(index=False),
        "",
        "## Interpretation rule",
        "",
        "A statistically significant result is not automatically a useful regime. Candidate systems are rejected if they rely on tiny states, lack persistence, fail out of sample, depend on one period, or yield too few independent transition episodes.",
    ]
    (outdir / "regime_discovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)

    persistence_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []

    specs = candidate_specs()
    manifest = {
        "data": str(args.data),
        "n_rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "train_end": str(TRAIN_END.date()),
        "permutations": args.permutations,
        "block_size": args.block_size,
        "transition_cooldown": args.transition_cooldown,
        "seed": args.seed,
        "candidates": [spec.__dict__ for spec in specs],
        "outcomes": OUTCOMES,
    }

    for i, spec in enumerate(specs):
        candidate_col = f"state__{spec.name}"
        df[candidate_col] = fit_candidate(df, spec, seed=args.seed + i)
        persistence_frames.append(persistence_table(df, candidate_col))
        tests, summaries = economic_separation(
            df,
            candidate_col,
            n_perm=args.permutations,
            block_size=args.block_size,
            seed=args.seed + 1000 + i,
        )
        test_frames.append(tests)
        summary_frames.append(summaries)
        tr = transition_feasibility(df, candidate_col, args.transition_cooldown)
        if not tr.empty:
            transition_frames.append(tr)

    persistence = pd.concat(persistence_frames, ignore_index=True)
    tests = pd.concat(test_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()

    tests["q_value"] = bh_fdr(tests["block_permutation_p"])
    decisions = candidate_decision_table(persistence, tests, transitions)

    persistence.to_csv(args.outdir / "state_persistence.csv", index=False)
    tests.to_csv(args.outdir / "economic_separation_tests.csv", index=False)
    summaries.to_csv(args.outdir / "state_outcome_summaries.csv", index=False)
    transitions.to_csv(args.outdir / "transition_feasibility.csv", index=False)
    decisions.to_csv(args.outdir / "candidate_decisions.csv", index=False)
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.outdir, decisions, persistence, tests, transitions, args)

    print("\n=== Conservative candidate decisions ===")
    print(decisions.to_string(index=False))
    print(f"\nWrote audit outputs to {args.outdir}")


if __name__ == "__main__":
    main()
