from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.fixed_esn import esn_state, fit_esn_predict, fixed_esn_weights
from qpitome_qrc.day5.protocol import D1, EVAL_START, MIN_TRAIN
from qpitome_qrc.evaluation.binary import fit_feature_only_predict
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary
from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)
from qpitome_qrc.regimes.market_portability import DEFAULT_MARKETS, audit_market_portability

CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/baselines/cross_market_day5_fixed_esn_v1")
FOLLOWUP = 120
LANDMARK = 5
INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}


def complete(episodes, n_rows):
    return episodes[(n_rows - episodes["branch_idx"].astype(int) - 1) >= FOLLOWUP].copy()


def add_sequences(market_key, market, daily, episodes, events):
    events = events[
        events["resolved_within_followup"]
        & events["event_type"].isin(["recovery", "relapse"])
    ].copy()
    episodes = episodes.merge(
        events[["episode_id", "event_day", "event_type", "upper_barrier", "lower_barrier"]],
        on="episode_id",
    )
    prices = daily["spy_adj_close"].to_numpy(float)
    rows = []
    seqs = []
    for ep in episodes.itertuples(index=False):
        branch_index = int(ep.branch_idx)
        if int(ep.event_day) <= LANDMARK or branch_index + LANDMARK >= len(daily):
            continue
        branch_price = prices[branch_index]
        upper = float(ep.upper_barrier)
        lower = float(ep.lower_barrier)
        width = upper - lower
        sequence = []
        for offset in range(LANDMARK + 1):
            return_from_branch = prices[branch_index + offset] / branch_price - 1.0
            sequence.append([
                return_from_branch,
                upper - return_from_branch,
                return_from_branch - lower,
                width,
            ])
        price_day5 = prices[branch_index + LANDMARK]
        return_day5 = price_day5 / branch_price - 1.0
        rows.append({
            "market_key": market_key,
            "market": market,
            "episode_id": int(ep.episode_id),
            "branch_date": ep.branch_date,
            "landmark_date": daily.iloc[branch_index + LANDMARK]["date"],
            "y_recovery": int(ep.event_type == "recovery"),
            "current_return_5d_from_branch": return_day5,
            "distance_to_recovery_barrier": upper - return_day5,
            "distance_to_relapse_barrier": return_day5 - lower,
            "barrier_width": width,
        })
        seqs.append(sequence)
    frame = pd.DataFrame(rows)
    return frame, np.asarray(seqs, dtype=float)


def market_sequences(key, path):
    daily, episodes, events, _ = audit_market_portability(path, key)
    episodes = complete(episodes, len(daily))
    events = events[events["episode_id"].isin(episodes["episode_id"])].copy()
    return add_sequences(key, DEFAULT_MARKETS[key].market, daily, episodes, events)


def spy_sequences():
    daily = pd.read_csv(
        "data/processed/phase3_spy_vix_volatility_extended.csv",
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(
        "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv",
        parse_dates=["branch_date"],
    )
    episodes = complete(episodes, len(daily))
    events = build_first_passage_episode_table(
        daily,
        episodes,
        IntermediateProbeConfig(max_followup=FOLLOWUP),
    )
    return add_sequences("spy", "SPY", daily, episodes, events)


WIN, WRES, BIAS = fixed_esn_weights(4)

# Backward-compatible historical name.
score = binary_summary


def fit_predict_d1(train, test):
    return fit_feature_only_predict(
        train[D1].to_numpy(float),
        train["y_recovery"].to_numpy(int),
        test[D1].to_numpy(float),
        C=1.0,
    )


def fit_predict_esn(train_idx, test_idx, frame, seqs):
    return fit_esn_predict(
        seqs[train_idx],
        frame.iloc[train_idx]["y_recovery"].to_numpy(int),
        seqs[test_idx],
        WIN,
        WRES,
        BIAS,
        C=0.1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    arrays = []
    frame, sequences = spy_sequences()
    frames.append(frame)
    arrays.append(sequences)
    for key, path in INPUTS.items():
        frame, sequences = market_sequences(key, path)
        frames.append(frame)
        arrays.append(sequences)
    frame = pd.concat(frames, ignore_index=True)
    seqs = np.concatenate(arrays, axis=0)
    clusters = pd.read_csv(CLUSTERS)[["market_key", "episode_id", "cluster_id"]]
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left")
    cluster_start = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(cluster_start, on="cluster_id", how="left")
    keep = frame.dropna(
        subset=D1 + ["y_recovery", "landmark_date", "cluster_start"]
    ).index.to_numpy()
    frame = frame.iloc[keep].sort_values(
        ["landmark_date", "market_key", "episode_id"]
    ).reset_index(drop=True)
    seqs = seqs[keep]

    rows = []
    for index, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train_idx = frame.index[frame["landmark_date"] < row["cluster_start"]].to_numpy()
        if (
            len(train_idx) < MIN_TRAIN
            or frame.iloc[train_idx]["y_recovery"].nunique() < 2
        ):
            continue
        test = frame.iloc[[index]]
        train = frame.iloc[train_idx]
        prior = float(np.clip(train["y_recovery"].mean(), 1e-6, 1 - 1e-6))
        rows.append({
            "row_id": index,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D0_prior": prior,
            "D1_geometry": fit_predict_d1(train, test),
            "ESN1_fixed_short_path": fit_predict_esn(train_idx, index, frame, seqs),
        })
    preds = pd.DataFrame(rows)
    models = ["D0_prior", "D1_geometry", "ESN1_fixed_short_path"]
    metrics = pd.DataFrame([binary_summary(preds, model) for model in models])
    cluster_metrics = pd.DataFrame([
        cluster_weighted_summary(preds, model) for model in models
    ])

    preds.to_csv(OUT / "purged_calendar_prequential_predictions.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    cluster_metrics.to_csv(OUT / "cluster_weighted_metrics.csv", index=False)
    print("Cross-market day-5 fixed ESN")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
