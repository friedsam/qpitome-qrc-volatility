from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.regimes.branch_intermediate_dynamics import IntermediateProbeConfig, build_first_passage_episode_table
from qpitome_qrc.regimes.market_portability import DEFAULT_MARKETS, audit_market_portability

CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/baselines/cross_market_day5_fixed_esn_v1")
FOLLOWUP = 120
LANDMARK = 5
MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}
D1 = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]


def complete(episodes, n_rows):
    return episodes[(n_rows - episodes["branch_idx"].astype(int) - 1) >= FOLLOWUP].copy()


def add_sequences(market_key, market, daily, episodes, events):
    events = events[events["resolved_within_followup"] & events["event_type"].isin(["recovery", "relapse"])].copy()
    episodes = episodes.merge(events[["episode_id", "event_day", "event_type", "upper_barrier", "lower_barrier"]], on="episode_id")
    prices = daily["spy_adj_close"].to_numpy(float)
    rows = []
    seqs = []
    for ep in episodes.itertuples(index=False):
        b = int(ep.branch_idx)
        if int(ep.event_day) <= LANDMARK or b + LANDMARK >= len(daily):
            continue
        branch_price = prices[b]
        upper = float(ep.upper_barrier)
        lower = float(ep.lower_barrier)
        width = upper - lower
        seq = []
        for k in range(LANDMARK + 1):
            r = prices[b + k] / branch_price - 1.0
            seq.append([r, upper - r, r - lower, width])
        p5 = prices[b + LANDMARK]
        r5 = p5 / branch_price - 1.0
        rows.append({
            "market_key": market_key,
            "market": market,
            "episode_id": int(ep.episode_id),
            "branch_date": ep.branch_date,
            "landmark_date": daily.iloc[b + LANDMARK]["date"],
            "y_recovery": int(ep.event_type == "recovery"),
            "current_return_5d_from_branch": r5,
            "distance_to_recovery_barrier": upper - r5,
            "distance_to_relapse_barrier": r5 - lower,
            "barrier_width": width,
        })
        seqs.append(seq)
    frame = pd.DataFrame(rows)
    return frame, np.asarray(seqs, dtype=float)


def market_sequences(key, path):
    daily, episodes, events, _ = audit_market_portability(path, key)
    episodes = complete(episodes, len(daily))
    events = events[events["episode_id"].isin(episodes["episode_id"])].copy()
    return add_sequences(key, DEFAULT_MARKETS[key].market, daily, episodes, events)


def spy_sequences():
    daily = pd.read_csv("data/processed/phase3_spy_vix_volatility_extended.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv", parse_dates=["branch_date"])
    episodes = complete(episodes, len(daily))
    events = build_first_passage_episode_table(daily, episodes, IntermediateProbeConfig(max_followup=FOLLOWUP))
    return add_sequences("spy", "SPY", daily, episodes, events)


def fixed_esn_weights(input_dim, n_reservoir=32, seed=23):
    rng = np.random.default_rng(seed)
    win = rng.normal(0.0, 0.45, size=(n_reservoir, input_dim))
    w = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    mask = rng.random(w.shape) < 0.18
    w *= mask
    eig = np.linalg.eigvals(w)
    radius = np.max(np.abs(eig))
    if radius > 0:
        w *= 0.75 / radius
    bias = rng.normal(0.0, 0.05, size=n_reservoir)
    return win, w, bias


WIN, WRES, BIAS = fixed_esn_weights(4)


def esn_state(seq):
    h = np.zeros(WRES.shape[0])
    for u in seq:
        h = 0.35 * h + 0.65 * np.tanh(WIN @ u + WRES @ h + BIAS)
    return h


def score(group, model):
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = np.clip(use[model].to_numpy(float), 1e-6, 1 - 1e-6)
    return {
        "model": model,
        "n": len(use),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


def fit_predict_d1(train, test):
    sx = StandardScaler()
    x_train = sx.fit_transform(train[D1].to_numpy(float))
    x_test = sx.transform(test[D1].to_numpy(float))
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(x_train, train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(x_test)[0, 1])


def fit_predict_esn(train_idx, test_idx, frame, seqs):
    train = frame.iloc[train_idx]
    sx = StandardScaler()
    flat_train = seqs[train_idx].reshape(-1, seqs.shape[-1])
    sx.fit(flat_train)
    train_states = np.vstack([esn_state(sx.transform(seq)) for seq in seqs[train_idx]])
    test_state = np.vstack([esn_state(sx.transform(seqs[test_idx]))])
    model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
    model.fit(train_states, train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test_state)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    arrays = []
    f, s = spy_sequences(); frames.append(f); arrays.append(s)
    for key, path in INPUTS.items():
        f, s = market_sequences(key, path); frames.append(f); arrays.append(s)
    frame = pd.concat(frames, ignore_index=True)
    seqs = np.concatenate(arrays, axis=0)
    clusters = pd.read_csv(CLUSTERS)[["market_key", "episode_id", "cluster_id"]]
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left")
    cluster_start = frame.groupby("cluster_id", as_index=False)["branch_date"].min().rename(columns={"branch_date": "cluster_start"})
    frame = frame.merge(cluster_start, on="cluster_id", how="left")
    keep = frame.dropna(subset=D1 + ["y_recovery", "landmark_date", "cluster_start"]).index.to_numpy()
    frame = frame.iloc[keep].sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)
    seqs = seqs[keep]

    rows = []
    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train_idx = frame.index[frame["landmark_date"] < row["cluster_start"]].to_numpy()
        if len(train_idx) < MIN_TRAIN or frame.iloc[train_idx]["y_recovery"].nunique() < 2:
            continue
        test = frame.iloc[[i]]
        train = frame.iloc[train_idx]
        prior = float(np.clip(train["y_recovery"].mean(), 1e-6, 1 - 1e-6))
        rows.append({
            "row_id": i,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D0_prior": prior,
            "D1_geometry": fit_predict_d1(train, test),
            "ESN1_fixed_short_path": fit_predict_esn(train_idx, i, frame, seqs),
        })
    preds = pd.DataFrame(rows)
    models = ["D0_prior", "D1_geometry", "ESN1_fixed_short_path"]
    metrics = pd.DataFrame([score(preds, m) for m in models])

    cluster_rows = []
    for model in models:
        losses = []
        for cid, g in preds.dropna(subset=[model]).groupby("cluster_id"):
            y = g["y"].to_numpy(int)
            p = np.clip(g[model].to_numpy(float), 1e-6, 1 - 1e-6)
            losses.append({"cluster_id": cid, "n": len(g), "mean_logloss": float(log_loss(y, p, labels=[0, 1])), "mean_brier": float(np.mean((p - y) ** 2))})
        loss = pd.DataFrame(losses)
        cluster_rows.append({"model": model, "n_clusters": int(loss["cluster_id"].nunique()), "cluster_mean_logloss": float(loss["mean_logloss"].mean()), "cluster_mean_brier": float(loss["mean_brier"].mean()), "median_cluster_size": float(loss["n"].median())})
    cluster_metrics = pd.DataFrame(cluster_rows)

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
