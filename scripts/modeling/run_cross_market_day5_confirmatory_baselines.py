from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/baselines/cross_market_day5_confirmatory_v1")
MIN_TRAIN = 30
MIN_MARKET_PRIOR = 5
EVAL_START = pd.Timestamp("1990-01-01")

D1 = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]
D2 = D1 + ["downside_shock_pressure_day5"]
MODELS = {
    "D0_pooled_prior": {"kind": "prior", "market": False, "cols": []},
    "D0_market_prior": {"kind": "prior", "market": True, "cols": []},
    "D1_geometry": {"kind": "logit", "cols": D1},
    "D2_geometry_downside": {"kind": "logit", "cols": D2},
}


def predict_prior(train, market_key=None):
    use = train if market_key is None else train[train["market_key"] == market_key]
    if len(use) < MIN_MARKET_PRIOR:
        use = train
    return float(np.clip(use["y_recovery"].mean(), 1e-6, 1 - 1e-6))


def predict_logit(train, test, cols):
    if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
        return np.nan
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(train[cols].to_numpy(float), train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test[cols].to_numpy(float))[0, 1])


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "day5_landmark_frame.csv", parse_dates=["branch_date", "landmark_date"])
    clusters = pd.read_csv(CLUSTERS, parse_dates=["branch_date"])
    clusters = clusters[["market_key", "episode_id", "cluster_id"]]
    cluster_start = clusters.groupby("cluster_id", as_index=False)["branch_date"].min().rename(columns={"branch_date": "cluster_start"})
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left").merge(cluster_start, on="cluster_id", how="left")
    frame = frame.dropna(subset=D2 + ["y_recovery", "landmark_date", "cluster_start"]).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)

    rows = []
    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) < MIN_TRAIN:
            continue
        test = frame.iloc[[i]]
        out = {"row_id": i, "market_key": row["market_key"], "episode_id": row["episode_id"], "cluster_id": row["cluster_id"], "landmark_date": row["landmark_date"], "y": int(row["y_recovery"])}
        out["D0_pooled_prior"] = predict_prior(train)
        out["D0_market_prior"] = predict_prior(train, row["market_key"])
        out["D1_geometry"] = predict_logit(train, test, D1)
        out["D2_geometry_downside"] = predict_logit(train, test, D2)
        rows.append(out)
    preds = pd.DataFrame(rows)

    metric_rows = []
    groups = {
        "all_post1990_purged": preds,
        "validation_non_spy": preds[preds["market_key"] != "spy"],
        "leave_nikkei_out_eval": preds[preds["market_key"] != "nikkei_225"],
        "nikkei_only_eval": preds[preds["market_key"] == "nikkei_225"],
    }
    for group_name, group in groups.items():
        for model in MODELS:
            row = score(group, model)
            row["group"] = group_name
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)[["group", "model", "n", "recovery_rate", "auc", "pr_auc", "logloss", "brier"]]

    cluster_rows = []
    for model in MODELS:
        by_cluster = []
        for cid, g in preds.dropna(subset=[model]).groupby("cluster_id"):
            y = g["y"].to_numpy(int)
            p = np.clip(g[model].to_numpy(float), 1e-6, 1 - 1e-6)
            by_cluster.append({"cluster_id": cid, "n": len(g), "mean_logloss": float(log_loss(y, p, labels=[0, 1])), "mean_brier": float(np.mean((p - y) ** 2))})
        detail = pd.DataFrame(by_cluster)
        cluster_rows.append({"model": model, "n_clusters": int(detail["cluster_id"].nunique()), "cluster_mean_logloss": float(detail["mean_logloss"].mean()), "cluster_mean_brier": float(detail["mean_brier"].mean()), "median_cluster_size": float(detail["n"].median())})
    cluster_metrics = pd.DataFrame(cluster_rows)

    preds.to_csv(OUT / "purged_calendar_prequential_predictions.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    cluster_metrics.to_csv(OUT / "cluster_weighted_metrics.csv", index=False)
    print("Cross-market day-5 confirmatory baselines")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
