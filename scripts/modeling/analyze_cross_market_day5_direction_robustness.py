from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = BASE / "robustness"
MODELS = ["D0_prior", "D1_geometry", "D2_geometry_downside", "D3_compact_quadratic"]


def metric_row(name, group, model):
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = np.clip(use[model].to_numpy(float), 1e-6, 1 - 1e-6)
    return {
        "group": name,
        "model": model,
        "n": len(use),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


def cluster_weighted(group, model):
    rows = []
    for cid, g in group.dropna(subset=[model, "cluster_id"]).groupby("cluster_id"):
        y = g["y"].to_numpy(int)
        p = np.clip(g[model].to_numpy(float), 1e-6, 1 - 1e-6)
        rows.append({
            "cluster_id": cid,
            "model": model,
            "n": len(g),
            "mean_logloss": float(log_loss(y, p, labels=[0, 1])),
            "mean_brier": float(np.mean((p - y) ** 2)),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {"model": model, "n_clusters": 0, "cluster_mean_logloss": np.nan, "cluster_mean_brier": np.nan}
    return detail, {
        "model": model,
        "n_clusters": int(detail["cluster_id"].nunique()),
        "cluster_mean_logloss": float(detail["mean_logloss"].mean()),
        "cluster_mean_brier": float(detail["mean_brier"].mean()),
        "median_cluster_size": float(detail["n"].median()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(BASE / "calendar_prequential_predictions.csv", parse_dates=["landmark_date"])
    if CLUSTERS.exists():
        clusters = pd.read_csv(CLUSTERS, parse_dates=["branch_date"])
        clusters = clusters[["market_key", "episode_id", "cluster_id"]]
        preds = preds.merge(clusters, on=["market_key", "episode_id"], how="left")
    rows = []
    for model in MODELS:
        rows.append(metric_row("pooled", preds, model))
        for market, group in preds.groupby("market_key"):
            rows.append(metric_row(f"market={market}", group, model))
    by_group = pd.DataFrame(rows)

    cluster_rows = []
    cluster_details = []
    for model in MODELS:
        detail, summary = cluster_weighted(preds, model)
        cluster_rows.append(summary)
        if not detail.empty:
            cluster_details.append(detail)
    cluster_summary = pd.DataFrame(cluster_rows)
    cluster_detail = pd.concat(cluster_details, ignore_index=True) if cluster_details else pd.DataFrame()

    by_group.to_csv(OUT / "metrics_by_market.csv", index=False)
    cluster_summary.to_csv(OUT / "cluster_weighted_metrics.csv", index=False)
    cluster_detail.to_csv(OUT / "cluster_loss_detail.csv", index=False)
    print("Cross-market day-5 robustness")
    print("\nBy market:")
    print(by_group.to_string(index=False))
    print("\nCluster-weighted:")
    print(cluster_summary.to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
