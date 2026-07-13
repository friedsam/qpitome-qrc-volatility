from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.priors import predict_empirical_prior
from qpitome_qrc.day5.protocol import D1, EVAL_START, MIN_TRAIN
from qpitome_qrc.evaluation.binary import fit_feature_only_predict
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary

BASE = Path("results/modeling/day5_branching/baseline")
LANDMARK = BASE / "cross_market_day5_direction_v1__day5_landmark_frame.csv"
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
PREFIX = "cross_market_day5_confirmatory_v1__"
MIN_MARKET_PRIOR = 5

D2 = D1 + ["downside_shock_pressure_day5"]
MODELS = {
    "D0_pooled_prior": {"kind": "prior", "market": False, "cols": []},
    "D0_market_prior": {"kind": "prior", "market": True, "cols": []},
    "D1_geometry": {"kind": "logit", "cols": D1},
    "D2_geometry_downside": {"kind": "logit", "cols": D2},
}

score = binary_summary


def predict_prior(train, market_key=None):
    return predict_empirical_prior(
        train,
        target="y_recovery",
        group_column="market_key" if market_key is not None else None,
        group_value=market_key,
        min_group_rows=MIN_MARKET_PRIOR,
        clip=1e-6,
    )


def predict_logit(train, test, cols):
    if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
        return np.nan
    return fit_feature_only_predict(
        train[cols].to_numpy(float),
        train["y_recovery"].to_numpy(int),
        test[cols].to_numpy(float),
        C=1.0,
    )


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        LANDMARK,
        parse_dates=["branch_date", "landmark_date"],
    )
    clusters = pd.read_csv(CLUSTERS)[["market_key", "episode_id", "cluster_id"]]
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left")
    cluster_start = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(cluster_start, on="cluster_id", how="left")
    frame = frame.dropna(
        subset=D2 + ["y_recovery", "landmark_date", "cluster_start"]
    ).sort_values(
        ["landmark_date", "market_key", "episode_id"]
    ).reset_index(drop=True)

    rows = []
    for index, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) < MIN_TRAIN:
            continue
        test = frame.iloc[[index]]
        out = {
            "row_id": index,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
        }
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
            metric = binary_summary(group, model)
            metric["group"] = group_name
            metric_rows.append(metric)
    metrics = pd.DataFrame(metric_rows)[
        ["group", "model", "n", "recovery_rate", "auc", "pr_auc", "logloss", "brier"]
    ]

    cluster_metrics = pd.DataFrame([
        cluster_weighted_summary(preds, model) for model in MODELS
    ])

    preds.to_csv(BASE / f"{PREFIX}purged_calendar_prequential_predictions.csv", index=False)
    metrics.to_csv(BASE / f"{PREFIX}summary_metrics.csv", index=False)
    cluster_metrics.to_csv(BASE / f"{PREFIX}cluster_weighted_metrics.csv", index=False)
    print("Cross-market day-5 confirmatory baselines")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {BASE}")


if __name__ == "__main__":
    main()
