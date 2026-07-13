from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.day5.protocol import D1, EVAL_START, MIN_TRAIN
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary
from qpitome_qrc.qrc.fixed_rydberg_feature import (
    EVOLVE_TIME,
    N_OPS,
    N_QUBITS,
    N_STATE,
    NN_OPS,
    OMEGA,
    PAIR_V,
    POSITIONS,
    X_OPS,
    bit_value,
    build_operators,
    rydberg_features,
)

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/qrc/cross_market_day5_fixed_rydberg_feature_v1")

# Backward-compatible historical name.
score = binary_summary


def fit_linear(train, test):
    model = logistic_pipeline(1.0)
    model.fit(train[D1].to_numpy(float), train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test[D1].to_numpy(float))[0, 1])


def fit_rydberg(train, test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[D1].to_numpy(float))
    x_test = scaler.transform(test[D1].to_numpy(float))
    r_train = np.vstack([rydberg_features(x) for x in x_train])
    r_test = np.vstack([rydberg_features(x) for x in x_test])
    model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
    model.fit(r_train, train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(r_test)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "day5_landmark_frame.csv", parse_dates=["branch_date", "landmark_date"])
    clusters = pd.read_csv(CLUSTERS)[["market_key", "episode_id", "cluster_id"]]
    frame = frame.merge(clusters, on=["market_key", "episode_id"], how="left")
    cluster_start = frame.groupby("cluster_id", as_index=False)["branch_date"].min().rename(columns={"branch_date": "cluster_start"})
    frame = frame.merge(cluster_start, on="cluster_id", how="left")
    frame = frame.dropna(subset=D1 + ["y_recovery", "landmark_date", "cluster_start"]).sort_values(["landmark_date", "market_key", "episode_id"]).reset_index(drop=True)

    rows = []
    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
            continue
        test = frame.iloc[[i]]
        prior = float(np.clip(train["y_recovery"].mean(), 1e-6, 1 - 1e-6))
        rows.append({
            "row_id": i,
            "market_key": row["market_key"],
            "episode_id": row["episode_id"],
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D0_prior": prior,
            "D1_geometry": fit_linear(train, test),
            "QR1_fixed_rydberg_feature": fit_rydberg(train, test),
        })
    preds = pd.DataFrame(rows)
    models = ["D0_prior", "D1_geometry", "QR1_fixed_rydberg_feature"]
    metrics = pd.DataFrame([binary_summary(preds, model) for model in models])
    cluster_metrics = pd.DataFrame([
        cluster_weighted_summary(preds, model) for model in models
    ])

    preds.to_csv(OUT / "purged_calendar_prequential_predictions.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    cluster_metrics.to_csv(OUT / "cluster_weighted_metrics.csv", index=False)
    print("Cross-market day-5 fixed Rydberg feature")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
