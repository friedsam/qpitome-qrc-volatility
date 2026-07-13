from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.day5.protocol import D1, EVAL_START, MIN_TRAIN
from qpitome_qrc.evaluation.binary import logistic_pipeline
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/qrc/cross_market_day5_fixed_rydberg_feature_v1")
N_QUBITS = 4
N_STATE = 2 ** N_QUBITS
OMEGA = 1.0
EVOLVE_TIME = 1.35
POSITIONS = np.array([0.0, 1.0, 2.15, 3.6])

# Backward-compatible historical name.
score = binary_summary


def bit_value(index, qubit):
    return 1 if (index & (1 << qubit)) else 0


def build_operators():
    x_ops = []
    n_ops = []
    nn_ops = []
    for q in range(N_QUBITS):
        x = np.zeros((N_STATE, N_STATE), dtype=complex)
        n = np.zeros((N_STATE, N_STATE), dtype=complex)
        for i in range(N_STATE):
            x[i ^ (1 << q), i] = 1.0
            n[i, i] = bit_value(i, q)
        x_ops.append(x)
        n_ops.append(n)
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            nn_ops.append((i, j, n_ops[i] @ n_ops[j]))
    return x_ops, n_ops, nn_ops


X_OPS, N_OPS, NN_OPS = build_operators()
PAIR_V = {}
for i in range(N_QUBITS):
    for j in range(i + 1, N_QUBITS):
        r = abs(POSITIONS[i] - POSITIONS[j])
        PAIR_V[(i, j)] = 0.85 / (r ** 6)


def rydberg_features(x):
    x = np.clip(np.asarray(x, dtype=float), -3.0, 3.0)
    deltas = 0.65 * x
    h = np.zeros((N_STATE, N_STATE), dtype=complex)
    for q in range(N_QUBITS):
        h += 0.5 * OMEGA * X_OPS[q]
        h += -deltas[q] * N_OPS[q]
    for i, j, nn in NN_OPS:
        h += PAIR_V[(i, j)] * nn
    eigvals, eigvecs = np.linalg.eigh(h)
    psi0 = np.zeros(N_STATE, dtype=complex)
    psi0[0] = 1.0
    coeff = eigvecs.conj().T @ psi0
    psi = eigvecs @ (np.exp(-1j * eigvals * EVOLVE_TIME) * coeff)
    feats = []
    for q in range(N_QUBITS):
        feats.append(float(np.real(np.vdot(psi, N_OPS[q] @ psi))))
    for i, j, nn in NN_OPS:
        feats.append(float(np.real(np.vdot(psi, nn @ psi))))
    # Add final-state probability entropy and excitation count as fixed observables.
    probs = np.abs(psi) ** 2
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    excitation_count = sum(feats[:N_QUBITS])
    feats.extend([entropy, excitation_count])
    return np.asarray(feats, dtype=float)


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
