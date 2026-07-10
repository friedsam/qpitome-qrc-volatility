from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/qrc/cross_market_day5_fixed_quantum_feature_v1")
MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
D1 = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]
N_QUBITS = 4
N_STATE = 2 ** N_QUBITS


def apply_ry(state, qubit, theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    out = state.copy()
    bit = 1 << qubit
    for i in range(N_STATE):
        if i & bit:
            continue
        j = i | bit
        a = state[i]
        b = state[j]
        out[i] = c * a - s * b
        out[j] = s * a + c * b
    return out


def apply_rx(state, qubit, theta):
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    out = state.copy()
    bit = 1 << qubit
    for i in range(N_STATE):
        if i & bit:
            continue
        j = i | bit
        a = state[i]
        b = state[j]
        out[i] = c * a + s * b
        out[j] = s * a + c * b
    return out


def apply_zz_phase(state, q1, q2, theta):
    out = state.copy()
    b1 = 1 << q1
    b2 = 1 << q2
    for i in range(N_STATE):
        z1 = -1 if (i & b1) else 1
        z2 = -1 if (i & b2) else 1
        out[i] *= np.exp(-0.5j * theta * z1 * z2)
    return out


def expectation_z(state, qubit):
    bit = 1 << qubit
    probs = np.abs(state) ** 2
    signs = np.array([-1 if (i & bit) else 1 for i in range(N_STATE)])
    return float(np.sum(signs * probs).real)


def expectation_x(state, qubit):
    bit = 1 << qubit
    total = 0.0j
    for i in range(N_STATE):
        total += np.conj(state[i]) * state[i ^ bit]
    return float(total.real)


def expectation_zz(state, q1, q2):
    b1 = 1 << q1
    b2 = 1 << q2
    probs = np.abs(state) ** 2
    signs = np.array([((-1 if (i & b1) else 1) * (-1 if (i & b2) else 1)) for i in range(N_STATE)])
    return float(np.sum(signs * probs).real)


def quantum_features(x):
    x = np.clip(np.asarray(x, dtype=float), -3.0, 3.0)
    state = np.zeros(N_STATE, dtype=complex)
    state[0] = 1.0
    for layer in range(2):
        for q in range(N_QUBITS):
            state = apply_ry(state, q, 0.85 * x[q] + 0.17 * layer)
        for q1, q2 in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]:
            state = apply_zz_phase(state, q1, q2, 0.35 + 0.08 * (q1 + q2))
        for q in range(N_QUBITS):
            state = apply_rx(state, q, 0.23 + 0.05 * q)
    feats = []
    feats.extend(expectation_z(state, q) for q in range(N_QUBITS))
    feats.extend(expectation_x(state, q) for q in range(N_QUBITS))
    feats.extend(expectation_zz(state, i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS))
    return np.asarray(feats, dtype=float)


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


def fit_linear(train, test):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(train[D1].to_numpy(float), train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(test[D1].to_numpy(float))[0, 1])


def fit_quantum(train, test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[D1].to_numpy(float))
    x_test = scaler.transform(test[D1].to_numpy(float))
    q_train = np.vstack([quantum_features(x) for x in x_train])
    q_test = np.vstack([quantum_features(x) for x in x_test])
    model = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
    model.fit(q_train, train["y_recovery"].to_numpy(int))
    return float(model.predict_proba(q_test)[0, 1])


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
            "Q1_fixed_quantum_feature": fit_quantum(train, test),
        })
    preds = pd.DataFrame(rows)
    metrics = pd.DataFrame([score(preds, m) for m in ["D0_prior", "D1_geometry", "Q1_fixed_quantum_feature"]])

    cluster_rows = []
    for model in ["D0_prior", "D1_geometry", "Q1_fixed_quantum_feature"]:
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
    print("Cross-market day-5 fixed quantum feature")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
