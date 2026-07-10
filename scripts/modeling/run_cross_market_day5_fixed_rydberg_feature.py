from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("results/baselines/cross_market_day5_direction_v1")
CLUSTERS = Path("results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv")
OUT = Path("results/qrc/cross_market_day5_fixed_rydberg_feature_v1")
MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
D1 = ["current_return_5d_from_branch", "distance_to_recovery_barrier", "distance_to_relapse_barrier", "barrier_width"]
N_QUBITS = 4
N_STATE = 2 ** N_QUBITS
OMEGA = 1.0
EVOLVE_TIME = 1.35
POSITIONS = np.array([0.0, 1.0, 2.15, 3.6])


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
    print("Cross-market day-5 fixed Rydberg feature")
    print("Predictions:", len(preds))
    print("\nSummary:")
    print(metrics.to_string(index=False))
    print("\nCluster weighted:")
    print(cluster_metrics.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
