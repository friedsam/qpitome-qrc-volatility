#!/usr/bin/env python3
"""
Exploratory branch-state ESN probe.

Purpose
-------
Run one bounded, reproducible ESN assessment on the day-0..5 branch path:

1. Reproduce locked D1.
2. Test a direct temporal ESN.
3. Test joint D1 + ESN features.
4. Test an ESN correction to honest prequential D1 logits.

This runner is intentionally exploratory. It defaults to writing under /tmp and
should not be promoted into the final comparison without review.

Required existing inputs
------------------------
scratch/path_panel_day0_5.csv
results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.numpy_esn import make_esn_weights, esn_states


MIN_TRAIN = 30
EVAL_START = pd.Timestamp("1990-01-01")
EPS = 1e-6

D1 = [
    "current_return_5d_from_branch",
    "distance_to_recovery_barrier",
    "distance_to_relapse_barrier",
    "barrier_width",
]

CHANNELS = [
    "signed_return_over_local_vol",
    "downside_shock_pressure",
    "d_log_rv5_over_rv20",
    "drawdown_repair_over_local_vol",
]


@dataclass(frozen=True)
class ESNConfig:
    name: str
    n_reservoir: int
    spectral_radius: float
    input_scale: float
    leak: float
    seeds: tuple[int, ...]
    correction_l2: float


PRIMARY = ESNConfig(
    name="primary_n64",
    n_reservoir=64,
    spectral_radius=0.90,
    input_scale=1.00,
    leak=1.00,
    seeds=(7, 42, 123, 1001, 2026),
    correction_l2=10.0,
)

SENSITIVITY = ESNConfig(
    name="sensitivity_n32",
    n_reservoir=32,
    spectral_radius=0.80,
    input_scale=0.50,
    leak=0.70,
    seeds=(7, 42, 123, 1001, 2026),
    correction_l2=10.0,
)


def clip_prob(p: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p: np.ndarray | float) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def load_frame(path_panel: Path, clusters_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path_panel,
        parse_dates=["branch_date", "landmark_date"],
    )
    clusters = pd.read_csv(clusters_path)[
        ["market_key", "episode_id", "cluster_id"]
    ].drop_duplicates()

    frame = frame.merge(
        clusters,
        on=["market_key", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    if frame["cluster_id"].isna().any():
        n = int(frame["cluster_id"].isna().sum())
        raise ValueError(f"{n} path rows have no cluster_id")

    starts = (
        frame.groupby("cluster_id", as_index=False)["branch_date"]
        .min()
        .rename(columns={"branch_date": "cluster_start"})
    )
    frame = frame.merge(starts, on="cluster_id", how="left", validate="many_to_one")

    needed = D1 + ["y_recovery", "landmark_date", "cluster_start"]
    needed += [f"{ch}_d{d}" for d in range(1, 6) for ch in CHANNELS]
    frame = (
        frame.dropna(subset=needed)
        .sort_values(["landmark_date", "market_key", "episode_id"])
        .reset_index(drop=True)
    )

    if not np.allclose(
        frame["r_d5"].to_numpy(float),
        frame["current_return_5d_from_branch"].to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("r_d5 does not match locked day-5 endpoint return")

    return frame


def make_sequence_tensor(frame: pd.DataFrame) -> np.ndarray:
    """Five-step, four-channel causal sequence. No duplicate endpoint-return channel."""
    return np.stack(
        [
            np.column_stack(
                [frame[f"{ch}_d{d}"].to_numpy(float) for ch in CHANNELS]
            )
            for d in range(1, 6)
        ],
        axis=1,
    )


def fit_d1(train: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[D1].to_numpy(float), train["y_recovery"].to_numpy(int))
    return model


def compute_honest_d1_predictions(frame: pd.DataFrame) -> np.ndarray:
    """
    D1 prediction for every row using only rows whose landmark date precedes
    that row's crisis-cluster start. These are honest offsets for correction training.
    """
    pred = np.full(len(frame), np.nan)
    for i, row in frame.iterrows():
        train = frame[frame["landmark_date"] < row["cluster_start"]]
        if len(train) < MIN_TRAIN or train["y_recovery"].nunique() < 2:
            continue
        pred[i] = fit_d1(train).predict_proba(
            frame.loc[[i], D1].to_numpy(float)
        )[0, 1]
    return pred


def fit_offset_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, float]:
    """
    Fit p = sigmoid(offset + intercept + X @ beta).
    Features must already be standardized.
    Intercept is unpenalized; beta has L2 penalty.
    """
    n_features = X.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        beta = theta[1:]
        eta = offset + intercept + X @ beta
        p = clip_prob(sigmoid(eta))
        loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss += 0.5 * l2 * float(beta @ beta)

        residual = p - y
        grad_intercept = float(np.sum(residual))
        grad_beta = X.T @ residual + l2 * beta
        grad = np.concatenate([[grad_intercept], grad_beta])
        return float(loss), grad

    result = minimize(
        fun=lambda th: objective(th)[0],
        x0=np.zeros(n_features + 1),
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Offset optimization failed: {result.message}")
    return result.x[1:], float(result.x[0])


def reservoir_features_for_fold(
    X_seq: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ESNConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit causal input scaling on the outer training rows, then generate ESN states.
    The shared repository ESN mechanics reset state for each episode.
    """
    n_rows, n_steps, n_inputs = X_seq.shape
    scaler = StandardScaler()
    scaler.fit(X_seq[train_idx].reshape(-1, n_inputs))
    X_scaled = scaler.transform(X_seq.reshape(-1, n_inputs)).reshape(
        n_rows, n_steps, n_inputs
    )

    W_in, W = make_esn_weights(
        n_inputs=n_inputs,
        n_reservoir=config.n_reservoir,
        spectral_radius=config.spectral_radius,
        input_scale=config.input_scale,
        seed=seed,
    )
    features = esn_states(X_scaled, W_in, W, config.leak)
    return features[train_idx], features[test_idx]


def evaluate_config(
    frame: pd.DataFrame,
    X_seq: np.ndarray,
    honest_d1: np.ndarray,
    config: ESNConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    for i, row in frame.iterrows():
        if row["landmark_date"] < EVAL_START:
            continue

        train_mask = frame["landmark_date"] < row["cluster_start"]
        train_idx = np.flatnonzero(train_mask.to_numpy())
        if len(train_idx) < MIN_TRAIN:
            continue

        train = frame.iloc[train_idx]
        if train["y_recovery"].nunique() < 2:
            continue

        test_idx = np.array([i], dtype=int)
        y_train = train["y_recovery"].to_numpy(int)

        d1_model = fit_d1(train)
        p_d1_test = float(
            d1_model.predict_proba(frame.loc[[i], D1].to_numpy(float))[0, 1]
        )

        out = {
            "config": config.name,
            "row_id": int(i),
            "market_key": row["market_key"],
            "episode_id": int(row["episode_id"]),
            "cluster_id": row["cluster_id"],
            "landmark_date": row["landmark_date"],
            "y": int(row["y_recovery"]),
            "D1": p_d1_test,
        }

        direct_seed_preds = []
        joint_seed_preds = []
        offset_seed_preds = []

        for seed in config.seeds:
            F_train, F_test = reservoir_features_for_fold(
                X_seq, train_idx, test_idx, config, seed
            )

            direct = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logit",
                        LogisticRegression(
                            C=0.1,
                            max_iter=5000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            direct.fit(F_train, y_train)
            direct_seed_preds.append(float(direct.predict_proba(F_test)[0, 1]))

            joint_X_train = np.column_stack(
                [train[D1].to_numpy(float), F_train]
            )
            joint_X_test = np.column_stack(
                [frame.loc[[i], D1].to_numpy(float), F_test]
            )
            joint = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logit",
                        LogisticRegression(
                            C=0.1,
                            max_iter=5000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            joint.fit(joint_X_train, y_train)
            joint_seed_preds.append(float(joint.predict_proba(joint_X_test)[0, 1]))

            valid = np.isfinite(honest_d1[train_idx])
            if valid.sum() >= MIN_TRAIN and np.unique(y_train[valid]).size == 2:
                corr_scaler = StandardScaler()
                F_corr_train = corr_scaler.fit_transform(F_train[valid])
                F_corr_test = corr_scaler.transform(F_test)
                beta, intercept = fit_offset_logistic(
                    X=F_corr_train,
                    y=y_train[valid],
                    offset=logit(honest_d1[train_idx][valid]),
                    l2=config.correction_l2,
                )
                eta_test = (
                    float(logit(p_d1_test))
                    + intercept
                    + float(F_corr_test @ beta)
                )
                offset_seed_preds.append(float(sigmoid(eta_test)))

        out["ESN_direct"] = float(np.mean(direct_seed_preds))
        out["D1_plus_ESN_joint"] = float(np.mean(joint_seed_preds))
        out["D1_plus_ESN_offset"] = (
            float(np.mean(offset_seed_preds)) if offset_seed_preds else np.nan
        )
        rows.append(out)

    return pd.DataFrame(rows)


def score_group(group: pd.DataFrame, model: str) -> dict:
    use = group[["y", model]].dropna()
    y = use["y"].to_numpy(int)
    p = clip_prob(use[model].to_numpy(float))
    return {
        "model": model,
        "n": int(len(use)),
        "recovery_rate": float(y.mean()) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc": (
            float(average_precision_score(y, p))
            if np.unique(y).size == 2
            else np.nan
        ),
        "logloss": float(log_loss(y, p)) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }


def cluster_weighted(group: pd.DataFrame, model: str) -> dict:
    details = []
    for cluster_id, g in group.dropna(subset=[model]).groupby("cluster_id"):
        y = g["y"].to_numpy(int)
        p = clip_prob(g[model].to_numpy(float))
        details.append(
            {
                "cluster_id": cluster_id,
                "n": len(g),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(np.mean((p - y) ** 2)),
            }
        )
    detail = pd.DataFrame(details)
    return {
        "model": model,
        "n_clusters": int(len(detail)),
        "cluster_mean_logloss": float(detail["logloss"].mean()),
        "cluster_mean_brier": float(detail["brier"].mean()),
        "median_cluster_size": float(detail["n"].median()),
    }


def summarize(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = ["D1", "ESN_direct", "D1_plus_ESN_joint", "D1_plus_ESN_offset"]
    groups = {
        "all_post1990_purged": preds,
        "validation_non_spy": preds[preds["market_key"] != "spy"],
        "leave_nikkei_out_eval": preds[preds["market_key"] != "nikkei_225"],
        "nikkei_only_eval": preds[preds["market_key"] == "nikkei_225"],
    }

    metric_rows = []
    cluster_rows = []
    for group_name, group in groups.items():
        for model in models:
            row = score_group(group, model)
            row["group"] = group_name
            metric_rows.append(row)

            crow = cluster_weighted(group, model)
            crow["group"] = group_name
            cluster_rows.append(crow)

    metrics = pd.DataFrame(metric_rows)[
        ["group", "model", "n", "recovery_rate", "auc", "pr_auc", "logloss", "brier"]
    ]
    cluster_metrics = pd.DataFrame(cluster_rows)[
        [
            "group",
            "model",
            "n_clusters",
            "cluster_mean_logloss",
            "cluster_mean_brier",
            "median_cluster_size",
        ]
    ]
    return metrics, cluster_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-panel",
        type=Path,
        default=Path("scratch/path_panel_day0_5.csv"),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path(
            "results/diagnostics/cross_market_crisis_clusters_v2/"
            "branch_sync_cluster_detail.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/tmp/qpitome_branch_esn_probe"),
    )
    parser.add_argument(
        "--configs",
        choices=["primary", "both"],
        default="primary",
        help="Run the primary configuration only, or primary plus one sensitivity.",
    )
    args = parser.parse_args()

    if not args.path_panel.exists():
        raise FileNotFoundError(
            f"Missing {args.path_panel}. Generate the exploratory path panel first."
        )
    if not args.clusters.exists():
        raise FileNotFoundError(f"Missing {args.clusters}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.path_panel, args.clusters)
    X_seq = make_sequence_tensor(frame)
    honest_d1 = compute_honest_d1_predictions(frame)

    configs = [PRIMARY]
    if args.configs == "both":
        configs.append(SENSITIVITY)

    all_preds = []
    for config in configs:
        print(f"\nRunning {config.name}: {asdict(config)}")
        preds = evaluate_config(frame, X_seq, honest_d1, config)
        all_preds.append(preds)

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics, cluster_metrics = summarize(predictions)

    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    metrics.to_csv(args.outdir / "summary_metrics.csv", index=False)
    cluster_metrics.to_csv(args.outdir / "cluster_weighted_metrics.csv", index=False)

    manifest = {
        "purpose": "Exploratory bounded ESN assessment for day-5 branch direction",
        "path_panel": str(args.path_panel),
        "clusters": str(args.clusters),
        "minimum_training_rows": MIN_TRAIN,
        "evaluation_start": str(EVAL_START.date()),
        "d1_features": D1,
        "sequence_channels": CHANNELS,
        "configs": [asdict(c) for c in configs],
        "models": [
            "D1 locked geometry",
            "ESN_direct",
            "D1_plus_ESN_joint",
            "D1_plus_ESN_offset with honest prequential D1 training offsets",
        ],
        "status": "exploratory; do not promote without review",
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list)
    )

    print("\nSummary metrics")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster-weighted metrics")
    print(
        cluster_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nSaved exploratory outputs to {args.outdir}")


if __name__ == "__main__":
    main()
