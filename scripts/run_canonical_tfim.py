#!/usr/bin/env python3
"""Canonical purged walk-forward evaluation of the exact frozen Phase 2 final TFIM-QRC.

Source-faithful model lineage:
- full FEATURE_COLUMNS -> train-only StandardScaler -> PCA6
- 40-day windows
- leaky-integrated input, leak=0.3
- clipped-linear angle encoding (repository default)
- 10 recent anchors
- 6-qubit full-topology TFIM-QRC
- 3 Trotter steps per anchor
- 3 virtual nodes per anchor
- ZXZZ observables collected across anchors (459 raw features)
- fixed disorder strength 0.20
- train-only 1st/99th percentile feature clipping
- train-only absolute feature-target correlation ranking
- top 240 features
- train-only StandardScaler
- Ridge(alpha=1000) on log future RV

This reproduces ``linear_clip_top240_alpha1000`` under the common Phase 3
purged walk-forward protocol. No TFIM tuning is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    _safe_feature_target_correlations,
    build_qrc_feature_matrix,
    make_qrc_sequence_splits,
)

MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")
TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")
LEAK = 0.3
TOP_K = 240
READOUT_ALPHA = 1000.0


def load_master():
    spec = importlib.util.spec_from_file_location("canonical_master_helpers", MASTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/canonical/segments/tfim_phase2_final"))
    p.add_argument("--tag", default="tfim_phase2_final")
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def leaky_integrate_windows(X: np.ndarray, leak: float = LEAK) -> np.ndarray:
    out = np.empty_like(X, dtype=float)
    for i, window in enumerate(X):
        h = np.zeros(window.shape[1], dtype=float)
        for t, u_t in enumerate(window):
            h = (1.0 - leak) * h + leak * u_t
            out[i, t] = h
    return out


def exact_phase2_readout(features: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Exact ``linear_clip_top240_alpha1000`` readout from the Phase 2 notebook."""
    lower = np.percentile(features["train"], 1.0, axis=0)
    upper = np.percentile(features["train"], 99.0, axis=0)
    clipped = {s: np.clip(features[s], lower, upper) for s in SPLITS}

    corr = _safe_feature_target_correlations(clipped["train"], y["train"])
    k = min(TOP_K, clipped["train"].shape[1])
    selected_idx = np.argsort(np.abs(corr))[-k:]
    selected = {s: clipped[s][:, selected_idx] for s in SPLITS}

    scaler = StandardScaler()
    H_train = scaler.fit_transform(selected["train"])
    H_val = scaler.transform(selected["val"])
    H_test = scaler.transform(selected["test"])

    model = Ridge(alpha=READOUT_ALPHA)
    model.fit(H_train, np.log(np.maximum(y["train"], 1e-8)))
    scores = {
        "train": model.predict(H_train),
        "val": model.predict(H_val),
        "test": model.predict(H_test),
    }
    metadata = {
        "n_raw_features": int(features["train"].shape[1]),
        "n_selected_features": int(k),
        "selected_feature_indices": selected_idx.tolist(),
        "clip_percentiles": [1.0, 99.0],
        "feature_selection": "train_abs_feature_target_correlation",
        "ridge_alpha": READOUT_ALPHA,
        "target_transform": "log",
    }
    return scores, metadata


def main() -> None:
    args = parse_args()
    master = load_master()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    pred_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    requested_folds = set(args.only_folds or range(1, args.n_folds + 1))
    completed_folds: set[int] = set()
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    if not args.force and per_fold_path.exists() and pred_path.exists():
        existing_metrics = pd.read_csv(per_fold_path)
        existing_predictions = pd.read_csv(pred_path)
        if not existing_metrics.empty:
            completed_folds = set(existing_metrics["fold"].astype(int)) & requested_folds
            metric_rows = existing_metrics[existing_metrics["fold"].astype(int).isin(completed_folds)].to_dict("records")
            prediction_rows = existing_predictions[existing_predictions["fold"].astype(int).isin(completed_folds)].to_dict("records")
            if completed_folds:
                print(f"Resuming TFIM segment; completed folds={sorted(completed_folds)}")

    if completed_folds == requested_folds and aggregate_path.exists() and manifest_path.exists() and not args.force:
        print(f"TFIM segment already complete: {args.out_dir}")
        return

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = master.make_folds(
        len(df), n_folds=args.n_folds, min_train=args.min_train,
        val_size=args.val_size, purge=args.purge,
    )
    folds = [f for f in folds if f["fold"] in requested_folds]

    config = TFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=40,
        anchor_count=10,
        anchor_policy="recent",
        observable_mode="zxzz",
        trotter_steps_per_anchor=3,
        virtual_nodes_per_anchor=3,
        topology="full",
        coupling_scale=0.7,
        transverse_field=0.5,
        evolution_time=0.5,
        angle_max=np.pi / 2,
        ridge_alpha=READOUT_ALPHA,
        target_transform="log",
        seed=42,
        collect_anchor_features=True,
        use_disorder=True,
        disorder_strength=0.20,
    )

    for fold in folds:
        fold_id = fold["fold"]
        if fold_id in completed_folds and not args.force:
            print(f"SKIP completed TFIM fold {fold_id}")
            continue

        print(f"\n=== Canonical exact Phase 2 TFIM fold {fold_id} ===")
        frames = {s: df.iloc[fold[s][0] : fold[s][1]].copy().reset_index(drop=True) for s in SPLITS}

        scaler = StandardScaler()
        pca = PCA(n_components=config.pca_components, random_state=42)
        pca.fit(scaler.fit_transform(frames["train"][FEATURE_COLUMNS]))
        pca_cols = [f"pca{i+1}" for i in range(config.pca_components)]

        pca_frames = {}
        for split in SPLITS:
            z = pca.transform(scaler.transform(frames[split][FEATURE_COLUMNS]))
            frame = pd.DataFrame(z, columns=pca_cols)
            frame[TARGET] = frames[split][TARGET].to_numpy()
            frame["date"] = frames[split]["date"].to_numpy()
            pca_frames[split] = frame

        raw_seq = make_qrc_sequence_splits(
            pca_frames,
            feature_columns=pca_cols,
            target_column=TARGET,
            lookback_days=config.lookback_days,
        )
        seq = {
            s: (leaky_integrate_windows(raw_seq[s][0], leak=LEAK), raw_seq[s][1], raw_seq[s][2])
            for s in SPLITS
        }
        y = {s: np.asarray(seq[s][1], dtype=float) for s in SPLITS}
        dates = {s: np.asarray(seq[s][2]) for s in SPLITS}
        q90_target, lab90 = master.labels_for(y["train"], y, 0.90)
        q95_target, lab95 = master.labels_for(y["train"], y, 0.95)

        start = time.perf_counter()
        features = {s: build_qrc_feature_matrix(seq[s][0], config, verbose=(s == "train")) for s in SPLITS}
        feature_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores, readout_meta = exact_phase2_readout(features, y)
        fit_seconds = time.perf_counter() - start

        row, preds = master.evaluate_model(
            model_name="tfim_phase2_final",
            protocol="exact",
            fold_id=fold_id,
            y=y,
            dates=dates,
            scores=scores,
            q90_threshold=q90_target,
            lab90=lab90,
            q95_threshold=q95_target,
            lab95=lab95,
            metadata={
                "quantum_tasks_per_date": 1,
                "shots": np.nan,
                "shot_seed": np.nan,
                "fit_seconds": fit_seconds,
                "feature_seconds": feature_seconds,
                "selected_config": json.dumps({
                    **config.__dict__,
                    "input_leak": LEAK,
                    **readout_meta,
                }, sort_keys=True, default=str),
                "selection_metric": "frozen_phase2_linear_clip_top240_alpha1000",
                "n_raw_features": readout_meta["n_raw_features"],
                "n_selected_features": readout_meta["n_selected_features"],
            },
        )

        metric_rows = [r for r in metric_rows if int(r["fold"]) != fold_id] + [row]
        prediction_rows = [r for r in prediction_rows if int(r["fold"]) != fold_id] + preds
        atomic_csv(pd.DataFrame(metric_rows).sort_values("fold"), per_fold_path)
        atomic_csv(pd.DataFrame(prediction_rows).sort_values(["fold", "split", "date"]), pred_path)
        print(f"Checkpointed TFIM fold {fold_id}")

    per_fold = pd.DataFrame(metric_rows).sort_values("fold").reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold", "split", "date"]).reset_index(drop=True)
    aggregate = master.aggregate_metrics(per_fold)
    atomic_csv(per_fold, per_fold_path)
    atomic_csv(predictions, pred_path)
    atomic_csv(aggregate, aggregate_path)
    manifest_path.write_text(json.dumps({
        "model": "tfim_phase2_final",
        "historical_source": "phase2-volatility-regression-qrc:notebooks/phase2_qrc_final_encoding_readout_probe.ipynb",
        "historical_run_name": "linear_clip_top240_alpha1000",
        "protocol": "exact",
        "folds": sorted(requested_folds),
        "configuration": config.__dict__,
        "input_leak": LEAK,
        "readout": {
            "train_clip_percentiles": [1.0, 99.0],
            "feature_selection": "top 240 by absolute train feature-target correlation",
            "scaler": "StandardScaler fit on selected train features",
            "ridge_alpha": READOUT_ALPHA,
            "target_transform": "log",
        },
        "checkpoint_policy": "atomic CSV write after every completed fold; completed folds skipped on restart",
    }, indent=2, default=str))

    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
