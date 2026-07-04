#!/usr/bin/env python3
"""Canonical purged walk-forward evaluation of the frozen Phase 2 final TFIM-QRC.

The configuration is the final Phase 2 model, not a new search:
- 6 qubits / PCA6
- 40-day lookback
- 10 recent-biased anchors
- Z/X/ZZ readout collected across anchors
- full ZZ topology
- 3 Trotter steps per anchor
- 3 virtual nodes per anchor
- fixed disorder strength 0.20
- ridge alpha 1000 on log future RV

Outputs use the same schema as the canonical master comparison.
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
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    build_qrc_feature_matrix,
    make_qrc_sequence_splits,
)

MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")
TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")


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


def main() -> None:
    args = parse_args()
    master = load_master()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    pred_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    if per_fold_path.exists() and pred_path.exists() and aggregate_path.exists() and not args.force:
        print(f"TFIM segment already complete: {args.out_dir}")
        return

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = master.make_folds(
        len(df), n_folds=args.n_folds, min_train=args.min_train,
        val_size=args.val_size, purge=args.purge,
    )
    if args.only_folds:
        keep = set(args.only_folds)
        folds = [f for f in folds if f["fold"] in keep]

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
        ridge_alpha=1000.0,
        target_transform="log",
        seed=42,
        collect_anchor_features=True,
        use_disorder=True,
        disorder_strength=0.20,
    )

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold in folds:
        fold_id = fold["fold"]
        print(f"\n=== Canonical TFIM fold {fold_id} ===")
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

        seq = make_qrc_sequence_splits(
            pca_frames,
            feature_columns=pca_cols,
            target_column=TARGET,
            lookback_days=config.lookback_days,
        )
        y = {s: np.asarray(seq[s][1], dtype=float) for s in SPLITS}
        dates = {s: np.asarray(seq[s][2]) for s in SPLITS}
        q90_target, lab90 = master.labels_for(y["train"], y, 0.90)
        q95_target, lab95 = master.labels_for(y["train"], y, 0.95)

        start = time.perf_counter()
        features = {s: build_qrc_feature_matrix(seq[s][0], config, verbose=(s == "train")) for s in SPLITS}
        feature_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores = master.fit_log_ridge(features, y, config.ridge_alpha)
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
                "selected_config": json.dumps(config.__dict__, sort_keys=True, default=str),
                "selection_metric": "frozen_phase2_final",
            },
        )
        metric_rows.append(row)
        prediction_rows.extend(preds)

        # Atomic checkpoint after every completed fold.
        pd.DataFrame(metric_rows).to_csv(per_fold_path.with_suffix(".tmp"), index=False)
        per_fold_path.with_suffix(".tmp").replace(per_fold_path)
        pd.DataFrame(prediction_rows).to_csv(pred_path.with_suffix(".tmp"), index=False)
        pred_path.with_suffix(".tmp").replace(pred_path)

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = master.aggregate_metrics(per_fold)
    per_fold.to_csv(per_fold_path, index=False)
    predictions.to_csv(pred_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_path.write_text(json.dumps({
        "model": "tfim_phase2_final",
        "protocol": "exact",
        "folds": [f["fold"] for f in folds],
        "configuration": config.__dict__,
        "checkpoint_policy": "atomic CSV write after every completed fold",
    }, indent=2, default=str))

    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
