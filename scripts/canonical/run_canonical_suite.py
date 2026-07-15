#!/usr/bin/env python3
"""Combine completed Stage 1 baseline runs into one canonical comparison.

This script does not retrain models. It reads the standardized classical master
outputs plus the standalone LSTM and GARCH outputs, normalizes their schemas,
and writes one unified comparison directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MASTER_SCRIPT = SCRIPT_DIR / "run_master_comparison.py"


def load_master_module():
    spec = importlib.util.spec_from_file_location("canonical_master", MASTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="stage1_7fold")
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=REPO_ROOT / "results/canonical/run_master_comparison",
    )
    parser.add_argument(
        "--lstm-dir",
        type=Path,
        default=REPO_ROOT / "results/baselines/lstm/run_phase3_lstm_walkforward",
    )
    parser.add_argument(
        "--garch-dir",
        type=Path,
        default=REPO_ROOT / "results/baselines/garch/run_phase3_garch_walkforward",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/canonical/run_canonical_suite",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required completed result file not found: {path}")
    return path


def normalize_lstm_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("LSTM metrics contain no test rows")
    test["protocol"] = "classical"
    rename = {
        "rmse": "test_rmse",
        "qlike": "test_qlike",
        "mz_alpha": "test_mz_alpha",
        "mz_beta": "test_mz_beta",
        "mz_r2": "test_mz_r2",
    }
    test = test.rename(columns=rename)
    for column in (
        "q90_test_ap",
        "q90_test_auc",
        "q90_test_f1",
        "q90_test_precision",
        "q90_test_recall",
        "q90_test_called_rate",
        "q95_test_ap",
        "q95_test_auc",
        "q95_test_f1",
        "q95_test_precision",
        "q95_test_recall",
        "q95_test_called_rate",
    ):
        test[column] = np.nan
    return test


def normalize_garch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[(frame["split"] == "test") & (frame["task"] == "level")].copy()
    if test.empty:
        raise ValueError("GARCH metrics contain no level-task test rows")
    test["protocol"] = "classical"
    rename = {
        "rmse": "test_rmse",
        "qlike": "test_qlike",
        "mz_alpha": "test_mz_alpha",
        "mz_beta": "test_mz_beta",
        "mz_r2": "test_mz_r2",
    }
    test = test.rename(columns=rename)
    for column in (
        "q90_test_ap",
        "q90_test_auc",
        "q90_test_f1",
        "q90_test_precision",
        "q90_test_recall",
        "q90_test_called_rate",
        "q95_test_ap",
        "q95_test_auc",
        "q95_test_f1",
        "q95_test_precision",
        "q95_test_recall",
        "q95_test_called_rate",
    ):
        test[column] = np.nan
    return test


def normalize_lstm_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["protocol"] = "classical"
    output["log_score"] = np.log(np.clip(output["y_pred"].to_numpy(float), 1e-12, None))
    output["q90_label"] = np.nan
    output["q95_label"] = np.nan
    return output[
        [
            "fold",
            "model",
            "protocol",
            "split",
            "date",
            "y_true",
            "log_score",
            "y_pred",
            "q90_label",
            "q95_label",
        ]
    ]


def normalize_garch_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[frame["task"] == "level"].copy()
    output["protocol"] = "classical"
    output["y_true"] = output["future_rv_true"]
    output["y_pred"] = output["predicted_future_rv"]
    output["log_score"] = np.log(np.clip(output["y_pred"].to_numpy(float), 1e-12, None))
    output["q90_label"] = np.nan
    output["q95_label"] = np.nan
    return output[
        [
            "fold",
            "model",
            "protocol",
            "split",
            "date",
            "y_true",
            "log_score",
            "y_pred",
            "q90_label",
            "q95_label",
        ]
    ]


def main() -> None:
    args = parse_args()
    tag = args.tag

    master_metrics_path = require_file(args.master_dir / f"per_fold_metrics_{tag}.csv")
    master_predictions_path = require_file(args.master_dir / f"predictions_{tag}.csv")
    lstm_metrics_path = require_file(args.lstm_dir / f"lstm_metrics_{tag}.csv")
    lstm_predictions_path = require_file(args.lstm_dir / f"lstm_predictions_{tag}.csv")
    garch_metrics_path = require_file(args.garch_dir / f"garch_metrics_{tag}.csv")
    garch_predictions_path = require_file(args.garch_dir / f"garch_predictions_{tag}.csv")

    master_metrics = pd.read_csv(master_metrics_path)
    master_predictions = pd.read_csv(master_predictions_path)
    lstm_metrics = normalize_lstm_metrics(pd.read_csv(lstm_metrics_path))
    garch_metrics = normalize_garch_metrics(pd.read_csv(garch_metrics_path))
    lstm_predictions = normalize_lstm_predictions(pd.read_csv(lstm_predictions_path))
    garch_predictions = normalize_garch_predictions(pd.read_csv(garch_predictions_path))

    combined_metrics = pd.concat(
        [master_metrics, lstm_metrics, garch_metrics],
        ignore_index=True,
        sort=False,
    )
    combined_predictions = pd.concat(
        [master_predictions, lstm_predictions, garch_predictions],
        ignore_index=True,
        sort=False,
    )

    master = load_master_module()
    aggregate = master.aggregate_metrics(combined_metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = args.out_dir / f"per_fold_metrics_{tag}.csv"
    predictions_path = args.out_dir / f"predictions_{tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{tag}.json"

    combined_metrics.to_csv(per_fold_path, index=False)
    combined_predictions.to_csv(predictions_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "mode": "aggregation_only",
                "sources": {
                    "master_metrics": str(master_metrics_path),
                    "master_predictions": str(master_predictions_path),
                    "lstm_metrics": str(lstm_metrics_path),
                    "lstm_predictions": str(lstm_predictions_path),
                    "garch_metrics": str(garch_metrics_path),
                    "garch_predictions": str(garch_predictions_path),
                },
                "models": sorted(combined_metrics["model"].dropna().unique().tolist()),
                "classification_note": (
                    "GARCH and LSTM standalone runners do not currently emit the "
                    "canonical q90/q95 classification fields; those aggregate fields "
                    "remain NaN for these two models."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
