#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from transition_forecasting.qrc.stratified_control_metrics import (
    build_stratified_result_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach frozen control strata and write calm/hard result metrics."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.result_dir / "predictions.csv.gz"
    manifest_path = args.fold_dir / "rematched_rolling_manifest.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing predictions: {predictions_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing fold manifest: {manifest_path}")

    predictions = pd.read_csv(predictions_path)
    manifest = pd.read_csv(manifest_path)
    linked, pooled, by_lead = build_stratified_result_tables(
        predictions,
        manifest,
    )
    linked.to_csv(
        args.result_dir / "predictions_with_control_strata.csv.gz",
        index=False,
        compression="gzip",
    )
    pooled.to_csv(args.result_dir / "control_stratum_metrics.csv", index=False)
    by_lead.to_csv(
        args.result_dir / "control_stratum_metrics_by_lead.csv",
        index=False,
    )
    print(f"WROTE {args.result_dir / 'control_stratum_metrics.csv'}")


if __name__ == "__main__":
    main()
