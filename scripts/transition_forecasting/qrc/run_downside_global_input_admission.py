from __future__ import annotations

import argparse
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.downside_global_input_admission import (
    DownsideGlobalAdmissionConfig,
    run_downside_global_input_admission,
)
from transition_forecasting.qrc.input_admission_assay import (
    HAR_COLUMNS,
    PREQ_COLUMNS,
    TARGET_COLUMNS,
)

DEFAULT_PANEL = Path(
    "data/fallback/transition_forecasting/"
    "global_stock_indices_historical_data/all_indices_data.csv"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_downside_global_input_admission"
)


def evaluation_frame_from_handoff_zip(path: Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    blocks: list[pd.DataFrame] = []
    with zipfile.ZipFile(source) as bundle:
        archives = sorted(
            name
            for name in bundle.namelist()
            if "/feature_archives/ladder_symmetric_modes_fold_" in name
            and name.endswith(".npz")
        )
        if not archives:
            raise ValueError("handoff ZIP contains no ladder feature archives")
        for name in archives:
            with np.load(io.BytesIO(bundle.read(name)), allow_pickle=True) as archive:
                required = {
                    "fold",
                    "sample_id",
                    "fold_split",
                    "lead",
                    "label",
                    "origin_date",
                    "target_path",
                    "har_prediction_path",
                    "prequential_residual_path",
                    "prequential_residual_valid",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(f"{name} missing arrays: {sorted(missing)}")
                target = np.asarray(archive["target_path"], dtype=float)
                har = np.asarray(archive["har_prediction_path"], dtype=float)
                residual = np.asarray(archive["prequential_residual_path"], dtype=float)
                if target.ndim != 2 or target.shape[1] != 10:
                    raise ValueError(f"{name}: target_path must have shape (rows, 10)")
                if har.shape != target.shape or residual.shape != target.shape:
                    raise ValueError(f"{name}: forecast arrays are not aligned")
                frame = pd.DataFrame(
                    {
                        "fold": np.asarray(archive["fold"], dtype=int),
                        "sample_id": np.asarray(archive["sample_id"]).astype(str),
                        "fold_split": np.asarray(archive["fold_split"]).astype(str),
                        "lead": np.asarray(archive["lead"], dtype=int),
                        "label": np.asarray(archive["label"], dtype=int),
                        "origin_date": np.asarray(archive["origin_date"]).astype(str),
                        "prequential_valid": np.asarray(
                            archive["prequential_residual_valid"], dtype=bool
                        ),
                    }
                )
                for horizon in range(10):
                    frame[TARGET_COLUMNS[horizon]] = target[:, horizon]
                    frame[HAR_COLUMNS[horizon]] = har[:, horizon]
                    frame[PREQ_COLUMNS[horizon]] = residual[:, horizon]
                blocks.append(frame)
    output = pd.concat(blocks, ignore_index=True)
    if output["fold_split"].eq("test").any():
        raise RuntimeError("handoff archive unexpectedly contains test rows")
    if output.duplicated(["fold", "sample_id"]).any():
        raise ValueError("handoff archive contains duplicate fold/sample_id rows")
    return output.sort_values(["fold", "sample_id"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test market-local downside state and strictly lagged leave-one-out "
            "global breadth on the exact frozen L5 handoff rows."
        )
    )
    parser.add_argument("--handoff-zip", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--minimum-peer-markets", type=int, default=8)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out_root / args.run_id
    frame = evaluation_frame_from_handoff_zip(args.handoff_zip)
    with tempfile.TemporaryDirectory(prefix="downside_global_admission_") as temporary:
        manifest = Path(temporary) / "evaluation_manifest.csv"
        frame.to_csv(manifest, index=False)
        output = run_downside_global_input_admission(
            evaluation_manifest=manifest,
            panel_path=args.panel,
            output_dir=output_dir,
            config=DownsideGlobalAdmissionConfig(
                folds=tuple(args.folds),
                minimum_peer_markets=args.minimum_peer_markets,
            ),
        )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
