from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_manifest(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "sample_manifest.csv")
    for column in ("event_onset", "origin_date", "input_start_date", "target_end_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _positive_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["index"].astype(str)
        + "|"
        + frame["event_onset"].dt.strftime("%Y-%m-%d")
        + "|"
        + frame["lead"].astype(int).astype(str)
        + "|"
        + frame["split"].astype(str)
    )


def _control_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["index"].astype(str)
        + "|"
        + frame["origin_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + frame["lead"].astype(int).astype(str)
        + "|"
        + frame["split"].astype(str)
    )


def _catalogue_key(frame: pd.DataFrame) -> pd.Series:
    return frame["index"].astype(str) + "|" + pd.to_datetime(frame["onset_date"]).dt.strftime("%Y-%m-%d")


def _set_delta(left: set[str], right: set[str]) -> dict[str, object]:
    return {
        "shared": len(left & right),
        "removed": len(left - right),
        "added": len(right - left),
        "removed_keys": sorted(left - right),
        "added_keys": sorted(right - left),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare corrected build with exact GPT-2 parity baseline.")
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/corrected_vs_parity_audit.json"),
    )
    args = parser.parse_args()

    parity_manifest = _load_manifest(args.parity_dir)
    corrected_manifest = _load_manifest(args.corrected_dir)
    parity_catalogue = pd.read_csv(args.parity_dir / "transition_catalogue.csv")
    corrected_catalogue = pd.read_csv(args.corrected_dir / "transition_catalogue.csv")
    corrections = pd.read_csv(args.corrected_dir / "row_corrections.csv")

    parity_pos = parity_manifest.loc[parity_manifest["label"] == 1].copy()
    corrected_pos = corrected_manifest.loc[corrected_manifest["label"] == 1].copy()
    parity_ctl = parity_manifest.loc[parity_manifest["label"] == 0].copy()
    corrected_ctl = corrected_manifest.loc[corrected_manifest["label"] == 0].copy()

    parity_pos_keys = set(_positive_key(parity_pos))
    corrected_pos_keys = set(_positive_key(corrected_pos))
    parity_ctl_keys = set(_control_key(parity_ctl))
    corrected_ctl_keys = set(_control_key(corrected_ctl))
    parity_event_keys = set(_catalogue_key(parity_catalogue))
    corrected_event_keys = set(_catalogue_key(corrected_catalogue))

    structural = corrections.loc[
        corrections.get("reason", pd.Series(dtype=str)).astype(str).eq("structural_bad_print")
    ].copy()

    parity_npz = np.load(args.parity_dir / "sequence_tensors.npz", allow_pickle=False)
    corrected_npz = np.load(args.corrected_dir / "sequence_tensors.npz", allow_pickle=False)

    report = {
        "parity_counts": {
            "events": len(parity_catalogue),
            "samples": len(parity_manifest),
            "positives": len(parity_pos),
            "controls": len(parity_ctl),
        },
        "corrected_counts": {
            "events": len(corrected_catalogue),
            "samples": len(corrected_manifest),
            "positives": len(corrected_pos),
            "controls": len(corrected_ctl),
        },
        "structural_corrections": {
            "rows": int(len(structural)),
            "indices": sorted(structural["index"].astype(str).unique().tolist()) if not structural.empty else [],
            "dates": sorted(pd.to_datetime(structural["date"]).dt.strftime("%Y-%m-%d").tolist()) if not structural.empty else [],
        },
        "catalogue_delta": _set_delta(parity_event_keys, corrected_event_keys),
        "positive_delta": _set_delta(parity_pos_keys, corrected_pos_keys),
        "control_delta": _set_delta(parity_ctl_keys, corrected_ctl_keys),
        "tensor_shapes": {
            "parity": list(parity_npz["X"].shape),
            "corrected": list(corrected_npz["X"].shape),
        },
        "invariants": {
            "corrected_controls_per_positive": (
                int(len(corrected_ctl)) == 3 * int(len(corrected_pos))
            ),
            "corrected_tensor_rows_match_manifest": (
                int(corrected_npz["X"].shape[0]) == int(len(corrected_manifest))
            ),
            "corrected_tensor_shape_40x1": (
                list(corrected_npz["X"].shape[1:]) == [40, 1]
            ),
            "corrected_episode_split_exclusive": (
                corrected_manifest.groupby("episode_id")["split"].nunique().max() == 1
            ),
        },
    }

    report["passed"] = bool(
        report["structural_corrections"]["rows"] == 12
        and len(report["structural_corrections"]["indices"]) == 2
        and all(report["invariants"].values())
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
