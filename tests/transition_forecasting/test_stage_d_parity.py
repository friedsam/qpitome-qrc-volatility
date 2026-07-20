from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/transition_forecasting/validate_stage_d_parity.py"
)
SPEC = importlib.util.spec_from_file_location("validate_stage_d_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["P_E1_IDX_L1", "N_P_E1_IDX_L1_1"],
            "label": [1, 0],
            "index": ["IDX", "IDX"],
            "episode_id": ["E1", "E1"],
            "split": ["train", "train"],
            "event_onset": ["2010-01-05", "2010-01-05"],
            "origin_date": ["2010-01-04", "2009-12-15"],
            "matched_positive_id": [None, "P_E1_IDX_L1"],
            "match_distance": [np.nan, 0.25],
        }
    )


def _write_npz(path: Path, ids: list[str], x: np.ndarray) -> None:
    np.savez_compressed(path, sample_id=np.asarray(ids), X=x)


def test_manifest_parity_passes_for_identical_content_in_different_row_order(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    rebuilt = tmp_path / "rebuilt.csv"
    frame = _manifest()
    frame.to_csv(reference, index=False)
    frame.iloc[::-1].to_csv(rebuilt, index=False)

    result = MODULE.compare_manifests(reference, rebuilt)

    assert result["passed"] is True
    assert result["same_sample_id_set"] is True
    assert result["mismatch_counts"] == {}


def test_manifest_parity_reports_changed_value(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    rebuilt = tmp_path / "rebuilt.csv"
    frame = _manifest()
    frame.to_csv(reference, index=False)
    changed = frame.copy()
    changed.loc[changed["label"] == 0, "match_distance"] = 0.5
    changed.to_csv(rebuilt, index=False)

    result = MODULE.compare_manifests(reference, rebuilt)

    assert result["passed"] is False
    assert result["mismatch_counts"] == {"match_distance": 1}


def test_tensor_parity_aligns_rows_by_sample_id(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    rebuilt = tmp_path / "rebuilt.npz"
    ids = ["a", "b"]
    x = np.arange(8, dtype=float).reshape(2, 4, 1)
    _write_npz(reference, ids, x)
    _write_npz(rebuilt, list(reversed(ids)), x[::-1])

    result = MODULE.compare_tensors(reference, rebuilt)

    assert result["passed"] is True
    assert result["numeric_match"] is True
    assert result["max_abs_difference"] == 0.0


def test_tensor_parity_rejects_numeric_change(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    rebuilt = tmp_path / "rebuilt.npz"
    ids = ["a", "b"]
    x = np.arange(8, dtype=float).reshape(2, 4, 1)
    changed = x.copy()
    changed[1, 2, 0] += 0.01
    _write_npz(reference, ids, x)
    _write_npz(rebuilt, ids, changed)

    result = MODULE.compare_tensors(reference, rebuilt)

    assert result["passed"] is False
    assert result["numeric_match"] is False
    assert result["max_abs_difference"] == 0.01
