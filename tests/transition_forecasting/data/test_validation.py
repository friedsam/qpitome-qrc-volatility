from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.validation import audit_processed_dataset


def _write_dataset(root: Path, *, tensor_ids: list[str] | None = None, controls: int = 3) -> None:
    root.mkdir()
    pd.DataFrame(
        {
            "index": ["IDX", "IDX"],
            "date": ["2020-01-01", "2020-01-02"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    ).to_csv(root / "cleaned_ohlc.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "index": ["IDX", "IDX"],
            "date": ["2020-01-01", "2020-01-02"],
            "log_parkinson_volatility": [-4.0, -3.9],
        }
    ).to_csv(root / "daily_volatility.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"episode_id": ["GE001"], "onset_date": ["2020-02-01"]}).to_csv(
        root / "transition_catalogue.csv", index=False
    )

    rows = [
        {
            "sample_id": "P1",
            "label": 1,
            "index": "IDX",
            "episode_id": "GE001",
            "split": "train",
            "lead": 1,
            "matched_positive_id": np.nan,
            "origin_date": "2020-01-20",
        }
    ]
    for number in range(controls):
        rows.append(
            {
                "sample_id": f"N{number + 1}",
                "label": 0,
                "index": "IDX",
                "episode_id": "GE001",
                "split": "train",
                "lead": 1,
                "matched_positive_id": "P1",
                "origin_date": f"2019-12-{number + 10:02d}",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "sample_manifest.csv", index=False)

    ids = tensor_ids or manifest["sample_id"].astype(str).tolist()
    np.savez_compressed(
        root / "sequence_tensors.npz",
        X=np.ones((len(ids), 40, 1), dtype=float),
        sample_id=np.asarray(ids),
    )
    pd.DataFrame(columns=["index", "reason", "action"]).to_csv(
        root / "row_corrections.csv", index=False
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"structural_removed_rows": 0, "affected_indices": 0},
                "rules": {
                    "interpolation": False,
                    "forward_fill": False,
                    "winsorization": False,
                    "arbitrary_clipping": False,
                    "synthetic_dates": False,
                },
                "test_evaluated": False,
            }
        ),
        encoding="utf-8",
    )


def test_audit_accepts_three_controls_and_exact_tensor_alignment(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)

    report = audit_processed_dataset(dataset)

    assert report["passed"] is True
    assert report["counts"]["controls_per_positive"] == 3


def test_audit_rejects_wrong_control_ratio(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, controls=2)

    report = audit_processed_dataset(dataset)

    assert report["passed"] is False
    assert any("control/positive ratio is not 3:1" in item for item in report["failures"])


def test_audit_rejects_tensor_ids_in_different_order(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, tensor_ids=["N1", "P1", "N2", "N3"])

    report = audit_processed_dataset(dataset)

    assert report["passed"] is False
    assert "tensor and manifest IDs are not exactly aligned in order" in report["failures"]
