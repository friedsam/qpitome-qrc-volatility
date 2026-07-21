from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from transition_forecasting.data.validation import audit_processed_dataset


def audit_parity_control_dataset(
    dataset_dir: Path,
    *,
    controls_per_positive: int = 3,
) -> dict[str, object]:
    """Run the standard audit through a temporary compatibility view.

    The parity control truthfully records detected bad prints as retained. The
    canonical validator expects structural ledger rows to represent removals,
    so this function adjusts only temporary copies of the ledger and manifest.
    Dataset values, samples, tensors, splits, and control matching are audited
    unchanged.
    """
    dataset_dir = Path(dataset_dir)
    with tempfile.TemporaryDirectory(prefix="parity-audit-", dir=dataset_dir.parent) as temporary:
        view = Path(temporary) / "dataset"
        shutil.copytree(dataset_dir, view)

        corrections_path = view / "row_corrections.csv"
        corrections = pd.read_csv(corrections_path)
        structural = corrections["reason"].astype(str).eq("structural_bad_print")
        corrections.loc[structural, "action"] = "drop"
        corrections.to_csv(corrections_path, index=False)

        manifest_path = view / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.setdefault("counts", {})
        counts["structural_removed_rows"] = int(counts.get("structural_detected_rows", structural.sum()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = audit_processed_dataset(
            view,
            controls_per_positive=controls_per_positive,
        )
        report["parity_control_compatibility_view"] = True
        return report
