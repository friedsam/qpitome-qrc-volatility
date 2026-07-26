from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPEC_PATH = (
    REPO_ROOT
    / "config"
    / "transition_forecasting"
    / "classical_benchmarks"
    / "frozen_submission.json"
)


def load_frozen_spec(path: Path | None = None) -> dict[str, object]:
    spec_path = Path(path) if path is not None else DEFAULT_SPEC_PATH
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "selection_folds",
        "confirmation_folds",
        "linear",
        "garch",
        "esn",
        "reporting",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"frozen classical spec missing fields: {sorted(missing)}")
    if payload["reporting"]["test_evaluated"] is not False:
        raise ValueError("frozen classical spec must keep test_evaluated=false")
    return payload
