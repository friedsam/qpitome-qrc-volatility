from __future__ import annotations

import json
from pathlib import Path

import pytest

from transition_forecasting.data import submission_provenance as provenance


def test_writer_requires_exact_frozen_inventory_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("contract\n", encoding="utf-8")
    (fallback / "source_manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        provenance,
        "validate_source",
        lambda root: {"valid": True},
    )
    monkeypatch.setattr(
        provenance,
        "compare_source_to_frozen_inventory",
        lambda root, path: {"matched": False, "reference": str(path)},
    )

    with pytest.raises(ValueError, match="refusing to write fallback manifest"):
        provenance.write_submission_fallback_manifest(fallback, inventory)
    assert not (fallback / "fallback_manifest.json").exists()


def test_writer_records_source_and_contract_then_reverifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("contract\n", encoding="utf-8")
    (fallback / "source_manifest.json").write_text(
        '{"source": "kaggle"}\n',
        encoding="utf-8",
    )
    (fallback / "all_indices_data.csv").write_text("data\n", encoding="utf-8")

    comparison = {"matched": True, "reference": str(inventory)}
    monkeypatch.setattr(
        provenance,
        "validate_source",
        lambda root: {"valid": True, "file_count": 1},
    )
    monkeypatch.setattr(
        provenance,
        "compare_source_to_frozen_inventory",
        lambda root, path: comparison,
    )
    monkeypatch.setattr(
        provenance,
        "file_inventory",
        lambda root: [
            {
                "path": "all_indices_data.csv",
                "size_bytes": 5,
                "sha256": "data-hash",
            }
        ],
    )
    monkeypatch.setattr(
        provenance,
        "sha256_file",
        lambda path: f"hash:{Path(path).name}",
    )
    monkeypatch.setattr(provenance, "utc_now", lambda: "2026-07-26T00:00:00+00:00")
    monkeypatch.setattr(
        provenance,
        "verify_submission_fallback",
        lambda root, path: {"verified": True},
    )

    report = provenance.write_submission_fallback_manifest(fallback, inventory)
    payload = json.loads(
        (fallback / "fallback_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 2
    assert payload["source_manifest"] == "source_manifest.json"
    assert payload["source_manifest_sha256"] == "hash:source_manifest.json"
    assert payload["frozen_inventory_sha256"] == "hash:inventory.csv"
    assert payload["frozen_inventory_comparison"]["matched"] is True
    assert payload["files"][0]["path"] == "all_indices_data.csv"
    assert report["files"] == 1
    assert report["verification"] == {"verified": True}
    assert not (fallback / "fallback_manifest.json.tmp").exists()
