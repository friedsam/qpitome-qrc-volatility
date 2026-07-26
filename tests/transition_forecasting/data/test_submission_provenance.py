from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from transition_forecasting.data import submission_provenance as provenance


def test_prefers_executable_beside_active_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    kaggle = bin_dir / "kaggle"
    python.write_text("", encoding="utf-8")
    kaggle.write_text("#!/bin/sh\n", encoding="utf-8")
    kaggle.chmod(0o755)
    monkeypatch.setenv("PATH", "")

    assert provenance.active_environment_executable(
        "kaggle",
        python_executable=python,
    ) == str(kaggle)


def test_expose_adds_environment_bin_to_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    kaggle = bin_dir / "kaggle"
    python.write_text("", encoding="utf-8")
    kaggle.write_text("#!/bin/sh\n", encoding="utf-8")
    kaggle.chmod(0o755)
    monkeypatch.setattr(provenance.sys, "executable", str(python))
    monkeypatch.setenv("PATH", "/usr/bin")

    assert provenance.expose_active_environment_executable("kaggle") == str(kaggle)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)


def test_fallback_must_match_frozen_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("contract\n", encoding="utf-8")

    monkeypatch.setattr(
        provenance,
        "verify_fallback_manifest",
        lambda root: {"manifest": str(root), "verified": True},
    )
    monkeypatch.setattr(
        provenance,
        "validate_source",
        lambda root: {"root": str(root), "valid": True},
    )
    monkeypatch.setattr(
        provenance,
        "compare_source_to_frozen_inventory",
        lambda root, path: {"matched": False, "reference": str(path)},
    )

    with pytest.raises(ValueError, match="frozen raw inventory"):
        provenance.verify_submission_fallback(fallback, inventory)

    monkeypatch.setattr(
        provenance,
        "compare_source_to_frozen_inventory",
        lambda root, path: {"matched": True, "reference": str(path)},
    )
    report = provenance.verify_submission_fallback(fallback, inventory)
    assert report["frozen_inventory_comparison"]["matched"] is True


def test_preserves_source_manifest_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source_manifest.json").write_text(
        '{"source": "kaggle"}\n',
        encoding="utf-8",
    )

    assert provenance.preserve_source_manifest(source, destination) is True
    assert provenance.preserve_source_manifest(source, destination) is False
    assert json.loads((destination / "source_manifest.json").read_text()) == {
        "source": "kaggle"
    }


def test_removes_manifest_self_hash_atomically(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "data.csv": {"sha256": "a"},
                    "manifest.json": {"sha256": "stale"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert provenance.remove_manifest_self_hash(tmp_path) is True
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "manifest.json" not in payload["files"]
    assert "outer run manifest" in payload["manifest_hash_scope"]
    assert provenance.remove_manifest_self_hash(tmp_path) is False
