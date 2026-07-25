from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transition_forecasting.data import acquisition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source(root: Path, *, marker: float) -> None:
    root.mkdir(parents=True)
    (root / "individual_indices_data").mkdir()
    combined = "date,open,high,low,close,ticker\n2024-01-02,1,2,0.5,{marker},X\n".format(marker=marker)
    (root / "all_indices_data.csv").write_text(combined, encoding="utf-8")
    template = "date,open,high,low,close\n2024-01-02,1,2,0.5,{marker}\n"
    for index in range(30):
        (root / "individual_indices_data" / f"index_{index:02d}.csv").write_text(
            template.format(marker=marker), encoding="utf-8"
        )


def _write_fallback(root: Path, *, marker: float) -> None:
    _write_source(root, marker=marker)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    (root / "fallback_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_live_mismatch_is_replaced_by_verified_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "fallback"
    destination = tmp_path / "destination"
    _write_fallback(fallback, marker=1.0)

    def fake_download(candidate: Path) -> None:
        _write_source(candidate, marker=9.0)

    monkeypatch.setattr(acquisition, "download_live", fake_download)

    report = acquisition.acquire_source(
        destination,
        fallback,
        source_mode="live",
    )

    assert report["source_mode_used"] == "fallback_after_live_mismatch"
    assert report["fallback_substitution"] is True
    assert report["source_comparison"]["matched"] is False
    assert report["authoritative_source_verified"] is True
    assert acquisition.compare_source_to_fallback(destination, fallback)["matched"] is True

    persisted = json.loads(
        (destination / "raw_acquisition_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["fallback_substitution"] is True
    assert "discarded" in persisted["substitution_reason"]


def test_live_match_is_kept_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "fallback"
    destination = tmp_path / "destination"
    _write_fallback(fallback, marker=1.0)

    def fake_download(candidate: Path) -> None:
        acquisition.copy_source(fallback, candidate)

    monkeypatch.setattr(acquisition, "download_live", fake_download)

    report = acquisition.acquire_source(
        destination,
        fallback,
        source_mode="live",
    )

    assert report["source_mode_used"] == "live_verified_against_fallback"
    assert report["fallback_substitution"] is False
    assert report["source_comparison"]["matched"] is True


def test_live_without_fallback_must_match_frozen_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination"
    fallback = tmp_path / "missing-fallback"
    inventory = tmp_path / "inventory.csv"

    expected = tmp_path / "expected"
    _write_source(expected, marker=1.0)
    rows = []
    for path in sorted(item for item in expected.rglob("*.csv")):
        rows.append(
            {
                "path": (
                    "data/raw/transition_forecasting/global_stock_indices_historical_data/"
                    + path.relative_to(expected).as_posix()
                ),
                "sha256": _sha256(path),
            }
        )
    inventory.write_text(
        "path,sha256\n"
        + "".join(f"{row['path']},{row['sha256']}\n" for row in rows),
        encoding="utf-8",
    )

    def fake_download(candidate: Path) -> None:
        _write_source(candidate, marker=9.0)

    monkeypatch.setattr(acquisition, "download_live", fake_download)

    with pytest.raises(RuntimeError, match="does not match the frozen raw inventory"):
        acquisition.acquire_source(
            destination,
            fallback,
            source_mode="live",
            frozen_inventory=inventory,
        )

    assert not destination.exists()
