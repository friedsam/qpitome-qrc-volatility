from pathlib import Path

import pytest

from data.download_external_datasets import DatasetSpec, acquire_dataset


def test_copies_explicit_fallback_when_remote_download_fails(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "dataset.csv"
    fallback = tmp_path / "fallback" / "dataset.csv"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("date,value\n2020-01-01,1\n", encoding="utf-8")

    spec = DatasetSpec(
        name="example",
        target_path=target,
        fallback_path=fallback,
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    result = acquire_dataset(spec)

    assert result.status == "fallback_copied"
    assert result.path == str(target)
    assert result.fallback_path == str(fallback)
    assert result.error is not None
    assert target.read_bytes() == fallback.read_bytes()


def test_existing_target_is_not_treated_as_fallback(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "existing.csv"
    target.parent.mkdir(parents=True)
    target.write_text("stale,data\n1,2\n", encoding="utf-8")

    spec = DatasetSpec(
        name="missing",
        target_path=target,
        fallback_path=tmp_path / "fallback" / "missing.csv",
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    with pytest.raises(RuntimeError, match="fallback unavailable or invalid"):
        acquire_dataset(spec)

    assert target.read_text(encoding="utf-8") == "stale,data\n1,2\n"


def test_fails_when_remote_and_fallback_sources_are_unavailable(
    tmp_path: Path,
) -> None:
    spec = DatasetSpec(
        name="missing",
        target_path=tmp_path / "raw" / "missing.csv",
        fallback_path=tmp_path / "fallback" / "missing.csv",
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    with pytest.raises(RuntimeError, match="fallback unavailable or invalid"):
        acquire_dataset(spec)


def test_rejects_empty_fallback_file(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback" / "empty.csv"
    fallback.parent.mkdir(parents=True)
    fallback.touch()

    spec = DatasetSpec(
        name="empty",
        target_path=tmp_path / "raw" / "empty.csv",
        fallback_path=fallback,
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    with pytest.raises(RuntimeError, match="fallback file is empty"):
        acquire_dataset(spec)
