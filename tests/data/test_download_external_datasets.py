from pathlib import Path

import pytest

from data.download_external_datasets import DatasetSpec, acquire_dataset


def test_uses_local_fallback_when_remote_download_fails(tmp_path: Path) -> None:
    target = tmp_path / "existing.csv"
    target.write_text("date,value\n2020-01-01,1\n", encoding="utf-8")
    spec = DatasetSpec(
        name="example",
        target_path=target,
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    result = acquire_dataset(spec)

    assert result.status == "local_fallback"
    assert result.path == str(target)
    assert result.error is not None
    assert target.read_text(encoding="utf-8").startswith("date,value")


def test_fails_when_remote_and_local_sources_are_unavailable(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="missing",
        target_path=tmp_path / "missing.csv",
        source_url="http://127.0.0.1:1/unavailable.csv",
    )

    with pytest.raises(RuntimeError, match="no local fallback"):
        acquire_dataset(spec)
