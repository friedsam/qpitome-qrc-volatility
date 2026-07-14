from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from data.download_market_data import fetch_yahoo_history


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_path: Path
    source_url: str | None = None
    yahoo_symbol: str | None = None
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class AcquisitionResult:
    name: str
    path: str
    status: str
    source: str
    sha256: str
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_url(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _download_yahoo(spec: DatasetSpec, destination: Path) -> None:
    if not spec.yahoo_symbol or not spec.start_date or not spec.end_date:
        raise ValueError(f"Incomplete Yahoo specification for {spec.name}")
    frame = fetch_yahoo_history(spec.yahoo_symbol, spec.start_date, spec.end_date)
    frame.to_csv(destination, index=False)


def acquire_dataset(spec: DatasetSpec) -> AcquisitionResult:
    """Attempt a public download and fall back to an existing local snapshot.

    A successful download is written atomically. If the remote source fails and
    ``target_path`` already exists, that file is retained and reported as a
    local fallback. If neither source is available, the function raises.
    """
    spec.target_path.parent.mkdir(parents=True, exist_ok=True)
    source = spec.source_url or f"Yahoo Finance:{spec.yahoo_symbol}"

    suffix = spec.target_path.suffix or ".tmp"
    with tempfile.NamedTemporaryFile(
        dir=spec.target_path.parent,
        prefix=f".{spec.target_path.name}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

    try:
        if spec.source_url:
            _download_url(spec.source_url, temporary_path)
        elif spec.yahoo_symbol:
            _download_yahoo(spec, temporary_path)
        else:
            raise ValueError(f"No remote source configured for {spec.name}")

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("download produced an empty file")
        temporary_path.replace(spec.target_path)
        return AcquisitionResult(
            name=spec.name,
            path=str(spec.target_path),
            status="downloaded",
            source=source,
            sha256=_sha256(spec.target_path),
        )
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        if spec.target_path.exists() and spec.target_path.stat().st_size > 0:
            return AcquisitionResult(
                name=spec.name,
                path=str(spec.target_path),
                status="local_fallback",
                source=source,
                sha256=_sha256(spec.target_path),
                error=f"{type(exc).__name__}: {exc}",
            )
        raise RuntimeError(
            f"Could not acquire {spec.name} from {source}, and no local fallback "
            f"exists at {spec.target_path}"
        ) from exc


def acquire_external_datasets(
    specs: list[DatasetSpec],
    *,
    manifest_path: Path,
) -> list[AcquisitionResult]:
    results = [acquire_dataset(spec) for spec in specs]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "remote_first_local_fallback",
        "datasets": [asdict(result) for result in results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return results
