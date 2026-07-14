from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from data.download_market_data import fetch_yahoo_history


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_path: Path
    fallback_path: Path | None = None
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
    fallback_path: str | None = None
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


def _copy_fallback(spec: DatasetSpec, destination: Path) -> None:
    fallback = spec.fallback_path
    if fallback is None:
        raise FileNotFoundError("no fallback path configured")
    if not fallback.exists() or not fallback.is_file():
        raise FileNotFoundError(fallback)
    if fallback.stat().st_size == 0:
        raise RuntimeError(f"fallback file is empty: {fallback}")
    shutil.copy2(fallback, destination)


def acquire_dataset(spec: DatasetSpec) -> AcquisitionResult:
    """Attempt a public download and copy a user-supplied fallback on failure.

    Both remote downloads and fallback copies are written atomically. Existing
    target files are never treated as fallbacks; the fallback must be supplied
    explicitly through ``fallback_path``.
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

    remote_error: Exception | None = None
    try:
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
                fallback_path=str(spec.fallback_path) if spec.fallback_path else None,
            )
        except Exception as exc:
            remote_error = exc
            temporary_path.unlink(missing_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=spec.target_path.parent,
            prefix=f".{spec.target_path.name}.fallback.",
            suffix=suffix,
            delete=False,
        ) as handle:
            fallback_temporary_path = Path(handle.name)

        try:
            _copy_fallback(spec, fallback_temporary_path)
            fallback_temporary_path.replace(spec.target_path)
        except Exception as fallback_error:
            fallback_temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not acquire {spec.name} from {source}; fallback unavailable "
                f"or invalid at {spec.fallback_path}. Remote error: "
                f"{type(remote_error).__name__}: {remote_error}. Fallback error: "
                f"{type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error

        return AcquisitionResult(
            name=spec.name,
            path=str(spec.target_path),
            status="fallback_copied",
            source=source,
            sha256=_sha256(spec.target_path),
            fallback_path=str(spec.fallback_path),
            error=f"{type(remote_error).__name__}: {remote_error}",
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def acquire_external_datasets(
    specs: list[DatasetSpec],
    *,
    manifest_path: Path,
) -> list[AcquisitionResult]:
    results = [acquire_dataset(spec) for spec in specs]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "remote_first_explicit_fallback_copy",
        "datasets": [asdict(result) for result in results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return results
