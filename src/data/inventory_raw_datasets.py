from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RawFileInventory:
    path: str
    suffix: str
    size_bytes: int
    sha256: str
    columns: list[str]
    sample_rows: list[list[str]]
    zip_members: list[str]
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_csv(path: Path, sample_size: int) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = []
        for _ in range(sample_size + 1):
            try:
                rows.append(next(reader))
            except StopIteration:
                break
    if not rows:
        return [], []
    return rows[0], rows[1:]


def inspect_raw_file(path: Path, *, sample_size: int = 3) -> RawFileInventory:
    columns: list[str] = []
    sample_rows: list[list[str]] = []
    zip_members: list[str] = []
    error: str | None = None

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            columns, sample_rows = _inspect_csv(path, sample_size)
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                zip_members = archive.namelist()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return RawFileInventory(
        path=str(path),
        suffix=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        columns=columns,
        sample_rows=sample_rows,
        zip_members=zip_members,
        error=error,
    )


def inventory_raw_datasets(
    raw_root: Path,
    *,
    output_path: Path,
    sample_size: int = 3,
    required_paths: list[Path] | None = None,
) -> list[RawFileInventory]:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    required_paths = required_paths or []
    missing_required = [path for path in required_paths if not path.is_file()]

    supported = {".csv", ".zip"}
    files = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )
    results = [inspect_raw_file(path, sample_size=sample_size) for path in files]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "file_count": len(results),
        "required_file_count": len(required_paths),
        "missing_required_files": [str(path) for path in missing_required],
        "files": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if missing_required:
        formatted = "\n".join(f"- {path}" for path in missing_required)
        raise FileNotFoundError(
            "Required canonical raw datasets are missing. "
            "Run the acquisition scripts to restore them from remote or fallback:\n"
            f"{formatted}"
        )

    return results
