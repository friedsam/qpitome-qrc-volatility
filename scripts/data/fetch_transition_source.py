from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_REPOSITORY = "bumhoson/SpilloverVolPrediction"
SOURCE_COMMIT = "f21cf71faf09249d028392fd92008891b876675f"
ARCHIVE_URL = f"https://github.com/{SOURCE_REPOSITORY}/archive/{SOURCE_COMMIT}.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.zip"
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
        archive_hash = sha256(archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one extracted repository root, found {roots}")
        source_root = roots[0]
        for relative in ("global index etf return", "rv_dataset.csv", "README.md"):
            src = source_root / relative
            if not src.exists():
                raise FileNotFoundError(f"Pinned source archive is missing {relative}")
            dst = destination / relative
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    manifest = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": archive_hash,
        "retrieved_files": sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()),
    }
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned public transition-data source archive.")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination), indent=2))


if __name__ == "__main__":
    main()
