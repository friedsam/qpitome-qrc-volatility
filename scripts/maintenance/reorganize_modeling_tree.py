#!/usr/bin/env python3
"""Reorganize the historically flat scripts/modeling directory.

Safe by design:
- refuses to run with a dirty working tree unless --allow-dirty is supplied;
- uses ``git mv`` rather than copy/delete;
- classifies every top-level Python script deterministically;
- updates Markdown path references after moves;
- writes a machine-readable move manifest and a human-readable audit report;
- never touches files already stored in a modeling subdirectory.

Run first without --apply to inspect the proposed mapping. Then run with
--apply and review ``git diff --stat`` and ``git status --short`` before commit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "scripts" / "modeling"
REPORT_DIR = ROOT / "docs" / "repository"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=check
    )


def classify(name: str) -> str:
    lower = name.lower()
    if "day5" in lower or "cross_market" in lower or "first_passage" in lower:
        return "day5_barrier"
    if "weekly_regime" in lower or "regime_baseline" in lower or "regime_state" in lower:
        return "weekly_regimes"
    if "branch_onset" in lower or "onset_trajectory" in lower:
        return "task_a_onset"
    if any(
        token in lower
        for token in (
            "branch_destination",
            "destination_target",
            "destination_structure",
            "destination_paths",
            "destination_esn",
            "destination_rydberg",
        )
    ):
        return "task_b_destination"
    if any(token in lower for token in ("hmm", "garch", "har", "lstm", "esn")):
        return "classical_models"
    if any(token in lower for token in ("qrc", "quantum", "rydberg")):
        return "quantum_models"
    return "legacy_unclassified"


def markdown_files() -> list[Path]:
    candidates = []
    for base in (ROOT / "docs", ROOT):
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            if ".git" not in path.parts and path not in candidates:
                candidates.append(path)
    return sorted(candidates)


def proposed_moves() -> list[tuple[Path, Path]]:
    moves = []
    for source in sorted(MODELING.glob("*.py")):
        destination = MODELING / classify(source.name) / source.name
        moves.append((source, destination))
    return moves


def replace_references(moves: list[tuple[Path, Path]]) -> list[str]:
    changed = []
    replacements = {
        source.relative_to(ROOT).as_posix(): destination.relative_to(ROOT).as_posix()
        for source, destination in moves
    }
    for path in markdown_files():
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(ROOT).as_posix())
    return changed


def scan_layout() -> dict[str, object]:
    top_level_scripts = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("*.py")
    )
    flat_modeling = sorted(path.name for path in MODELING.glob("*.py"))
    root_data_files = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".npy", ".npz", ".json"}
    )
    result_dirs_without_readme = []
    results = ROOT / "results"
    if results.exists():
        for directory in sorted(path for path in results.rglob("*") if path.is_dir()):
            files = [item for item in directory.iterdir() if item.is_file()]
            if files and not any(
                (directory / marker).exists()
                for marker in ("README.md", "run_manifest.json", "manifest.json")
            ):
                result_dirs_without_readme.append(directory.relative_to(ROOT).as_posix())
    return {
        "top_level_scripts": top_level_scripts,
        "flat_modeling_scripts": flat_modeling,
        "root_data_files": root_data_files,
        "result_directories_without_manifest": result_dirs_without_readme,
    }


def write_reports(
    moves: list[tuple[Path, Path]],
    changed_docs: list[str],
    applied: bool,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "applied": applied,
        "moves": [
            {
                "source": source.relative_to(ROOT).as_posix(),
                "destination": destination.relative_to(ROOT).as_posix(),
            }
            for source, destination in moves
        ],
        "markdown_files_updated": changed_docs,
        "layout_audit": scan_layout(),
    }
    (REPORT_DIR / "modeling_reorganization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    lines = [
        "# Repository layout audit",
        "",
        f"Reorganization applied: **{applied}**",
        "",
        "## Modeling moves",
        "",
    ]
    for item in manifest["moves"]:
        lines.append(f"- `{item['source']}` -> `{item['destination']}`")
    lines.extend(["", "## Markdown references updated", ""])
    lines.extend(f"- `{path}`" for path in changed_docs or ["None"])
    lines.extend(["", "## Remaining layout findings", ""])
    audit = manifest["layout_audit"]
    for key, values in audit.items():
        lines.append(f"### {key.replace('_', ' ').title()}")
        lines.append("")
        if values:
            lines.extend(f"- `{value}`" for value in values)
        else:
            lines.append("- None")
        lines.append("")
    (REPORT_DIR / "layout_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    status = git("status", "--porcelain").stdout.strip()
    if args.apply and status and not args.allow_dirty:
        raise SystemExit(
            "Refusing to reorganize a dirty tree. Commit/stash current changes or pass --allow-dirty."
        )

    moves = proposed_moves()
    if not moves:
        print("No top-level scripts remain in scripts/modeling; nothing to move.")
        write_reports([], [], applied=args.apply)
        return

    print("Proposed modeling-script moves:")
    for source, destination in moves:
        print(f"  {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")

    changed_docs: list[str] = []
    if args.apply:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            git("mv", source.relative_to(ROOT).as_posix(), destination.relative_to(ROOT).as_posix())
        changed_docs = replace_references(moves)
        print(f"Updated {len(changed_docs)} Markdown files containing moved paths.")

    write_reports(moves, changed_docs, applied=args.apply)
    print("Wrote docs/repository/layout_audit.md")
    print("Wrote docs/repository/modeling_reorganization_manifest.json")
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the mapping.")


if __name__ == "__main__":
    main()
