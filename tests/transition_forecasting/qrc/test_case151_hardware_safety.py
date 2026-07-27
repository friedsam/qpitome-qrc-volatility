from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_case151_hardware_tools_are_retrieval_only() -> None:
    paths = (
        REPO_ROOT / "scripts/hardware/aquila_case151_support.py",
        REPO_ROOT / "scripts/hardware/aquila_case151_shrink_run.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert 'choices=("prepare", "status", "collect")' in text
    assert '"--submit"' not in text
    assert ".run(" not in text
    assert "create_quantum_task" not in text
