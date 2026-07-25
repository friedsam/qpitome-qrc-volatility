from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "qbraid_skill"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
PREFLIGHT_PATH = SKILL_ROOT / "scripts" / "preflight.py"
README_PATH = REPO_ROOT / "README.md"


def load_preflight() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qbraid_skill_preflight", PREFLIGHT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---", maxsplit=2)
    assert body.strip()
    fields: dict[str, str] = {}
    for raw_line in frontmatter.strip().splitlines():
        key, separator, value = raw_line.partition(":")
        assert separator == ":"
        fields[key.strip()] = value.strip()
    return fields


def test_skill_uses_agent_skills_frontmatter() -> None:
    fields = parse_frontmatter(SKILL_PATH.read_text(encoding="utf-8"))

    assert fields["name"] == "qpitome-qrc-volatility"
    assert "qBraid" in fields["description"]
    assert "reproduce" in fields["description"].lower()
    assert set(fields) == {"name", "description"}


def test_skill_references_exist_and_remain_inside_package() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    references = re.findall(r"`(references/[^`]+\.md)`", text)

    assert set(references) == {
        "references/repository-map.md",
        "references/run-contract.md",
    }
    for relative in references:
        assert (SKILL_ROOT / relative).is_file()


def test_skill_invokes_only_existing_stage1_python_entry_points() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "scripts/runs/run_submission.py" in text
    assert "qbraid_skill/scripts/preflight.py" in text
    for relative in (
        "scripts/runs/run_submission.py",
        "qbraid_skill/scripts/preflight.py",
    ):
        assert (REPO_ROOT / relative).is_file(), relative


def test_skill_is_explicit_about_current_scope_and_hardware_boundary() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "transition-forecasting data stage only" in text
    assert "Do not submit a hardware job" in text
    assert "The final financial QRC" in text
    assert "MNIST benchmark" in text
    assert "qbraid jobs submit" not in text.lower()


def test_qbraid_setup_is_agent_safe_and_idempotent() -> None:
    skill = SKILL_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    required_fragments = (
        "if [ ! -x .venv/bin/python ]; then",
        "python3 -m venv .venv",
        ".venv/bin/python -m pip install --upgrade pip",
        '.venv/bin/python -m pip install -e ".[test]"',
        ".venv/bin/python qbraid_skill/scripts/preflight.py --json",
        ".venv/bin/python scripts/runs/run_submission.py transition-data",
    )
    for fragment in required_fragments:
        assert fragment in skill
        assert fragment in readme

    assert "qbraid envs create" not in skill
    assert "qbraid envs create" not in readme
    assert "source .venv/bin/activate" not in skill
    assert "source .venv/bin/activate" not in readme
    assert "The agent must create the environment itself" in skill
    assert "Do not ask the user to create or activate it" in skill


def test_readme_contains_official_launch_on_qbraid_link() -> None:
    text = README_PATH.read_text(encoding="utf-8")

    assert "https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" in text
    assert (
        "https://account.qbraid.com?gitHubUrl="
        "https://github.com/friedsam/qpitome-qrc-volatility.git"
    ) in text


def test_preflight_reports_repository_contract_without_requiring_credentials() -> None:
    preflight = load_preflight()
    report = preflight.build_report()

    assert report["passed"] is True
    assert report["repository_root"] == str(REPO_ROOT)
    assert report["repository_contract"]["missing_required_paths"] == []
    assert report["python"]["supported"] is True
    assert report["data_source"]["recommended_mode"] in {None, "live", "fallback"}


def test_preflight_strict_mode_has_distinct_blocked_exit_code() -> None:
    preflight = load_preflight()
    report = preflight.build_report()

    expected = 0 if report["data_source"]["ready"] else 2
    assert preflight.main(["--strict-data-source"]) == expected
