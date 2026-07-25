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
QBRAID_REQUIREMENTS = REPO_ROOT / "requirements-qbraid.txt"


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
    paths = set(re.findall(r"(?:python|pytest -q)\s+([A-Za-z0-9_./-]+\.py)", text))

    assert "scripts/runs/run_submission.py" in text
    assert "qbraid_skill/scripts/preflight.py" in text
    for relative in paths:
        assert (REPO_ROOT / relative).is_file(), relative


def test_skill_is_explicit_about_current_scope_and_hardware_boundary() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "transition-forecasting data stage only" in text
    assert "Do not submit a hardware job" in text
    assert "The final financial QRC" in text
    assert "MNIST benchmark" in text
    assert "qbraid jobs submit" not in text.lower()


def test_qbraid_environment_setup_matches_cli_013() -> None:
    skill = SKILL_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    required_fragments = (
        "qbraid envs create",
        "--name qrc-volatility",
        "--requirements requirements-qbraid.txt",
        "qbraid envs activate qrc-volatility",
        "python -m pip install -e . --no-deps",
    )
    for fragment in required_fragments:
        assert fragment in skill
        assert fragment in readme

    assert "qbraid envs create -f" not in skill
    assert "qbraid envs create -f" not in readme
    assert "does not create an environment or install dependencies" in skill
    assert "does not create the project environment or install dependencies" in readme


def test_qbraid_requirements_cover_current_stage1_dependencies() -> None:
    text = QBRAID_REQUIREMENTS.read_text(encoding="utf-8")

    for package in (
        "setuptools",
        "kaggle",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "pytest",
    ):
        assert package in text


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
