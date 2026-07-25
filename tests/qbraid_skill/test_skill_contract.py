from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "qbraid_skill"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
BOOTSTRAP_PATH = SKILL_ROOT / "scripts" / "bootstrap.py"
PREFLIGHT_PATH = SKILL_ROOT / "scripts" / "preflight.py"
README_PATH = REPO_ROOT / "README.md"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
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

    for relative in (
        "references/repository-map.md",
        "references/run-contract.md",
    ):
        assert f"`{relative}`" in text
        assert (SKILL_ROOT / relative).is_file()


def test_skill_invokes_only_existing_entry_points() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    for relative in (
        "qbraid_skill/scripts/bootstrap.py",
        "qbraid_skill/scripts/preflight.py",
        "scripts/runs/run_submission.py",
    ):
        assert relative in text
        assert (REPO_ROOT / relative).is_file(), relative


def test_skill_is_explicit_about_scope_and_hardware_boundary() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "transition-forecasting data stage only" in text
    assert "Do not submit or query a hardware job" in text
    assert "The final financial QRC" in text
    assert "MNIST benchmark" in text
    assert "qbraid jobs submit" not in text.lower()


def test_qbraid_setup_is_agent_owned() -> None:
    skill = SKILL_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "The agent owns environment setup" in skill
    assert "Do not ask the judge to run terminal commands" in skill
    assert "python3 qbraid_skill/scripts/bootstrap.py --json" in skill
    assert "python3 qbraid_skill/scripts/bootstrap.py --json" in readme
    assert "Do not ask me to run terminal commands" in readme
    assert "qbraid envs create" not in skill
    assert "qbraid envs create" not in readme
    assert "source .venv/bin/activate" not in skill
    assert "source .venv/bin/activate" not in readme


def test_bootstrap_plan_is_idempotent_and_uses_local_interpreter(tmp_path: Path) -> None:
    bootstrap = load_module("qbraid_skill_bootstrap", BOOTSTRAP_PATH)
    venv_dir = tmp_path / "qrc-venv"

    plan = bootstrap.build_command_plan(venv_dir, skip_tests=False)
    expected_python = bootstrap.venv_python(venv_dir)

    assert plan[0] == [sys.executable, "-m", "venv", str(venv_dir)]
    assert plan[1] == [
        str(expected_python),
        "-m",
        "pip",
        "install",
        "-e",
        ".[test]",
    ]
    assert plan[2] == [
        str(expected_python),
        "qbraid_skill/scripts/preflight.py",
        "--json",
    ]
    assert plan[3][:4] == [str(expected_python), "-m", "pytest", "-q"]
    assert "tests/qbraid_skill/test_skill_contract.py" in plan[3]


def test_bootstrap_reuses_existing_environment(tmp_path: Path) -> None:
    bootstrap = load_module("qbraid_skill_bootstrap_existing", BOOTSTRAP_PATH)
    venv_dir = tmp_path / "qrc-venv"
    python_path = bootstrap.venv_python(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    plan = bootstrap.build_command_plan(venv_dir, skip_tests=True)

    assert all(command[1:3] != ["-m", "venv"] for command in plan)
    assert plan[0][:4] == [str(python_path), "-m", "pip", "install"]
    assert plan[-1] == [
        str(python_path),
        "qbraid_skill/scripts/preflight.py",
        "--json",
    ]


def test_readme_contains_launch_link_and_agent_prompt() -> None:
    text = README_PATH.read_text(encoding="utf-8")

    assert "https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" in text
    assert (
        "https://account.qbraid.com?gitHubUrl="
        "https://github.com/friedsam/qpitome-qrc-volatility.git"
    ) in text
    assert "Read qbraid_skill/SKILL.md before acting" in text
    assert "enable **Agent Mode**" in text


def test_preflight_reports_repository_contract_without_credentials() -> None:
    preflight = load_module("qbraid_skill_preflight", PREFLIGHT_PATH)
    report = preflight.build_report()

    assert report["passed"] is True
    assert report["repository_root"] == str(REPO_ROOT)
    assert report["repository_contract"]["missing_required_paths"] == []
    assert report["python"]["supported"] is True
    assert report["data_source"]["recommended_mode"] in {None, "live", "fallback"}


def test_preflight_strict_mode_has_distinct_blocked_exit_code() -> None:
    preflight = load_module("qbraid_skill_preflight_strict", PREFLIGHT_PATH)
    report = preflight.build_report()

    expected = 0 if report["data_source"]["ready"] else 2
    assert preflight.main(["--strict-data-source"]) == expected
