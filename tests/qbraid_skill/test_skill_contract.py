from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "qbraid_skill" / "qpitome-qrc-volatility"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
BOOTSTRAP_PATH = SKILL_ROOT / "scripts" / "bootstrap.py"
PREFLIGHT_PATH = SKILL_ROOT / "scripts" / "preflight.py"
CLASSICAL_PREFLIGHT_PATH = SKILL_ROOT / "scripts" / "preflight_classical.py"
ORCHESTRATOR_PATH = SKILL_ROOT / "scripts" / "run_submission_scope.py"
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


def test_skill_uses_agent_skills_frontmatter_and_matching_directory() -> None:
    fields = parse_frontmatter(SKILL_PATH.read_text(encoding="utf-8"))
    assert fields["name"] == "qpitome-qrc-volatility"
    assert SKILL_ROOT.name == fields["name"]
    assert "qBraid" in fields["description"]
    assert "reproduce" in fields["description"].lower()
    assert "Case151" in fields["description"]
    assert set(fields) == {"name", "description"}


def test_skill_references_and_entry_points_exist() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    for relative in (
        "references/repository-map.md",
        "references/run-contract.md",
        "scripts/bootstrap.py",
        "scripts/preflight.py",
        "scripts/preflight_classical.py",
        "scripts/run_submission_scope.py",
    ):
        assert relative in text
        assert (SKILL_ROOT / relative).is_file()
    repository_paths = (
        "scripts/runs/run_submission.py",
        "scripts/runs/run_submission_layout.py",
        "scripts/runs/run_submission_stage_layout.py",
        "scripts/reproduction/run_case151_simulation.py",
        "scripts/runs/run_submission_benchmarks.py",
        "config/case151/expected_metrics.json",
    )
    for relative in repository_paths:
        assert (REPO_ROOT / relative).is_file(), relative
    for executable_contract in (
        "scripts/runs/run_submission_stage_layout.py",
        "scripts/reproduction/run_case151_simulation.py",
        "scripts/runs/run_submission_benchmarks.py",
        "config/case151/expected_metrics.json",
    ):
        assert executable_contract in text


def test_skill_requires_root_safe_state_free_execution() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "Do not assume the current working directory is the repository root" in text
    assert "Resolve every relative skill path against `SKILL_ROOT`" in text
    assert "For file reads, use absolute paths" in text
    assert "working directory explicitly to `REPO_ROOT`" in text
    assert "Do not rely on `cd`, shell variables, exported environment variables" in text
    assert "Do not create `RUN_ID` or `RUN_DIR` shell variables" in text
    assert "<SKILL_ROOT>/references/repository-map.md" in text
    assert "<SKILL_ROOT>/references/run-contract.md" in text


def test_skill_default_full_smoke_includes_all_submission_stages() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "Default full-smoke scope" in text
    assert "persistence, canonical HAR, and direct sequence-ridge" in text
    assert "Student-t GARCH(1,1)" in text
    assert "frozen tuned direct ESN" in text
    assert "financial-classical" in text
    assert "canonical Case151 QRC" in text
    assert "occupation_pair_raw" in text
    assert "fold-8 alpha `0.1`" in text
    assert "lambda `0.25`" in text
    assert "zero financial test rows" in text
    for family in ("MNIST", "noise", "scaling", "finite-shot"):
        assert family in text
    assert "validate-existing" in text
    assert "--profile smoke" in text


def test_skill_has_one_state_free_orchestrator_command() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    command = (
        ".venv/bin/python "
        "qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py"
    )
    assert command in text
    assert "--scope full-smoke" in text
    assert "--scope core" in text
    assert "--resume-existing" in text
    assert "The orchestrator generates one literal run ID" in text
    assert "RUN_ID=$(" not in text
    assert 'RUN_DIR="' not in text


def test_skill_preserves_test_and_hardware_boundaries() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "Do not evaluate the fixed financial test partition" in text
    assert "Do not perform any hardware action" in text
    assert "Hardware is not part of full-smoke or core" in text
    assert "Do not submit, query, select, retrieve, or package hardware jobs" in text
    assert "qbraid jobs submit" not in text.lower()


def test_qbraid_setup_is_agent_owned() -> None:
    skill = SKILL_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    bootstrap_command = (
        "python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json"
    )
    assert "Do not ask me to run terminal commands" in readme
    assert bootstrap_command in skill
    assert bootstrap_command in readme
    assert "Do not substitute a bare `pip`" in skill
    assert "source .venv/bin/activate" not in skill
    assert "source .venv/bin/activate" not in readme


def test_agent_orchestrator_builds_literal_argument_vectors(tmp_path: Path) -> None:
    orchestrator = load_module("qbraid_agent_scope", ORCHESTRATOR_PATH)
    run_id = "qbraid-submission-test"
    run_dir = tmp_path / run_id

    core = orchestrator.command_plan(
        scope="core",
        run_id=run_id,
        run_dir=run_dir,
        transition_source_mode="fallback",
        resume_existing=False,
    )
    assert len(core) == 2
    assert core[0][:3] == (
        sys.executable,
        "scripts/runs/run_submission_stage_layout.py",
        "financial-classical",
    )
    assert core[0][-2:] == ("--transition-source-mode", "fallback")
    assert core[1][1] == "scripts/reproduction/run_case151_simulation.py"
    assert str(run_dir) in " ".join(core[1])

    full = orchestrator.command_plan(
        scope="full-smoke",
        run_id=run_id,
        run_dir=run_dir,
        transition_source_mode="fallback",
        resume_existing=False,
    )
    assert len(full) == 4
    assert full[2][1:4] == (
        "scripts/runs/run_submission_benchmarks.py",
        "all",
        "--profile",
    )
    assert full[3][1:4] == (
        "scripts/runs/run_submission_benchmarks.py",
        "validate-existing",
        "--profile",
    )
    for argv in full:
        joined = " ".join(argv)
        assert "$RUN_ID" not in joined
        assert "$RUN_DIR" not in joined
        assert argv[0] == sys.executable


def test_agent_orchestrator_defaults_to_full_smoke_and_requires_source_mode() -> None:
    orchestrator = load_module("qbraid_agent_scope_args", ORCHESTRATOR_PATH)
    args = orchestrator.parse_args(["--transition-source-mode", "fallback"])
    assert args.scope == "full-smoke"
    assert args.transition_source_mode == "fallback"
    assert args.run_id is None


def test_bootstrap_plan_runs_preflights_and_complete_focused_contracts(
    tmp_path: Path,
) -> None:
    bootstrap = load_module("qbraid_skill_bootstrap", BOOTSTRAP_PATH)
    venv_dir = tmp_path / "qrc-venv"
    plan = bootstrap.build_command_plan(venv_dir, skip_tests=False)
    expected_python = bootstrap.venv_python(venv_dir)
    assert bootstrap.REPO_ROOT == REPO_ROOT
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
        "qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py",
        "--json",
    ]
    assert plan[3] == [
        str(expected_python),
        "qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py",
        "--json",
    ]
    assert plan[4][:4] == [str(expected_python), "-m", "pytest", "-q"]
    for required_test in (
        "tests/qbraid_skill/test_skill_contract.py",
        "tests/runs/test_run_submission_classical.py",
        "tests/runs/test_run_submission_benchmarks.py",
        "tests/transition_forecasting/modeling/classical_benchmarks/test_spec.py",
        "tests/transition_forecasting/qrc/test_case151_reproduction_contract.py",
        "tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py",
        "tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py",
        "tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py",
        "tests/transition_forecasting/qrc/test_palindrome_shot_assay.py",
    ):
        assert required_test in plan[4]


def test_bootstrap_requires_complete_agent_entry_points() -> None:
    bootstrap = load_module("qbraid_skill_bootstrap_paths", BOOTSTRAP_PATH)
    required = set(bootstrap.REQUIRED_REPOSITORY_PATHS)
    for relative in (
        "qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py",
        "scripts/runs/run_submission_stage_layout.py",
        "scripts/reproduction/run_case151_simulation.py",
        "scripts/runs/run_submission_benchmarks.py",
        "config/case151/expected_metrics.json",
        "config/case151/agent_run_spec.json",
    ):
        assert relative in required
        assert (REPO_ROOT / relative).is_file()


def test_bootstrap_reuses_existing_environment(tmp_path: Path) -> None:
    bootstrap = load_module("qbraid_skill_bootstrap_existing", BOOTSTRAP_PATH)
    venv_dir = tmp_path / "qrc-venv"
    python_path = bootstrap.venv_python(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    plan = bootstrap.build_command_plan(venv_dir, skip_tests=True)
    assert all(command[1:3] != ["-m", "venv"] for command in plan)
    assert plan[0][:4] == [str(python_path), "-m", "pip", "install"]
    assert plan[1][-1] == "--json"
    assert plan[2][-1] == "--json"
    assert "preflight_classical.py" in plan[2][1]


def test_readme_contains_one_complete_agent_prompt_and_command() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" in text
    assert (
        "https://account.qbraid.com?gitHubUrl="
        "https://github.com/friedsam/qpitome-qrc-volatility.git"
    ) in text
    assert "locate */qbraid_skill/qpitome-qrc-volatility/SKILL.md" in text
    assert "read it by absolute path" in text
    assert "enable **Agent Mode**" in text
    assert "this single prompt" in text
    assert "complete default full-smoke scope" in text
    assert "run_submission_scope.py" in text
    assert "--scope full-smoke" in text
    assert "run_submission_stage_layout.py financial-classical" in text
    assert "run_case151_simulation.py" in text
    assert "run_submission_benchmarks.py" in text
    assert "files/classical_baselines/" in text
    assert "files/qrc/simulation/run" in text
    assert "RUN_ID=$(" not in text
    assert 'RUN_DIR="' not in text
    assert "For the bounded Phase-3 integration extension, append" not in text


def test_data_preflight_reports_repository_contract() -> None:
    preflight = load_module("qbraid_skill_preflight", PREFLIGHT_PATH)
    report = preflight.build_report()
    assert preflight.REPO_ROOT == REPO_ROOT
    assert report["passed"] is True
    assert report["repository_root"] == str(REPO_ROOT)
    assert report["repository_contract"]["missing_required_paths"] == []
    assert report["python"]["supported"] is True
    assert report["data_source"]["recommended_mode"] in {
        None,
        "auto",
        "live",
        "fallback",
    }


def test_classical_preflight_reports_frozen_contract() -> None:
    preflight = load_module(
        "qbraid_skill_classical_preflight",
        CLASSICAL_PREFLIGHT_PATH,
    )
    report = preflight.build_report()
    assert preflight.REPO_ROOT == REPO_ROOT
    assert report["passed"] is True
    assert report["frozen_spec"]["garch"]["backend"] == "arch"
    assert report["frozen_spec"]["reporting"]["test_evaluated"] is False


def test_data_preflight_strict_mode_has_distinct_blocked_exit_code() -> None:
    preflight = load_module("qbraid_skill_preflight_strict", PREFLIGHT_PATH)
    report = preflight.build_report()
    expected = 0 if report["data_source"]["ready"] else 2
    assert preflight.main(["--strict-data-source"]) == expected
