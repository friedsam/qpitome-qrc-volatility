#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = Path(__file__).with_name("run_submission.py")
LAYOUT_PATH = Path(__file__).with_name("run_submission_layout.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_core = _load("submission_runner_core", CORE_PATH)
_layout = _load("submission_runner_layout", LAYOUT_PATH)
_core.transition_paths = _layout.transition_paths


def __getattr__(name: str):
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


transition_paths = _layout.transition_paths
main = _core.main


if __name__ == "__main__":
    raise SystemExit(main())
