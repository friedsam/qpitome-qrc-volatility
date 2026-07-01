"""Test configuration for local script imports.

Some tests import modules from the repository-level scripts/ directory.
Ensure the repository root is importable during pytest collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))