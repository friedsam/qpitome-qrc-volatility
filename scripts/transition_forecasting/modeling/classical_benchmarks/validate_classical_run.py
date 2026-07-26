from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from transition_forecasting.modeling.classical_benchmarks.validation import validate_classical_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate_classical_run(
        classical_root=args.classical_root,
        run_id=args.run_id,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
