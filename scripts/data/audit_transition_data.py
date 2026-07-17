from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.transition_data_audit import audit_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dated eight-index OHLC transition data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_data(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
