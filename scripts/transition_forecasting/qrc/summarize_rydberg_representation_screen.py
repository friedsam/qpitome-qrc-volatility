from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.representation_screen_summary import (
    summarize_representation_screen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild an all-fold representation-screen summary."
    )
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, plot_path = summarize_representation_screen(args.run_dir)
    print(f"WROTE {csv_path}")
    print(f"WROTE {plot_path}")


if __name__ == "__main__":
    main()
