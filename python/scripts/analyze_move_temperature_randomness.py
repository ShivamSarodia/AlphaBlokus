#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_PYTHON_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON_SRC))

from alphablokus.analysis.move_temperature_randomness import (  # noqa: E402
    analyze_move_temperature_file,
    format_summary_report,
)
from alphablokus.files import localize_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze move-selection concentration from game data at a given temperature."
    )
    parser.add_argument("--game-data-file", required=True, help="Local path or s3:// path to a game data .bin file.")
    parser.add_argument("--temperature", required=True, type=float, help="Move temperature to apply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_file_path = localize_file(args.game_data_file)
    summary = analyze_move_temperature_file(local_file_path, args.temperature)
    print(
        format_summary_report(
            input_path=args.game_data_file,
            temperature=args.temperature,
            summary=summary,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
