#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of generated QuadArena configs.")
    parser.add_argument("--config-directory", required=True, help="Directory containing TOML configs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs whose result file already exists and is non-empty.",
    )
    return parser.parse_args()


def result_file_for_config(path: Path) -> Path:
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    recorder = config.get("game_result_recorder", {})
    output_path = recorder.get("path")
    if not output_path:
        raise ValueError(f"Config {path} is missing game_result_recorder.path")
    return Path(output_path)


def should_skip(config_path: Path, resume: bool) -> bool:
    if not resume:
        return False
    result_path = result_file_for_config(config_path)
    return result_path.exists() and result_path.stat().st_size > 0


def main() -> int:
    args = parse_args()
    config_directory = Path(args.config_directory)
    config_paths = sorted(config_directory.glob("*.toml"))
    if not config_paths:
        raise ValueError(f"No TOML configs found in {config_directory}")

    for config_path in config_paths:
        if should_skip(config_path, args.resume):
            print(f"Skipping {config_path.name}: result file already exists")
            continue

        run_name = config_path.stem
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "self-play",
            "--",
            "--config",
            str(config_path),
            "--run-name",
            run_name,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
