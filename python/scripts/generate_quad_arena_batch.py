#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires Python 3.11+ or the 'tomli' package on older Python versions."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch of QuadArena configs.")
    parser.add_argument("--manifest", required=True, help="Path to the batch manifest TOML.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to break schedule ties deterministically.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)

    required = [
        "output_directory",
        "results_directory",
        "num_configs",
        "duration_seconds",
        "game",
        "observability",
        "mcts_recorder",
        "game_result_recorder",
        "inference",
        "competitors",
    ]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Manifest is missing required keys: {', '.join(missing)}")

    if manifest["game_result_recorder"].get("type") != "jsonl_file":
        raise ValueError("Manifest game_result_recorder.type must be jsonl_file")

    if len(manifest["competitors"]) < 4:
        raise ValueError("Manifest must define at least 4 competitors")

    return manifest


def validate_competitors(competitors: list[dict[str, Any]]) -> None:
    seen_names: set[str] = set()
    for competitor in competitors:
        name = competitor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each competitor must define a non-empty name")
        if name in seen_names:
            raise ValueError(f"Duplicate competitor name: {name}")
        seen_names.add(name)


def build_schedule(
    competitors: list[dict[str, Any]],
    num_configs: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    competitor_count = len(competitors)
    if competitor_count < 4:
        raise ValueError("Need at least 4 competitors to build a QuadArena schedule")

    all_quads = list(itertools.combinations(range(competitor_count), 4))
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    appearance_counts = [0 for _ in range(competitor_count)]
    schedule: list[tuple[int, int, int, int]] = []

    for _ in range(num_configs):
        best_score: tuple[int, int, float] | None = None
        best_quads: list[tuple[int, int, int, int]] = []
        for quad in all_quads:
            pair_score = sum(pair_counts[pair] for pair in itertools.combinations(quad, 2))
            appearance_score = sum(appearance_counts[index] for index in quad)
            tie_breaker = rng.random()
            score = (pair_score, appearance_score, tie_breaker)
            if best_score is None or score < best_score:
                best_score = score
                best_quads = [quad]
            elif score[:2] == best_score[:2]:
                best_quads.append(quad)

        chosen = rng.choice(best_quads)
        schedule.append(chosen)
        for pair in itertools.combinations(chosen, 2):
            pair_counts[pair] += 1
        for index in chosen:
            appearance_counts[index] += 1

    ensure_connected(schedule, competitor_count)
    return schedule


def ensure_connected(schedule: list[tuple[int, int, int, int]], competitor_count: int) -> None:
    neighbors: dict[int, set[int]] = {index: set() for index in range(competitor_count)}
    for quad in schedule:
        for left, right in itertools.combinations(quad, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)

    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(neighbors[node] - seen)

    if len(seen) != competitor_count:
        raise ValueError("Generated schedule graph is disconnected")


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {format_value(subvalue)}" for key, subvalue in value.items())
        return "{ " + items + " }"
    if isinstance(value, list):
        items = ", ".join(format_value(item) for item in value)
        return "[" + items + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def write_key_values(lines: list[str], data: dict[str, Any], skip_dicts: bool) -> None:
    for key, value in data.items():
        if skip_dicts and isinstance(value, dict):
            continue
        lines.append(f"{key} = {format_value(value)}")


def write_named_table(lines: list[str], name: str, data: dict[str, Any]) -> None:
    lines.append(f"[{name}]")
    write_key_values(lines, data, skip_dicts=True)
    lines.append("")
    for key, value in data.items():
        if isinstance(value, dict):
            write_named_table(lines, f"{name}.{key}", value)


def render_config(
    manifest: dict[str, Any],
    quad: tuple[int, int, int, int],
    config_index: int,
    rng: random.Random,
) -> str:
    lines: list[str] = []
    lines.append(f"num_concurrent_games = {manifest.get('num_concurrent_games', 1)}")
    lines.append(f"duration_seconds = {manifest['duration_seconds']}")
    lines.append("")

    write_named_table(lines, "observability", manifest["observability"])
    write_named_table(lines, "mcts_recorder", manifest["mcts_recorder"])

    result_path = (
        Path(manifest["results_directory"]) / f"{config_index:03d}.jsonl"
    ).as_posix()
    game_result_recorder = dict(manifest["game_result_recorder"])
    game_result_recorder["path"] = result_path
    write_named_table(lines, "game_result_recorder", game_result_recorder)

    competitors = [dict(manifest["competitors"][index]) for index in quad]
    rng.shuffle(competitors)

    needed_inference_names = {
        competitor["inference_config_name"]
        for competitor in competitors
        if "inference_config_name" in competitor
    }

    for inference in manifest["inference"]:
        if inference["name"] not in needed_inference_names:
            continue
        lines.append("[[inference]]")
        write_key_values(lines, inference, skip_dicts=False)
        lines.append("")

    for competitor in competitors:
        lines.append("[[agents.QuadArena]]")
        write_key_values(lines, competitor, skip_dicts=False)
        lines.append("")

    write_named_table(lines, "game", manifest["game"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    validate_competitors(manifest["competitors"])

    output_directory = Path(manifest["output_directory"])
    results_directory = Path(manifest["results_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    results_directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    schedule = build_schedule(manifest["competitors"], manifest["num_configs"], rng)

    for index, quad in enumerate(schedule):
        contents = render_config(manifest, quad, index, rng)
        output_path = output_directory / f"{index:03d}.toml"
        output_path.write_text(contents)

    print(f"Wrote {len(schedule)} configs to {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
