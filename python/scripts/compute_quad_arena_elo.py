#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGISTIC_SCALE = math.log(10.0) / 400.0
Z_95 = 1.959963984540054


@dataclass
class PairStats:
    games: float = 0.0
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Elo from QuadArena JSONL results.")
    parser.add_argument("--results-directory", required=True, help="Directory containing JSONL result files.")
    return parser.parse_args()


def iter_result_rows(results_directory: Path):
    for path in sorted(results_directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def aggregate_results(results_directory: Path):
    pairwise: dict[tuple[str, str], PairStats] = defaultdict(PairStats)
    games_played: dict[str, int] = defaultdict(int)
    pairwise_score_total: dict[str, float] = defaultdict(float)
    pairwise_comparison_count: dict[str, int] = defaultdict(int)

    for row in iter_result_rows(results_directory):
        agent_names = row["agent_names"]
        result = row["result"]

        for agent_name in agent_names:
            games_played[agent_name] += 1

        for left, right in itertools.combinations(range(len(agent_names)), 2):
            left_name = agent_names[left]
            right_name = agent_names[right]
            left_result = float(result[left])
            right_result = float(result[right])

            if left_result == 0.0 and right_result == 0.0:
                continue

            if left_result == right_result:
                if left_result <= 0.0:
                    continue
                left_score = 0.5
            elif left_result > right_result:
                left_score = 1.0
            else:
                left_score = 0.0

            ordered = tuple(sorted((left_name, right_name)))
            stats = pairwise[ordered]
            stats.games += 1.0
            if ordered[0] == left_name:
                stats.score += left_score
            else:
                stats.score += 1.0 - left_score

            pairwise_score_total[left_name] += left_score
            pairwise_score_total[right_name] += 1.0 - left_score
            pairwise_comparison_count[left_name] += 1
            pairwise_comparison_count[right_name] += 1

    return pairwise, games_played, pairwise_score_total, pairwise_comparison_count


def fit_elo(pairwise: dict[tuple[str, str], PairStats]):
    agents = sorted({name for pair in pairwise for name in pair})
    if len(agents) < 2:
        raise ValueError("Need at least two agents with pairwise results to fit Elo")

    index = {agent: idx for idx, agent in enumerate(agents)}
    variable_count = len(agents) - 1
    x = np.zeros(variable_count, dtype=np.float64)

    def ratings_from_x(vec: np.ndarray) -> np.ndarray:
        ratings = np.zeros(len(agents), dtype=np.float64)
        ratings[:-1] = vec
        return ratings

    for _ in range(50):
        ratings = ratings_from_x(x)
        gradient = np.zeros(variable_count, dtype=np.float64)
        hessian = np.zeros((variable_count, variable_count), dtype=np.float64)

        for (left_name, right_name), stats in pairwise.items():
            left = index[left_name]
            right = index[right_name]
            diff = LOGISTIC_SCALE * (ratings[left] - ratings[right])
            probability = 1.0 / (1.0 + math.exp(-diff))
            variance = stats.games * LOGISTIC_SCALE * LOGISTIC_SCALE * probability * (1.0 - probability)
            delta = LOGISTIC_SCALE * (stats.score - stats.games * probability)

            if left < variable_count:
                gradient[left] += delta
                hessian[left, left] -= variance
            if right < variable_count:
                gradient[right] -= delta
                hessian[right, right] -= variance
            if left < variable_count and right < variable_count:
                hessian[left, right] += variance
                hessian[right, left] += variance

        if variable_count == 0:
            break

        step = np.linalg.pinv(-hessian) @ gradient
        x += step
        if np.max(np.abs(step)) < 1e-8:
            break

    raw_ratings = ratings_from_x(x)
    centered_ratings = raw_ratings - raw_ratings.mean() + 1500.0

    if variable_count == 0:
        covariance_centered = np.zeros((1, 1), dtype=np.float64)
    else:
        covariance_x = np.linalg.pinv(-hessian)
        transform = np.full((len(agents), variable_count), -1.0 / len(agents), dtype=np.float64)
        for row in range(variable_count):
            transform[row, row] += 1.0
        covariance_centered = transform @ covariance_x @ transform.T

    return agents, centered_ratings, covariance_centered


def print_table(
    agents: list[str],
    ratings: np.ndarray,
    covariance: np.ndarray,
    games_played: dict[str, int],
    pairwise_score_total: dict[str, float],
    pairwise_comparison_count: dict[str, int],
) -> None:
    rows = []
    for index, agent in enumerate(agents):
        variance = max(0.0, float(covariance[index, index]))
        stddev = math.sqrt(variance)
        half_width = Z_95 * stddev
        score_total = pairwise_score_total.get(agent, 0.0)
        comparison_count = pairwise_comparison_count.get(agent, 0)
        score_average = score_total / comparison_count if comparison_count else 0.0
        rows.append(
            (
                float(ratings[index]),
                agent,
                games_played.get(agent, 0),
                score_total,
                score_average,
                float(ratings[index] - half_width),
                float(ratings[index] + half_width),
            )
        )

    rows.sort(reverse=True)
    print(
        f"{'agent':30} {'elo':>10} {'ci95':>23} {'games':>8} "
        f"{'score_total':>12} {'score_avg':>10}"
    )
    for rating, agent, games, score_total, score_average, lower, upper in rows:
        print(
            f"{agent:30} {rating:10.1f} [{lower:7.1f}, {upper:7.1f}] "
            f"{games:8d} {score_total:12.2f} {score_average:10.3f}"
        )


def main() -> int:
    args = parse_args()
    results_directory = Path(args.results_directory)
    pairwise, games_played, pairwise_score_total, pairwise_comparison_count = aggregate_results(
        results_directory
    )
    agents, ratings, covariance = fit_elo(pairwise)
    print_table(
        agents,
        ratings,
        covariance,
        games_played,
        pairwise_score_total,
        pairwise_comparison_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
