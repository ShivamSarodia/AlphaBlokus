from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import msgpack
import numpy as np
import zstandard


PERCENTILES = (10, 25, 50, 75, 90, 99)
QUANTILE_METHOD = "linear"


@dataclass(frozen=True)
class RankedMoveProbabilities:
    top1_probability: float
    top2_probability: float
    top3_probability: float


@dataclass(frozen=True)
class TemperatureRandomnessSummary:
    total_rows: int
    unique_game_count: int
    quantiles: dict[str, dict[str, float]]


def _validate_temperature(temperature: float) -> None:
    if not np.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")


def _load_game_rows(local_file_path: str | Path) -> list[dict]:
    with zstandard.open(str(local_file_path), "rb") as handle:
        return msgpack.unpackb(handle.read())


def compute_move_probabilities(
    visit_counts: Iterable[int | float],
    temperature: float,
) -> np.ndarray:
    _validate_temperature(temperature)

    counts = np.asarray(list(visit_counts), dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("visit_counts must be a 1D sequence")
    if counts.size == 0:
        raise ValueError("visit_counts must not be empty")
    if np.any(counts < 0.0):
        raise ValueError("visit_counts must be non-negative")

    if temperature == 0.0:
        probabilities = np.zeros_like(counts)
        probabilities[int(np.argmax(counts))] = 1.0
        return probabilities

    weights = np.power(counts, 1.0 / temperature)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("temperature-adjusted visit counts must sum to a positive value")
    return weights / weight_sum


def extract_ranked_move_probabilities(
    probabilities: Iterable[int | float],
) -> RankedMoveProbabilities:
    sorted_probabilities = np.sort(np.asarray(list(probabilities), dtype=np.float64))[::-1]

    return RankedMoveProbabilities(
        top1_probability=float(sorted_probabilities[0]) if sorted_probabilities.size >= 1 else 0.0,
        top2_probability=float(sorted_probabilities[1]) if sorted_probabilities.size >= 2 else 0.0,
        top3_probability=float(sorted_probabilities[2]) if sorted_probabilities.size >= 3 else 0.0,
    )


def compute_quantiles(
    values: Iterable[int | float],
    *,
    percentiles: tuple[int, ...] = PERCENTILES,
    method: str = QUANTILE_METHOD,
) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute quantiles for an empty collection")

    quantiles = np.percentile(array, percentiles, method=method)
    return {
        f"p{percentile:02d}": float(value)
        for percentile, value in zip(percentiles, quantiles, strict=True)
    }


def analyze_move_temperature_file(
    local_file_path: str | Path,
    temperature: float,
) -> TemperatureRandomnessSummary:
    _validate_temperature(temperature)

    rows = _load_game_rows(local_file_path)
    if not rows:
        raise ValueError(f"game data file is empty: {local_file_path}")

    game_ids: set[int] = set()
    top1_probabilities: list[float] = []
    top2_probabilities: list[float] = []
    top3_probabilities: list[float] = []

    for row in rows:
        game_ids.add(int(row["game_id"]))
        probabilities = compute_move_probabilities(row["visit_counts"], temperature)
        ranked = extract_ranked_move_probabilities(probabilities)
        top1_probabilities.append(ranked.top1_probability)
        top2_probabilities.append(ranked.top2_probability)
        top3_probabilities.append(ranked.top3_probability)

    return TemperatureRandomnessSummary(
        total_rows=len(rows),
        unique_game_count=len(game_ids),
        quantiles={
            "top1_probability": compute_quantiles(top1_probabilities),
            "top2_probability": compute_quantiles(top2_probabilities),
            "top3_probability": compute_quantiles(top3_probabilities),
        },
    )


def format_summary_report(
    *,
    input_path: str,
    temperature: float,
    summary: TemperatureRandomnessSummary,
) -> str:
    lines = [
        f"game_data_file: {input_path}",
        f"temperature: {temperature:.6f}",
        f"total_rows: {summary.total_rows}",
        f"unique_game_count: {summary.unique_game_count}",
        "",
    ]

    ordered_percentiles = [f"p{percentile:02d}" for percentile in PERCENTILES]
    header = "metric".ljust(20) + " ".join(label.rjust(10) for label in ordered_percentiles)
    lines.append(header)

    for metric_name in ("top1_probability", "top2_probability", "top3_probability"):
        quantiles = summary.quantiles[metric_name]
        values = " ".join(f"{quantiles[label]:10.6f}" for label in ordered_percentiles)
        lines.append(metric_name.ljust(20) + values)

    return "\n".join(lines)
