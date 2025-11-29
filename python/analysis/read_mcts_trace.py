"""
Utility to read MCTS trace logs.

Each line in the log is a JSON object serialized from `src/agents/mcts/tracing.rs::MCTSTrace`.
This script maps each line back to a typed Python dataclass mirroring the Rust enum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, TypedDict, Union, Any, Callable
import itertools


# ----- Python mirrors of the Rust MCTSTrace variants -----


@dataclass
class StartedSearch:
    type: Literal["started_search"] = "started_search"
    root_node_id: int = 0
    state: Dict[str, Any] | List[Any] | Any = None
    search_id: int = 0
    is_fast_move: bool = False
    num_rollouts: int = 0


@dataclass
class CreatedNode:
    type: Literal["created_node"] = "created_node"
    node_id: int = 0
    search_id: int = 0
    num_valid_moves: int = 0
    move_index_to_array_index: Dict[int, int] | None = None
    array_index_to_move_index: List[int] | None = None
    array_index_to_player_pov_move_index: List[int] | None = None


@dataclass
class NetworkEvalResult:
    type: Literal["network_eval_result"] = "network_eval_result"
    node_id: int = 0
    search_id: int = 0
    value: List[float] | None = None
    policy: List[float] | None = None


@dataclass
class SelectedMoveByUcb:
    type: Literal["selected_move_by_ucb"] = "selected_move_by_ucb"
    node_id: int = 0
    search_id: int = 0
    move_index: int = 0
    array_index: int = 0
    children_value_sums: List[List[float]] | None = None
    children_visit_counts: List[int] | None = None
    children_visit_counts_sum: int = 0
    children_prior_probabilities: List[float] | None = None
    exploration_scores: List[float] | None = None
    exploitation_scores: List[float] | None = None


@dataclass
class SelectedMoveToPlay:
    type: Literal["selected_move_to_play"] = "selected_move_to_play"
    node_id: int = 0
    search_id: int = 0
    temperature: float = 0.0
    children_visit_counts: List[int] | None = None
    children_value_sums: List[List[float]] | None = None
    children_visit_counts_sum: int = 0
    children_prior_probabilities: List[float] | None = None
    move_index: int = 0
    array_index: int = 0


@dataclass
class AddedChild:
    type: Literal["added_child"] = "added_child"
    parent_node_id: int = 0
    child_node_id: int = 0
    search_id: int = 0
    move_index: int = 0


MCTSTrace = Union[
    StartedSearch,
    CreatedNode,
    NetworkEvalResult,
    SelectedMoveByUcb,
    SelectedMoveToPlay,
    AddedChild,
]


# ----- Parsing helpers -----


class _TraceDict(TypedDict, total=False):
    type: str


TRACE_TYPE_MAP = {
    "started_search": StartedSearch,
    "created_node": CreatedNode,
    "network_eval_result": NetworkEvalResult,
    "selected_move_by_ucb": SelectedMoveByUcb,
    "selected_move_to_play": SelectedMoveToPlay,
    "added_child": AddedChild,
}


def parse_trace_obj(obj: _TraceDict) -> MCTSTrace:
    """Convert a single decoded JSON object into a typed trace dataclass."""
    trace_type = obj.get("type")
    cls = TRACE_TYPE_MAP.get(trace_type or "")
    if cls is None:
        raise ValueError(f"Unknown trace type {trace_type!r}")
    return cls(**{k: v for k, v in obj.items() if k != "type"})


def iter_trace_log(
    path: Path | str, newest_first: bool = False, rows: int | None = None
) -> Iterable[MCTSTrace]:
    """Yield traces from a log file written by `record_mcts_trace`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        print("Reading file into memory...")
        lines = handle.readlines()
        if newest_first:
            lines.reverse()
        if rows is not None:
            lines = lines[:rows]
        print("Done reading file into memory.")
        print("Parsing lines...")
        for line_num, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield parse_trace_obj(obj)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse trace line {line_num}: {exc}"
                ) from exc
        print("Done parsing lines.")


def get_search_ids(logs: List[MCTSTrace], max: int) -> List[int]:
    """Get the recent search ids."""
    search_ids = []
    for log in logs:
        search_id = getattr(log, "search_id", None)
        if search_id is not None and search_id not in search_ids:
            search_ids.append(search_id)
        if len(search_ids) >= max:
            break
    return search_ids


def filter_by_search_id(logs: List[MCTSTrace], search_id: int) -> List[MCTSTrace]:
    """Filter logs by search id."""
    return [log for log in logs if log.search_id == search_id]


def filter_to_latest_search(logs: List[MCTSTrace]) -> List[MCTSTrace]:
    """Filter logs to the latest search."""
    return filter_by_search_id(logs, logs[-1].search_id)


def find_sole_log(
    logs: List[MCTSTrace], condition: Callable[[MCTSTrace], bool]
) -> MCTSTrace:
    """Find the sole log entry that matches the condition."""
    matches = [log for log in logs if condition(log)]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 log entry to match condition, got {len(matches)}")
    return matches[0]
