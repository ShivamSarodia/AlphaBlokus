from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import msgpack
import torch
import zstandard

from alphablokus.configs import GameConfig


@dataclass(frozen=True)
class GameSample:
    board: torch.Tensor
    game_result: torch.Tensor
    q_value: torch.Tensor
    policy: torch.Tensor
    valid_policy_mask: torch.Tensor
    piece_availability: torch.Tensor


@dataclass(frozen=True)
class GameBatch:
    board: torch.Tensor
    game_result: torch.Tensor
    q_value: torch.Tensor
    policy: torch.Tensor
    valid_policy_mask: torch.Tensor
    piece_availability: torch.Tensor


def stack_game_samples(samples: Sequence[GameSample]) -> GameBatch:
    if not samples:
        raise ValueError("Cannot stack empty sample sequence.")
    return GameBatch(
        board=torch.stack([sample.board for sample in samples]),
        game_result=torch.stack([sample.game_result for sample in samples]),
        q_value=torch.stack([sample.q_value for sample in samples]),
        policy=torch.stack([sample.policy for sample in samples]),
        valid_policy_mask=torch.stack(
            [sample.valid_policy_mask for sample in samples]
        ),
        piece_availability=torch.stack(
            [sample.piece_availability for sample in samples]
        ),
    )


def _load_piece_availability(
    *,
    game_data: dict,
    game_config: GameConfig,
    row_idx: int,
    local_file_path: str,
    require_piece_availability: bool,
) -> torch.Tensor:
    raw_piece_availability = game_data.get("piece_availability")
    has_piece_availability = (
        raw_piece_availability is not None and len(raw_piece_availability) > 0
    )
    if not has_piece_availability:
        if require_piece_availability:
            raise ValueError(
                "Missing or empty piece_availability in "
                f"{local_file_path} at row {row_idx}"
            )
        return torch.ones((4, game_config.num_pieces), dtype=torch.float32)

    piece_availability = torch.as_tensor(raw_piece_availability, dtype=torch.float32)
    if piece_availability.shape != (4, game_config.num_pieces):
        raise ValueError(
            "Invalid piece_availability shape in "
            f"{local_file_path} at row {row_idx}. Expected (4, "
            f"{game_config.num_pieces}), got {tuple(piece_availability.shape)}"
        )
    if not torch.all((piece_availability == 0.0) | (piece_availability == 1.0)):
        raise ValueError(
            "Non-binary piece_availability values in "
            f"{local_file_path} at row {row_idx}"
        )
    return piece_availability


def load_game_samples(
    game_config: GameConfig,
    local_file_path: str,
    require_piece_availability: bool = False,
) -> List[GameSample]:
    samples: List[GameSample] = []
    with zstandard.open(local_file_path, "rb") as f:
        game_data_list = f.read()
        game_data_list = msgpack.unpackb(game_data_list)

        for row_idx, game_data in enumerate(game_data_list):
            board = [game_data["board"]["slices"][i]["cells"] for i in range(4)]
            board_tensor = torch.as_tensor(board, dtype=torch.float32)
            game_result = torch.as_tensor(game_data["game_result"], dtype=torch.float32)
            if game_result.shape != (4,):
                raise ValueError(
                    f"Invalid game_result shape in {local_file_path} at row {row_idx}. "
                    f"Expected (4,), got {tuple(game_result.shape)}"
                )
            raw_q_value = game_data.get("q_value")
            if raw_q_value is None or len(raw_q_value) == 0:
                q_value = game_result
            else:
                q_value = torch.as_tensor(raw_q_value, dtype=torch.float32)
                if q_value.shape != (4,):
                    raise ValueError(
                        f"Invalid q_value shape in {local_file_path} at row {row_idx}. "
                        f"Expected (4,), got {tuple(q_value.shape)}"
                    )

            policy_target = torch.zeros(
                (
                    game_config.num_piece_orientations,
                    game_config.board_size,
                    game_config.board_size,
                ),
                dtype=torch.float32,
            )

            valid_move_tuples = torch.as_tensor(
                game_data["valid_move_tuples"], dtype=torch.int64
            )
            visit_counts = torch.as_tensor(
                game_data["visit_counts"], dtype=torch.float32
            )

            valid_mask = torch.zeros(
                (
                    game_config.num_piece_orientations,
                    game_config.board_size,
                    game_config.board_size,
                ),
                dtype=torch.bool,
            )

            policy_target[
                valid_move_tuples[:, 0],
                valid_move_tuples[:, 1],
                valid_move_tuples[:, 2],
            ] = visit_counts

            policy_sum = policy_target.sum().item()
            assert policy_sum > 0.0, (
                f"Policy target sum is 0 in file: {local_file_path}"
            )
            policy = policy_target / policy_sum
            valid_mask[
                valid_move_tuples[:, 0],
                valid_move_tuples[:, 1],
                valid_move_tuples[:, 2],
            ] = True
            piece_availability = _load_piece_availability(
                game_data=game_data,
                game_config=game_config,
                row_idx=row_idx,
                local_file_path=local_file_path,
                require_piece_availability=require_piece_availability,
            )
            samples.append(
                GameSample(
                    board=board_tensor,
                    game_result=game_result,
                    q_value=q_value,
                    policy=policy,
                    valid_policy_mask=valid_mask,
                    piece_availability=piece_availability,
                )
            )

    return samples


def load_game_file(
    game_config: GameConfig,
    local_file_path: str,
    require_piece_availability: bool = False,
) -> List[GameSample]:
    return load_game_samples(
        game_config,
        local_file_path,
        require_piece_availability=require_piece_availability,
    )


def load_game_files_to_tensor(
    game_config: GameConfig,
    local_file_paths: List[str],
    require_piece_availability: bool = False,
) -> GameBatch:
    all_samples: List[GameSample] = []
    for file_path in local_file_paths:
        file_samples = load_game_samples(
            game_config,
            file_path,
            require_piece_availability=require_piece_availability,
        )
        all_samples.extend(file_samples)
    return stack_game_samples(all_samples)
