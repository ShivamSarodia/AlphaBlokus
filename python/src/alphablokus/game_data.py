from typing import List

import msgpack
import torch
import zstandard

from alphablokus.configs import GameConfig


def load_game_file(
    game_config: GameConfig,
    local_file_path: str,
) -> tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    with zstandard.open(local_file_path, "rb") as f:
        game_data_list = f.read()
        game_data_list = msgpack.unpackb(game_data_list)

        for game_data in game_data_list:
            board = [game_data["board"]["slices"][i]["cells"] for i in range(4)]
            boards.append(torch.as_tensor(board, dtype=torch.float32))

            values.append(torch.as_tensor(game_data["game_result"]))

            policy_target = torch.zeros(
                (
                    game_config.num_piece_orientations,
                    game_config.board_size,
                    game_config.board_size,
                ),
                dtype=torch.float32,
            )

            valid_move_tuples = torch.as_tensor(
                game_data["valid_move_tuples"], dtype=torch.int32
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
            policies.append(policy_target / policy_sum)
            valid_mask[
                valid_move_tuples[:, 0],
                valid_move_tuples[:, 1],
                valid_move_tuples[:, 2],
            ] = True
            valid_masks.append(valid_mask)

    return boards, values, policies, valid_masks


def load_game_files_to_tensor(
    game_config: GameConfig,
    local_file_paths: List[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    for file_path in local_file_paths:
        board, value, policy, valid_mask = load_game_file(game_config, file_path)
        boards += board
        values += value
        policies += policy
        valid_masks += valid_mask
    return (
        torch.stack(boards),
        torch.stack(values),
        torch.stack(policies),
        torch.stack(valid_masks),
    )
