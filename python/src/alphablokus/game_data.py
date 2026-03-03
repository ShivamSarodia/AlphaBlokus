from typing import List

import msgpack
import torch
import zstandard

from alphablokus.configs import GameConfig


def load_game_file(
    game_config: GameConfig,
    local_file_path: str,
    require_piece_availability: bool = False,
) -> tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    piece_availabilities = []
    with zstandard.open(local_file_path, "rb") as f:
        game_data_list = f.read()
        game_data_list = msgpack.unpackb(game_data_list)

        for row_idx, game_data in enumerate(game_data_list):
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
                piece_availabilities.append(
                    torch.ones((4, game_config.num_pieces), dtype=torch.float32)
                )
                continue

            piece_availability = torch.as_tensor(
                raw_piece_availability, dtype=torch.float32
            )
            if piece_availability.shape != (4, game_config.num_pieces):
                raise ValueError(
                    "Invalid piece_availability shape in "
                    f"{local_file_path} at row {row_idx}. Expected (4, "
                    f"{game_config.num_pieces}), got {tuple(piece_availability.shape)}"
                )
            if not torch.all(
                (piece_availability == 0.0) | (piece_availability == 1.0)
            ):
                raise ValueError(
                    "Non-binary piece_availability values in "
                    f"{local_file_path} at row {row_idx}"
                )
            piece_availabilities.append(piece_availability)

    return boards, values, policies, valid_masks, piece_availabilities


def load_game_files_to_tensor(
    game_config: GameConfig,
    local_file_paths: List[str],
    require_piece_availability: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    piece_availabilities = []
    for file_path in local_file_paths:
        board, value, policy, valid_mask, piece_availability = load_game_file(
            game_config,
            file_path,
            require_piece_availability=require_piece_availability,
        )
        boards += board
        values += value
        policies += policy
        valid_masks += valid_mask
        piece_availabilities += piece_availability
    return (
        torch.stack(boards),
        torch.stack(values),
        torch.stack(policies),
        torch.stack(valid_masks),
        torch.stack(piece_availabilities),
    )
