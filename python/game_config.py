import tomllib

from dataclasses import dataclass


@dataclass
class GameConfig:
    board_size: int
    num_moves: int
    num_pieces: int
    num_piece_orientations: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.board_size = data["game"]["board_size"]
        self.num_moves = data["game"]["num_moves"]
        self.num_pieces = data["game"]["num_pieces"]
        self.num_piece_orientations = data["game"]["num_piece_orientations"]
