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


@dataclass
class NetworkConfig:
    main_body_channels: int
    residual_blocks: int
    value_head_channels: int
    value_head_flat_layer_width: int
    policy_head_channels: int
    policy_convolution_kernel: int

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        self.main_body_channels = data["network"]["main_body_channels"]
        self.residual_blocks = data["network"]["residual_blocks"]
        self.value_head_channels = data["network"]["value_head_channels"]
        self.value_head_flat_layer_width = data["network"][
            "value_head_flat_layer_width"
        ]
        self.policy_head_channels = data["network"]["policy_head_channels"]
        self.policy_convolution_kernel = data["network"]["policy_convolution_kernel"]
