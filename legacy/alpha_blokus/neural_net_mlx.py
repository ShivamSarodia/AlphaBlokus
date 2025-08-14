import mlx.core as mx
import mlx.nn as nn

from typing import Dict

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return mx.flatten(x, 1)

class ResidualBlock(nn.Module):
    def __init__(self, net_config: Dict):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm(net_config["main_body_channels"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm(net_config["main_body_channels"]),
        )
    
    def __call__(self, x):
        return nn.relu(x + self.convolutional_block(x))

class NeuralNetMLX(nn.Module):
    def __init__(self, net_config: Dict):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm(net_config["main_body_channels"]),
            nn.ReLU(),
        )

        # Previously had a ModuleList here, but we'll see if this works
        # without it.
        self.residual_blocks = [
            ResidualBlock(net_config)
            for _ in range(net_config["residual_blocks"])
        ]

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["value_head_channels"],
                kernel_size=1,
            ),
            nn.BatchNorm(net_config["value_head_channels"]),
            nn.ReLU(),
            Flatten(),
            nn.Linear(
                net_config["value_head_channels"] * BOARD_SIZE * BOARD_SIZE,
                net_config["value_head_flat_layer_width"],
            ),
            nn.ReLU(), 
            nn.Linear(
                net_config["value_head_flat_layer_width"],
                4,
            ),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["policy_head_channels"],
                kernel_size=1,
            ),
            nn.BatchNorm(net_config["policy_head_channels"]),
            nn.ReLU(),
            Flatten(),
            nn.Linear(
                BOARD_SIZE * BOARD_SIZE * net_config["policy_head_channels"],
                NUM_MOVES,
            ),
        )

    def __call__(self, boards):
        # boards is an (N, 4, BOARD_SIZE, BOARD_SIZE) array.
        # We need to convert it to (N, BOARD_SIZE, BOARD_SIZE, 4)
        # by moving the channel dimension to the last dimension to match
        # the shape expected by the convolutional block by MLX conventions.

        # boards = mx.transpose(boards, (0, 2, 3, 1))

        x = self.convolutional_block(boards)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.value_head(x), self.policy_head(x)
