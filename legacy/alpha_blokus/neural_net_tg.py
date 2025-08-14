from typing import Dict

import tinygrad.nn as nn
from tinygrad.tensor import Tensor

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]

class Debug:
    def __init__(self, label: str):
        self.label = label

    def __call__(self, x):
        print(self.label, x.shape)
        return x

class Flatten:
    def __call__(self, x):
        return x.flatten(start_dim=1)
    
class ReLU:
    def __call__(self, x):
        return x.relu()

class ResidualBlock:
    def __init__(self, net_config: Dict):
        self.conv1 = nn.Conv2d(
            net_config["main_body_channels"],
            net_config["main_body_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm(net_config["main_body_channels"])
        self.conv2 = nn.Conv2d(
            net_config["main_body_channels"],
            net_config["main_body_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm(net_config["main_body_channels"])

    def __call__(self, x):
        residual = x
        x = self.conv1(x).relu()
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return (x + residual).relu()

class NeuralNet:
    def __init__(self, net_config: Dict):
        self.conv_block = [
            nn.Conv2d(4, net_config["main_body_channels"], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(net_config["main_body_channels"]),
            ReLU()
        ]
        self.residual_blocks = [ResidualBlock(net_config) for _ in range(net_config["residual_blocks"])]
        self.value_head = [
            nn.Conv2d(
                net_config["main_body_channels"],
                net_config["value_head_channels"],
                kernel_size=1,
            ),
            nn.BatchNorm(net_config["value_head_channels"]),
            ReLU(),
            Flatten(),
            nn.Linear(
                net_config["value_head_channels"] * BOARD_SIZE * BOARD_SIZE,
                net_config["value_head_flat_layer_width"],
            ),
            ReLU(),
            nn.Linear(
                net_config["value_head_flat_layer_width"],
                4,
            ),
        ]
        self.policy_head = [
            nn.Conv2d(
                net_config["main_body_channels"],
                net_config["policy_head_channels"],
                kernel_size=1,
            ),
            nn.BatchNorm(net_config["policy_head_channels"]),
            ReLU(),
            Flatten(),
            nn.Linear(
                BOARD_SIZE * BOARD_SIZE * net_config["policy_head_channels"],
                NUM_MOVES,
            ),
        ]

    def __call__(self, occupancies):
        x = occupancies
        for layer in self.conv_block:
            x = layer(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        value = x
        for layer in self.value_head:
            value = layer(value)

        policy = x
        for layer in self.policy_head:
            policy = layer(policy)

        return value, policy
