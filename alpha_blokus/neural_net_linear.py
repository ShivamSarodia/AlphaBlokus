# The network accepts an occupancies array and outputs both 
# a policy prediction and value. It expects inputs and outputs
# to be in the perspective of the current player. (So e.g. the 
# occupancies array should be rolled and rotated to the current
# player's perspective.)

from typing import Dict
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# True only when importing an older model that had bias here.
BIAS = True

class Debug(nn.Module):
    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def forward(self, x):
        print(self.label, x.shape)
        return x

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
                bias=BIAS,
            ),
            nn.BatchNorm2d(net_config["main_body_channels"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=BIAS,
            ),
            nn.BatchNorm2d(net_config["main_body_channels"]),
        )
    
    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))

class NeuralNet(nn.Module):
    def __init__(self, net_config: OmegaConf, config: OmegaConf):
        super().__init__()

        self.config = config

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=BIAS,
            ),
            nn.BatchNorm2d(net_config["main_body_channels"]),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(net_config)
            for _ in range(net_config["residual_blocks"])
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["value_head_channels"],
                kernel_size=1,
                bias=BIAS,
            ),
            nn.BatchNorm2d(net_config["value_head_channels"]),
            nn.ReLU(),
            nn.Flatten(),
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
                bias=BIAS,
            ),
            nn.BatchNorm2d(net_config["policy_head_channels"]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                BOARD_SIZE * BOARD_SIZE * net_config["policy_head_channels"],
                NUM_MOVES,
            ),
        )

    def forward(self, occupancies):
        x = self.convolutional_block(occupancies)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return (
            self.value_head(x),
            self.policy_head(x),
        )
