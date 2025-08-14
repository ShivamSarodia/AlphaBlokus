# The network accepts an occupancies array and outputs both 
# a policy prediction and value. It expects inputs and outputs
# to be in the perspective of the current player. (So e.g. the 
# occupancies array should be rolled and rotated to the current
# player's perspective.)

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from alpha_blokus.moves_data import moves_data

class Debug(nn.Module):
    def __init__(self, label: str = ""):
        super().__init__()
        self.label = label

    def forward(self, x):
        # max_activations = torch.max(x, dim=1)[0]
        # print(max_activations[0])
        for x_i in x:
            print(x_i)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, net_config: OmegaConf):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config["main_body_channels"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config["main_body_channels"],
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config["main_body_channels"]),
        )
    
    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))
    
class PolicyFlatten(nn.Module):
    def __init__(self, cfg: OmegaConf):
        super().__init__()

        self.cfg = cfg

    def forward(self, x):
        return x[
            :,
            moves_data(self.cfg)["piece_orientation_indices"],
            moves_data(self.cfg)["center_placement_x"],
            moves_data(self.cfg)["center_placement_y"],
        ]

class NeuralNet(nn.Module):
    def __init__(
        self,
        net_config: OmegaConf,
        cfg: OmegaConf,
        flatten_policy: bool = True,
        add_ones_channel: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        self.inference_dtype = getattr(torch, net_config["inference_dtype"])
        self.flatten_policy = flatten_policy
        self.add_ones_channel = add_ones_channel

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=net_config["main_body_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
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
                bias=False,
            ),
            nn.BatchNorm2d(net_config["value_head_channels"]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                net_config["value_head_channels"] * self.cfg.game.board_size * self.cfg.game.board_size,
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config["policy_head_channels"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config["policy_head_channels"],
                out_channels=self.cfg.game.num_piece_orientations,
                kernel_size=net_config["policy_convolution_kernel"],
                stride=1,
                padding=net_config["policy_convolution_kernel"] // 2,
                # Include bias because we're not using batch norm.
                bias=True,
            ),
            PolicyFlatten(self.cfg) if self.flatten_policy else nn.Identity(),
        )

    def forward(self, occupancies):
        # Add an all-ones channel to the input for edge detection.
        if self.add_ones_channel:
            ones = torch.ones(
                occupancies.shape[0],
                1,
                self.cfg.game.board_size,
                self.cfg.game.board_size,
                device=occupancies.device,
                dtype=self.inference_dtype,
            )
            x = torch.cat([occupancies, ones], dim=1)  # Shape: (batch, 5, board_size, board_size)
        else:
            x = occupancies
        
        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return (
            self.value_head(x),
            self.policy_head(x),
        )

def add_ones_channel(occupancies):
    ones = torch.ones(
        occupancies.shape[0],
        1,
        occupancies.shape[2],
        occupancies.shape[3],
        device=occupancies.device,
        dtype=occupancies.dtype,
    )
    return torch.cat([occupancies, ones], dim=1)  # Shape: (batch, 5, board_size, board_size)