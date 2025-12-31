import torch
from torch import nn
import torch.nn.functional as F

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.save_onnx import SaveOnnxMixin


class ResidualBlock(nn.Module):
    def __init__(self, net_config: NetworkConfig):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
        )

    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))


class ValueHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.value_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.value_head_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                net_config.value_head_channels, net_config.value_head_flat_layer_width
            ),
            nn.ReLU(),
            nn.Linear(net_config.value_head_flat_layer_width, 4),
        )

    def forward(self, x):
        return self.layers(x)


class PolicyHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.policy_head_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.policy_head_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config.policy_head_channels,
                out_channels=game_config.num_piece_orientations,
                kernel_size=net_config.policy_convolution_kernel,
                stride=1,
                padding=net_config.policy_convolution_kernel // 2,
                # Include bias because we're not using batch norm.
                bias=True,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class NeuralNet(nn.Module, SaveOnnxMixin):
    def __init__(
        self,
        net_config: NetworkConfig,
        game_config: GameConfig,
    ):
        super().__init__()

        self.net_config = net_config
        self.game_config = game_config

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(net_config) for _ in range(net_config.residual_blocks)]
        )
        self.value_head = ValueHead(net_config, game_config)
        self.policy_head = PolicyHead(net_config, game_config)

    def forward(self, board):
        # Add an all-ones channel to the input for edge detection.
        ones = torch.ones(
            board.shape[0],
            1,
            self.game_config.board_size,
            self.game_config.board_size,
            device=board.device,
            dtype=board.dtype,
        )
        x = torch.cat([board, ones], dim=1)  # Shape: (batch, 5, board_size, board_size)

        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        return (
            self.value_head(x),
            self.policy_head(x),
        )
