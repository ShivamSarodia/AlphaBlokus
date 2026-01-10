import torch
from torch import nn
import torch.nn.functional as F

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.save_onnx import SaveOnnxMixin


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, net_config: NetworkConfig, reduction: int = 2):
        super().__init__()
        channels = net_config.main_body_channels
        bottleneck_channels = max(1, channels // reduction)

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
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
                in_channels=6,
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
            [
                ResidualBottleneckBlock(net_config)
                for _ in range(net_config.residual_blocks)
            ]
        )
        self.value_head = ValueHead(net_config, game_config)
        self.policy_head = PolicyHead(net_config, game_config)

    def forward(self, board):
        # Add (x, y) positional channels in [0, 1]
        coords = torch.linspace(
            0.0,
            1.0,
            self.game_config.board_size,
            device=board.device,
            dtype=board.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (N, N)
        pos = (
            torch.stack([xx, yy], dim=0).unsqueeze(0).expand(board.shape[0], -1, -1, -1)
        )
        x = torch.cat([board, pos], dim=1)  # (batch, 6, board_size, board_size)

        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        return (
            self.value_head(x),
            self.policy_head(x),
        )


if __name__ == "__main__":
    from alphablokus.files import from_localized

    config_path = "configs/training/standalone_bottleneck_double.toml"
    output_path = (
        "s3://alpha-blokus/full_v2/models_untrained/res_net_bottleneck_double.onnx"
    )
    device = "cpu"

    model = NeuralNet(NetworkConfig(config_path), GameConfig(config_path))
    with from_localized(output_path) as localized_output_path:
        model.save_onnx(localized_output_path, device=device)
