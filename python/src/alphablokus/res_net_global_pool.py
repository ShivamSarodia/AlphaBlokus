import torch
from torch import nn
import torch.nn.functional as F

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.save_onnx import SaveOnnxMixin


class GlobalPoolingBias(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(3 * channels, channels)

    def forward(self, x):
        # Global pooling with mean, mean scaled by board width, and max.
        mean = x.mean(dim=(2, 3))
        mean_scaled = mean * x.shape[2]
        max_pool = x.amax(dim=(2, 3))
        pooled = torch.cat([mean, mean_scaled, max_pool], dim=1)
        bias = self.fc(pooled).view(x.shape[0], x.shape[1], 1, 1)
        return x + bias


def _gpool_block_indices(num_blocks: int) -> set[int]:
    if num_blocks <= 0:
        return set()
    if num_blocks <= 2:
        return set(range(num_blocks))
    return {0, num_blocks // 2, num_blocks - 1}


class ResidualBlock(nn.Module):
    def __init__(self, net_config: NetworkConfig, use_gpool: bool):
        super().__init__()
        channels = net_config.main_body_channels

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.gpool = GlobalPoolingBias(channels) if use_gpool else None
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        if self.gpool is not None:
            out = self.gpool(out)
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class ValueHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        channels = net_config.value_head_channels

        self.conv1 = nn.Conv2d(
            in_channels=net_config.main_body_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.gpool = GlobalPoolingBias(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels, net_config.value_head_flat_layer_width)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(net_config.value_head_flat_layer_width, 4)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.gpool(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        return self.fc2(x)


class PolicyHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        channels = net_config.policy_head_channels

        self.conv1 = nn.Conv2d(
            in_channels=net_config.main_body_channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.gpool = GlobalPoolingBias(channels)
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=game_config.num_piece_orientations,
            kernel_size=net_config.policy_convolution_kernel,
            stride=1,
            padding=net_config.policy_convolution_kernel // 2,
            # Include bias because we're not using batch norm.
            bias=True,
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.gpool(x)
        return self.conv2(x)


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
        gpool_blocks = _gpool_block_indices(net_config.residual_blocks)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(net_config, use_gpool=i in gpool_blocks)
                for i in range(net_config.residual_blocks)
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

    config_path = "configs/training/full_vast_simulated_position.toml"
    output_path = "s3://alpha-blokus/full_v2/models_untrained/res_net_conv_value_position_global_pool.onnx"
    device = "cpu"

    model = NeuralNet(NetworkConfig(config_path), GameConfig(config_path))
    with from_localized(output_path) as localized_output_path:
        model.save_onnx(localized_output_path, device=device)
