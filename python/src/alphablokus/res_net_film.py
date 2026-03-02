import torch
from torch import nn
import torch.nn.functional as F
from torch.export import Dim

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.save_onnx import SaveOnnxMixin


class ResidualBlock(nn.Module):
    def __init__(self, net_config: NetworkConfig):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=net_config.main_body_channels,
            out_channels=net_config.main_body_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(net_config.main_body_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=net_config.main_body_channels,
            out_channels=net_config.main_body_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(net_config.main_body_channels)

    def forward(self, x, gamma, beta):
        residual = self.conv_1(x)
        residual = self.bn_1(residual)
        residual = F.relu(residual)
        residual = self.conv_2(residual)
        residual = self.bn_2(residual)
        residual = residual * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.relu(x + residual)


class ValueHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=net_config.main_body_channels + 2,
            out_channels=net_config.value_head_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(net_config.value_head_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(
            net_config.value_head_channels, net_config.value_head_flat_layer_width
        )
        self.fc_2 = nn.Linear(net_config.value_head_flat_layer_width, 4)

    def forward(self, x, gamma, beta):
        x = self.conv(x)
        x = self.bn(x)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        return self.fc_2(x)


class PolicyHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=net_config.main_body_channels + 2,
            out_channels=net_config.policy_head_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(net_config.policy_head_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=net_config.policy_head_channels,
            out_channels=game_config.num_piece_orientations,
            kernel_size=net_config.policy_convolution_kernel,
            stride=1,
            padding=net_config.policy_convolution_kernel // 2,
            bias=True,
        )

    def forward(self, x, gamma, beta):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x = F.relu(x)
        return self.conv_2(x)


class NeuralNet(nn.Module, SaveOnnxMixin):
    def __init__(
        self,
        net_config: NetworkConfig,
        game_config: GameConfig,
    ):
        super().__init__()

        assert net_config.film_channels is not None, (
            "network.film_channels must be set for resnet_film"
        )
        assert net_config.residual_blocks > 0, (
            "resnet_film expects at least one residual block"
        )

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
            [ResidualBlock(net_config) for _ in range(net_config.residual_blocks)]
        )
        self.film_stage_count = len(self.residual_blocks)
        self.trunk_film_param_count = (
            self.film_stage_count * 2 * self.net_config.main_body_channels
        )
        self.value_head_film_param_count = 2 * self.net_config.value_head_channels
        self.policy_head_film_param_count = 2 * self.net_config.policy_head_channels
        self.total_film_param_count = (
            self.trunk_film_param_count
            + self.value_head_film_param_count
            + self.policy_head_film_param_count
        )

        self.piece_availability_encoder = nn.Sequential(
            nn.Linear(4 * game_config.num_pieces, net_config.film_channels),
            nn.ReLU(),
        )
        self.film_projection = nn.Linear(
            net_config.film_channels,
            self.total_film_param_count,
        )
        nn.init.normal_(self.film_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.film_projection.bias)
        self.value_head = ValueHead(net_config, game_config)
        self.policy_head = PolicyHead(net_config, game_config)

    @staticmethod
    def _build_pos_channels(
        batch_size: int, board_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, board_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, board, piece_availability):
        if board.ndim != 4:
            raise ValueError(f"board must have rank 4, got shape {tuple(board.shape)}")
        if board.shape[1:] != (
            4,
            self.game_config.board_size,
            self.game_config.board_size,
        ):
            raise ValueError(
                "board must have shape "
                f"(B, 4, {self.game_config.board_size}, {self.game_config.board_size}), "
                f"got {tuple(board.shape)}"
            )
        if piece_availability.ndim != 3:
            raise ValueError(
                "piece_availability must have rank 3, got shape "
                f"{tuple(piece_availability.shape)}"
            )
        if piece_availability.shape[0] != board.shape[0]:
            raise ValueError(
                "piece_availability batch size must match board batch size, got "
                f"{piece_availability.shape[0]} and {board.shape[0]}"
            )
        if piece_availability.shape[1:] != (4, self.game_config.num_pieces):
            raise ValueError(
                "piece_availability must have shape "
                f"(B, 4, {self.game_config.num_pieces}), got "
                f"{tuple(piece_availability.shape)}"
            )

        trunk_pos = self._build_pos_channels(
            board.shape[0], self.game_config.board_size, board.device, board.dtype
        )
        x = torch.cat([board, trunk_pos], dim=1)
        x = self.convolutional_block(x)

        film_embedding = self.piece_availability_encoder(
            piece_availability.reshape(piece_availability.shape[0], -1)
        )
        all_film_params = self.film_projection(film_embedding)
        trunk_film_params, value_head_film_params, policy_head_film_params = torch.split(
            all_film_params,
            [
                self.trunk_film_param_count,
                self.value_head_film_param_count,
                self.policy_head_film_param_count,
            ],
            dim=1,
        )
        film_params = trunk_film_params.view(
            board.shape[0], self.film_stage_count, 2, self.net_config.main_body_channels
        )
        for i, residual_block in enumerate(self.residual_blocks):
            x = residual_block(x, film_params[:, i, 0], film_params[:, i, 1])

        head_pos = self._build_pos_channels(
            x.shape[0], x.shape[-1], x.device, x.dtype
        )
        value_input = torch.cat([x, head_pos], dim=1)
        policy_input = torch.cat([x, head_pos], dim=1)
        value_head_film_params = value_head_film_params.view(
            board.shape[0], 2, self.net_config.value_head_channels
        )
        policy_head_film_params = policy_head_film_params.view(
            board.shape[0], 2, self.net_config.policy_head_channels
        )

        return (
            self.value_head(
                value_input, value_head_film_params[:, 0], value_head_film_params[:, 1]
            ),
            self.policy_head(
                policy_input, policy_head_film_params[:, 0], policy_head_film_params[:, 1]
            ),
        )

    def save_onnx(self, path: str, device: str):
        self.eval()
        dummy_batch_size = 128
        dummy_board = torch.randn(
            dummy_batch_size,
            4,
            self.game_config.board_size,
            self.game_config.board_size,
        ).to(device)
        dummy_piece_availability = torch.randn(
            dummy_batch_size,
            4,
            self.game_config.num_pieces,
        ).to(device)

        batch_size = Dim("batch_size", max=1024)
        torch.onnx.export(
            self,
            (dummy_board, dummy_piece_availability),
            path,
            dynamo=True,
            input_names=["board", "piece_availability"],
            output_names=["value", "policy"],
            dynamic_shapes={
                "board": {
                    0: batch_size,
                },
                "piece_availability": {
                    0: batch_size,
                },
            },
        )
