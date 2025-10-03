import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import GameConfig
from save_onnx import SaveOnnxMixin


class TrivialNet(nn.Module, SaveOnnxMixin):
    """
    A trivial neural network that pretty much just flattens and then returns some values/policy.
    """

    def __init__(self, game_config: GameConfig):
        super().__init__()
        self.game_config = game_config

        self.linear = nn.Linear(
            self.game_config.board_size * self.game_config.board_size * 4,
            64,
        )
        self.values = nn.Linear(64, 4)
        self.policy = nn.Linear(
            64,
            self.game_config.num_piece_orientations
            * self.game_config.board_size
            * self.game_config.board_size,
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert board.shape[1:] == (
            4,
            self.game_config.board_size,
            self.game_config.board_size,
        )

        x = torch.flatten(board, start_dim=1)
        x = self.linear(x)
        x = F.relu(x)
        values = self.values(x)
        policy = self.policy(x)
        return (
            values,
            torch.reshape(
                policy,
                (
                    policy.shape[0],
                    self.game_config.num_piece_orientations,
                    self.game_config.board_size,
                    self.game_config.board_size,
                ),
            ),
        )


if __name__ == "__main__":
    model = TrivialNet(GameConfig("configs/training/full.toml"))
    model.save_onnx("static/networks/trivial_net_full.onnx")
