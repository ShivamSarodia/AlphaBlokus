import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import GameConfig
from save_onnx import SaveOnnxMixin


class TrivialNet(nn.Module, SaveOnnxMixin):
    """
    A trivial neural network that pretty much just flattens and then returns some values/policy.

    The trivial network currently returns the policy as a flattened array, rather than 91 x N x N,
    but we should probably instead return the policy as 91 x N x N.
    """

    def __init__(self, game_config: GameConfig):
        super().__init__()
        self.game_config = game_config

        self.linear = nn.Linear(
            self.game_config.board_size * self.game_config.board_size * 4,
            64,
        )
        self.values = nn.Linear(64, 4)
        self.policy = nn.Linear(64, self.game_config.num_moves)

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
        return (values, policy)
