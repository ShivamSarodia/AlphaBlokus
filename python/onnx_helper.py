import torch
import torch.nn as nn
from torch.export import Dim

from game_config import GameConfig
from trivial_net import TrivialNet


def save_as_onnx(net: nn.Module, game_config: GameConfig, path: str):
    dummy_input = (torch.randn(128, 4, game_config.board_size, game_config.board_size),)
    batch_size = Dim("batch_size")
    torch.onnx.export(
        net,
        dummy_input,
        path,
        dynamo=True,
        input_names=["board"],
        output_names=["value", "policy"],
        dynamic_shapes={
            "board": {
                0: batch_size,
            },
        },
    )


if __name__ == "__main__":
    game_config = GameConfig("configs/generate_move_data/tiny.toml")
    net = TrivialNet(game_config)
    save_as_onnx(net, game_config, "static/networks/trivial_net.onnx")
