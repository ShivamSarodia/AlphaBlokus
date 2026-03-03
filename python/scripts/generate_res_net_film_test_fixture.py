from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from alphablokus.res_net_film import NeuralNet  # noqa: E402


@dataclass
class TinyGameConfig:
    board_size: int = 5
    num_moves: int = 958
    num_pieces: int = 21
    num_piece_orientations: int = 91


@dataclass
class TinyFilmNetConfig:
    model_class: str = "resnet_film"
    main_body_channels: int = 16
    film_channels: int = 16
    residual_blocks: int = 2
    value_head_channels: int = 8
    value_head_flat_layer_width: int = 32
    policy_head_channels: int = 8
    policy_convolution_kernel: int = 1


def main() -> None:
    torch.manual_seed(0)

    game_config = TinyGameConfig()
    net_config = TinyFilmNetConfig()
    model = NeuralNet(net_config, game_config).to("cpu").eval()

    output_path = ROOT / "static" / "networks" / "trivial_net_tiny.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_onnx(str(output_path), "cpu")

    print(f"Wrote {output_path}")
    data_path = output_path.with_suffix(".onnx.data")
    if data_path.exists():
        print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
