import torch
from torch.export import Dim


class SaveOnnxMixin:
    def save_onnx(self, path: str):
        self.to("cpu")
        dummy_batch_size = 128
        dummy_input = (
            torch.randn(
                dummy_batch_size,
                4,
                self.game_config.board_size,
                self.game_config.board_size,
            ),
        )
        batch_size = Dim("batch_size")
        torch.onnx.export(
            self,
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
