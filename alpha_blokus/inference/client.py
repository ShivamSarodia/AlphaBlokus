from typing import Union
from omegaconf import DictConfig

# This trick makes the typechecker happy without actually importing the classes,
# because not all environments have the dependencies necessary to import the
# modules themselves.
if False:
    from alpha_blokus.inference.actors.torch import TorchInferenceActor
    from alpha_blokus.inference.actors.tensorrt import TensorRTInferenceActor


class InferenceClient:
    def __init__(
        self,
        inference_actor: Union["TorchInferenceActor", "TensorRTInferenceActor"],
        batch_size: int,
        cfg: DictConfig,
    ):
        return
