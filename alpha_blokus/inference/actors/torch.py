import ray

@ray.remote(num_gpus=1)
class TorchInferenceActor:
    def __init__(self, network_config, cfg):
        self.network_config = network_config
        self.cfg = cfg

    @classmethod
    def remote(cls, network_config, cfg) -> "TorchInferenceActor":
        raise NotImplementedError("placeholder to make typechecker happy")