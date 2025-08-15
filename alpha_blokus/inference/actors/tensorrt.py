import ray


@ray.remote(num_gpus=1)
class TensorRTInferenceActor:
    def __init__(self, network_config, cfg):
        self.network_config = network_config
        self.cfg = cfg

    def maybe_create_initial_model(self):
        raise NotImplementedError("""
            implement this to check if the config indicates we need to create an
            initial model if missing, and create the model if so.
        """)
