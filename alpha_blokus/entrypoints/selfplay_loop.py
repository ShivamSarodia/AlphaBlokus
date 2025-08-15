import os
import ray

from alpha_blokus.inference.client import InferenceClient


def run(cfg):
    if not ray.is_initialized():
        ray.init(log_to_driver=cfg["log_to_console"])
    print("Running with working directory:", os.getcwd())

    inference_clients = start_inference_actors(cfg)
    return inference_clients


def start_inference_actors(cfg) -> dict[str, InferenceClient]:
    inference_clients: dict[str, InferenceClient] = {}

    for network_name, network_config in cfg.get("networks", {}).items():
        if cfg.get("gameplay"):
            print(f"Starting inference actor for network '{network_name}'...")
            if network_config.get("backend") == "torch":
                from alpha_blokus.inference.actors.torch import TorchInferenceActor
                actor_class = TorchInferenceActor
            elif network_config.get("backend") == "tensorrt":
                from alpha_blokus.inference.actors.tensorrt import TensorRTInferenceActor
                actor_class = TensorRTInferenceActor
            else:
                raise ValueError(f"Unsupported backend: {network_config.get('backend')}")

            inference_actor = ray.remote(actor_class).remote(network_config, cfg)

            # Load or create the initial model, and block until completion.
            ray.get(inference_actor.load_model_if_necessary.remote(maybe_create=True))

            # Create the associated inference client.
            inference_clients[network_name] = InferenceClient(inference_actor, network_config["batch_size"], cfg)

            print("Done starting inference actor for network '{network_name}'.")
    
    return inference_clients