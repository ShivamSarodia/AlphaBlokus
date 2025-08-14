import os
import ray

from alpha_blokus.inference.client import InferenceClient

def run(cfg):
    ray.init(log_to_driver=cfg["log_to_console"])
    print("Running with working directory:", os.getcwd())

    inference_clients = start_inference_actors(cfg)
    return inference_clients


def start_inference_actors(cfg) -> dict[str, InferenceClient]:
    inference_clients: dict[str, InferenceClient] = {}

    for network_name, network_config in cfg.get("networks", {}).items():
        # If we're doing gameplay, we need to start an inference actor and create a client for each network.
        if cfg.get("gameplay"):
            print(f"Starting inference actor for network '{network_name}'...")
            if network_config.get("backend") == "torch":
                from alpha_blokus.inference.actors.torch import TorchInferenceActor
                inference_actor = TorchInferenceActor.remote(network_config, cfg)

            inference_clients[network_name] = InferenceClient(inference_actor, network_config["batch_size"], cfg)
    
    return inference_clients