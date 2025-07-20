import numpy as np
import ray
from typing import Tuple
import torch
import os
import time
from typing import Dict

from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.event_logger import log_event

@ray.remote(
    num_gpus=1,
    runtime_env={ "nsight": "default" },
    # {
    #     "gpu-metrics-devices": "all",
    # }},
)
class InferenceActor:
    def __init__(self, network_config: Dict, cfg: dict) -> None:
        self.network_config = network_config

        self.cfg = cfg
        self.inference_dtype = getattr(torch, network_config["inference_dtype"])
        self.device = torch.device(network_config["device"])        

        self.model = None
        self.model_path = None
        self.last_checked_for_new_model = 0

        self._maybe_load_model(exclude_recent=False)
        
        if self.model is None:
            raise ValueError("Missing model")
    
    def evaluate_batch(self, boards) -> Tuple[np.ndarray, np.ndarray]:
        self._maybe_load_model()

        start_evaluation = time.perf_counter()

        # Include an extra .copy() here so we don't get a scary PyTorch warning about 
        # non-writeable tensors.
        boards_tensor = torch.from_numpy(boards.copy()).to(dtype=self.inference_dtype, device=self.device)
        with torch.inference_mode():
            values_logits_tensor, policy_logits_tensor = self.model(boards_tensor)
        
        values = torch.softmax(values_logits_tensor, dim=1).cpu().numpy()
        policy_logits = policy_logits_tensor.cpu().numpy()

        if self.network_config["log_gpu_evaluation"]:
            log_event("gpu_evaluation", {
                "duration": time.perf_counter() - start_evaluation,
                "batch_size": boards.shape[0],
            })

        return values, policy_logits
    
    def _maybe_load_model(self, exclude_recent=True):
        # First, if it hasn't been long enough since we last checked for a new model,
        # don't check again.
        current_time = time.time()
        time_since_last_check = current_time - self.last_checked_for_new_model
        if (
            # A negative check interval means we never check for new models, so if
            # we see that then return immediately if a model is already loaded.
            (self.network_config["new_model_check_interval"] <= 0 and self.model) or
            time_since_last_check < self.network_config["new_model_check_interval"]
        ):
            return
        
        # Ok, we're gonna actually check for a new model.
        self.last_checked_for_new_model = current_time
        
        # Next, find the path of the latest model. If it's the same as the latest model
        # don't reload it.
        latest_model_path = self._find_latest_model_path(exclude_recent)

        if latest_model_path is None:
            log_event("no_model_found")
            return

        if latest_model_path == self.model_path:
            log_event("no_new_model")
            return 
        
        # Finally, we have a new model to load!
        self._load_model(latest_model_path)

    def _find_latest_model_path(self, exclude_recent=True):
        model_path_or_dir = self.network_config["model_read_path"]
        if os.path.isfile(model_path_or_dir):
            assert model_path_or_dir.endswith(".pt")
            return model_path_or_dir

        # Otherwise, look up the latest model in a directory.
        current_time = time.time()
        model_paths = []
        with os.scandir(model_path_or_dir) as entries:
            for entry in entries:
                if (
                    entry.is_file() and 
                    entry.name.endswith(".pt") and
                    # Exclude files that were created in the last 15 seconds, because
                    # they may not be fully written yet.
                    (not exclude_recent or current_time - entry.stat().st_mtime > 15)
                ):
                    model_paths.append(entry.path)
            
        return max(model_paths)
    
    def _load_model(self, path):
        self.model = NeuralNet(self.network_config, self.cfg)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.to(device=self.device, dtype=self.inference_dtype)
        self.model.eval()
        self.model_path = path
        log_event(
            "loaded_model",
            { "model_name": path.split("/")[-1].split(".")[0] }
        )