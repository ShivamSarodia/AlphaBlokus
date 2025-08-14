import numpy as np
import ray
from typing import Tuple
import torch
import os
import time
from typing import Dict
import threading
import queue
from concurrent.futures import Future
import asyncio

from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.event_logger import log_event

@ray.remote(
    num_gpus=1,
    runtime_env={
        "nsight": {
        "gpu-metrics-devices": "all",
    }},
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

        # Queue to eliminate GPU idle gaps
        self.work_queue = queue.Queue()
        self.shutdown_event = threading.Event()

        self._maybe_load_model(exclude_recent=False)

        if self.model is None:
            raise ValueError("Missing model")

        # Start worker thread that keeps GPU busy
        self.worker_threads = [
            threading.Thread(target=self._worker_loop, daemon=True) for _ in range(2)
        ]
        for worker_thread in self.worker_threads:
            worker_thread.start()

    async def evaluate_batch(self, boards) -> Tuple[np.ndarray, np.ndarray]:
        self._maybe_load_model()

        # Create future and enqueue work
        result_future = Future()
        self.work_queue.put((boards, result_future))

        # Await result without blocking other async operations
        return await asyncio.wrap_future(result_future)

    def _worker_loop(self):
        """Keep GPU busy by continuously processing work queue."""
        stream = torch.cuda.Stream()

        while not self.shutdown_event.is_set():
            try:
                # Get next work item (blocks until available)
                boards, result_future = self.work_queue.get(timeout=0.1)
            except queue.Empty:
                continue  # Check shutdown and try again

            try:
                # Process immediately - no gaps
                start_time = time.perf_counter()
                with torch.cuda.stream(stream):
                    boards_tensor = torch.from_numpy(boards.copy()).to(
                        dtype=self.inference_dtype, device=self.device)

                    with torch.inference_mode():
                        values_logits, policy_logits = self.model(boards_tensor)

                    values = torch.softmax(values_logits, dim=1).cpu().numpy()
                    policy = policy_logits.cpu().numpy()

                if self.network_config["log_gpu_evaluation"]:
                    log_event("gpu_evaluation", {
                        "duration": time.perf_counter() - start_time,
                        "batch_size": boards.shape[0],
                    })

                # Return result
                result_future.set_result((values, policy))
            except Exception as e:
                result_future.set_exception(e)

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
