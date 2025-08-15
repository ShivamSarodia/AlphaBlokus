import os
import io
import torch
from typing import Optional

from alpha_blokus.model_stores.base import BaseModelStore, ModelFile
from alpha_blokus.model_stores.local_file_model_store import LocalFileModelStore
from alpha_blokus.model_stores.local_directory_model_store import (
    LocalDirectoryModelStore,
)
from alpha_blokus.torch_net import NeuralNet
from alpha_blokus.utils.event_logger import log_event


class TorchInferenceActor:
    def __init__(self, network_config, cfg):
        self.network_config = network_config
        self.cfg = cfg

        self.inference_dtype = getattr(torch, network_config["inference_dtype"])
        self.device = torch.device(network_config["device"])

        self.model = None
        self.current_model_file: Optional[ModelFile] = None
        self.model_store: BaseModelStore = None

        model_read_path = network_config["model_read_path"]
        if os.path.isfile(model_read_path):
            self.model_store = LocalFileModelStore(model_read_path)
        else:
            assert os.path.isdir(model_read_path), (
                "model read path must be a file or directory"
            )
            self.model_store = LocalDirectoryModelStore(
                model_path=model_read_path,
                model_extension=".pt",
                cache_duration=1.0,
                recency_threshold=15,
            )

    def evaluate_batch(self, boards):
        """
        Evaluate a batch of boards on the currently loaded model.
        """
        self.load_model_if_necessary()

        boards_tensor = torch.tensor(
            boards.copy(), dtype=self.inference_dtype, device=self.device
        )
        with torch.inference_mode():
            values_logits_tensor, policy_logits_tensor = self.model(boards_tensor)

        values = torch.softmax(values_logits_tensor, dim=1).cpu().numpy()
        policy_logits = policy_logits_tensor.cpu().numpy()

        return values, policy_logits

    def load_model_if_necessary(self, maybe_create=False):
        latest_model_file = self.model_store.cached_get_latest_model_file(
            include_recent=False
        )

        # If the latest model on disk is the same as the current model, there's nothing to do.
        if latest_model_file and self.current_model_file.path == latest_model_file.path:
            return

        # If there's a newer model on disk, load it.
        elif latest_model_file:
            assert (
                self.current_model_file.creation_time < latest_model_file.creation_time
            ), "loaded model is newer than latest model on disk"
            self._load_model_from_file(latest_model_file)

        # If we're allowed to create a new model, do so.
        elif maybe_create and not self.model:
            model = NeuralNet(self.network_config, self.cfg)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            self.current_model_file = self.model_store.create_model(
                "0.pt",
                buffer.getvalue(),
            )

        else:
            raise ValueError("no model to load")

    def _load_model_from_file(self, model_file: ModelFile):
        """
        Load a model from a path.
        """
        self.model = NeuralNet(self.network_config, self.cfg)
        self.model.load_state_dict(torch.load(model_file.path, weights_only=True))
        self.model.to(device=self.device, dtype=self.inference_dtype)
        self.model.eval()
        self.current_model_file = model_file
        log_event(
            "loaded_model", {"model_name": model_file.path.split("/")[-1].split(".")[0]}
        )
