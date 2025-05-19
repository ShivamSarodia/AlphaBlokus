import ray
import os
import time
import torch
from typing import Dict

from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.event_logger import log_event
from alpha_blokus.training.helpers import TrainingLoop

@ray.remote
class TrainingActor:
    def __init__(self, gamedata_path, cfg: dict) -> None:
        self.gamedata_path = gamedata_path
        self.cfg = cfg
        self.last_read_game_file_path = ""

    def run(self):
        # First, limit training to just one CPU so we don't hog resources needed for self-play.
        torch.set_num_threads(1)

        # Start by loading the base version of the model to train.
        # This is the latest model in the model directory.
        latest_model_path = self._find_latest_model_path()
        
        model = NeuralNet(self.cfg["networks"][self.cfg["training"]["network_name"]], self.cfg)
        model.to(self.cfg["training"]["device"])
        
        if latest_model_path is not None:
            latest_model_sample_count = int(latest_model_path.split("/")[-1].split(".")[0])
            model.load_state_dict(torch.load(latest_model_path, weights_only=True))
        else:
            latest_model_sample_count = 0
            
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["training"]["learning_rate"])

        training_loop = TrainingLoop(
            initial_model=model,
            initial_lifetime_loaded_samples=latest_model_sample_count,
            optimizer=optimizer,
            device=self.cfg["training"]["device"],
            gamedata_dir=self.gamedata_path,
            cfg=self.cfg,
        )
        previous_lifetime_loaded_samples = training_loop.lifetime_loaded_samples

        while True:
            action, result = training_loop.run_iteration()
            if action == "trained":
                log_event("training_batch", result)
            elif action == "read_new_data":
                log_event("training_loaded_new_data", result)
            elif action == "no_new_data":
                log_event("training_no_new_data", {})
                time.sleep(self.cfg["training"]["new_data_check_interval"])
            
            lifetime_loaded_samples = training_loop.lifetime_loaded_samples
            if lifetime_loaded_samples // self.cfg["training"]["samples_per_generation"] > previous_lifetime_loaded_samples // self.cfg["training"]["samples_per_generation"]:
                model_name = str(lifetime_loaded_samples).zfill(9)
                model_dir = self.cfg["training"]["model_write_directory"]
                os.makedirs(model_dir, exist_ok=True)
                assert os.path.isdir(model_dir)
                model_path = os.path.join(model_dir, f"{model_name}.pt")
                torch.save(model.state_dict(), model_path)
                log_event("saved_model", {
                    "cumulative_window_fed": lifetime_loaded_samples,
                    "model_name": model_name,
                })
            previous_lifetime_loaded_samples = lifetime_loaded_samples

    def _find_latest_model_path(self):
        model_path_or_dir = self.cfg["networks"][self.cfg["training"]["network_name"]]["model_read_path"]
        if model_path_or_dir is False:
            return None
            
        if os.path.isfile(model_path_or_dir):
            assert model_path_or_dir.endswith(".pt")
            return model_path_or_dir

        # Otherwise, look up the latest model in a directory.
        model_paths = [
            os.path.join(model_path_or_dir, filename)
            for filename in os.listdir(model_path_or_dir)
            if (
                os.path.isfile(os.path.join(model_path_or_dir, filename)) and 
                filename.endswith(".pt")
            )
        ]
        return max(model_paths)
