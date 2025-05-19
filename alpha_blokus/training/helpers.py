import time
import math
import os
import random
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import namedtuple
from zipfile import BadZipFile

from alpha_blokus.training.load_games import load_game_file
from alpha_blokus.event_logger import log_event

FileInfo = namedtuple('FileInfo', ['path', 'size'])

class GameDataset:
    def __init__(self, device, cfg: dict, max_samples_in_memory=50000, use_logging=False, shuffle_files=True):
        self.device = device
        self.cfg = cfg
        self.max_samples_in_memory = max_samples_in_memory
        self.files: List[FileInfo] = []
        self.total_samples = 0
        self.use_logging = use_logging
        self.shuffle_files = shuffle_files

        board_size = self.cfg["game"]["board_size"]
        num_moves = self.cfg["game"]["num_moves"]
        self.loaded_boards = np.empty((max_samples_in_memory, 4, board_size, board_size), dtype=np.float32)
        self.loaded_policies = np.empty((max_samples_in_memory, num_moves), dtype=np.float32)
        self.loaded_values = np.empty((max_samples_in_memory, 4), dtype=np.float32)
        self.loaded_valid_moves = np.empty((max_samples_in_memory, num_moves), dtype=np.bool_)
        self.loaded_samples_count = 0
        self.used_indices = set()  # Track which indices have been used
    
    def add_file(self, path: str) -> int:
        """Add file metadata to the dataset. Returns number of samples added."""
        try:
            # Peek into the file to get the size without loading full data yet
            with np.load(path) as data:
                # Check if 'boards' key exists and get its length
                if 'boards' not in data:
                    raise KeyError("'boards' key not found in NPZ file")
                size = len(data['boards'])
            
            # Create file info (metadata only)
            file_info = FileInfo(path=path, size=size)
            
            self.files.append(file_info)
            self.total_samples += size
            return size
        except (BadZipFile, FileNotFoundError, KeyError, ValueError) as e:
            log_event("bad_or_missing_gamedata_file", {
                "path": path,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return 0
    
    def remove_oldest_file(self) -> int:
        """Remove the oldest file's metadata. Returns number of samples removed."""
        if not self.files:
            return 0
            
        removed_file = self.files.pop(0)
        removed_samples = removed_file.size
        self.total_samples -= removed_samples
        
        return removed_samples
    
    def force_shuffle(self):
        random.shuffle(self.files)
    
    def _rotate_files_in_memory(self):
        """Loads data from a subset of files into the pre-allocated numpy arrays."""
        if not self.files:
            raise ValueError("No files in dataset to load.")

        # Create a list of file indices
        file_indices = list(range(len(self.files)))
        if self.shuffle_files:
            self.force_shuffle()

        # Copy data directly into the pre-allocated arrays
        current_offset = 0
        for idx in file_indices:
            file_info = self.files[idx]
            try:
                with np.load(file_info.path) as data:
                    num_samples_in_file = len(data['boards'])
                    if num_samples_in_file != file_info.size:
                        raise ValueError(f"Size mismatch for {file_info.path}. Expected {file_info.size}, found {num_samples_in_file}.")

                    # Calculate how many samples we can fit from this file
                    remaining_space = self.max_samples_in_memory - current_offset
                    samples_to_copy = min(num_samples_in_file, remaining_space)
                    
                    if samples_to_copy <= 0:
                        break

                    # Copy data directly into the pre-allocated arrays
                    self.loaded_boards[current_offset:current_offset + samples_to_copy] = data['boards'][:samples_to_copy]
                    self.loaded_policies[current_offset:current_offset + samples_to_copy] = data['policies'][:samples_to_copy]
                    self.loaded_values[current_offset:current_offset + samples_to_copy] = data['values'][:samples_to_copy]
                    self.loaded_valid_moves[current_offset:current_offset + samples_to_copy] = data['valid_moves_array'][:samples_to_copy]
                    
                    current_offset += samples_to_copy

            except (BadZipFile, FileNotFoundError, KeyError, ValueError, OSError) as e:
                log_event("error_loading_gamedata_during_rotation", {
                    "path": file_info.path,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                continue
        
        # If we're not shuffling files, move the ones we loaded to the end of the 
        # list of files so we don't load them again.
        if not self.shuffle_files:
            self.files.extend(self.files[:idx])
            self.files = self.files[idx:]

        self.loaded_samples_count = current_offset
        assert current_offset <= self.max_samples_in_memory
        self.used_indices.clear()
                
        if self.use_logging:
            log_event("training_rotating_files", {
                "current_offset": current_offset,
                "max_samples_in_memory": self.max_samples_in_memory,
            })
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a random batch of samples from files currently in memory."""
        if self.loaded_samples_count == 0:
            # Attempt to load data if none is loaded (e.g., first call or after errors)
            print("No data loaded in memory, attempting rotation...")
            self._rotate_files_in_memory()
            if self.loaded_samples_count == 0:
                 raise ValueError("No samples available in memory even after rotation attempt.")

        # Generate random indices from unused samples
        available_indices = list(set(range(self.loaded_samples_count)) - self.used_indices)
        if len(available_indices) < batch_size:
            # If we don't have enough unused samples, rotate and try again
            self._rotate_files_in_memory()
            available_indices = list(range(self.loaded_samples_count))
        
        # Randomly select indices from available ones
        selected_indices = random.sample(available_indices, batch_size)
        self.used_indices.update(selected_indices)

        # Select data directly from the loaded numpy arrays
        boards_np = self.loaded_boards[selected_indices]
        policies_np = self.loaded_policies[selected_indices]
        values_np = self.loaded_values[selected_indices]
        valid_moves_np = self.loaded_valid_moves[selected_indices]

        # Convert the selected NumPy slices to PyTorch tensors
        boards = torch.from_numpy(boards_np).to(dtype=torch.float32, device=self.device)
        policies = torch.from_numpy(policies_np).to(dtype=torch.float32, device=self.device)
        values = torch.from_numpy(values_np).to(dtype=torch.float32, device=self.device)
        valid_moves = torch.from_numpy(valid_moves_np).to(dtype=torch.bool, device=self.device)
        
        return boards, policies, values, valid_moves

class TrainingLoop:
    def __init__(self, *, initial_model, initial_lifetime_loaded_samples, optimizer, device, gamedata_dir, compute_top_one=False, use_logging=True, cfg: dict):
        self.cfg = cfg
        self.model = initial_model
        self.optimizer = optimizer
        self.device = device
        self.gamedata_dir = gamedata_dir
        self.compute_top_one = compute_top_one
        self.use_logging = use_logging
        
        self.lifetime_loaded_samples = 0
        self.lifetime_trained_samples = 0

        # Store the loaded gamedata paths
        self.paths_loaded = set()

        # Initialize the dataset
        self.dataset = GameDataset(device, self.cfg, use_logging=use_logging)

        # Initialize the values.
        for path in self._all_gamedata_paths():
            if self.lifetime_loaded_samples < initial_lifetime_loaded_samples - self.cfg["training"]["sample_window"]:
                sample_count = int(path.split(".")[-2].split("_")[-1])
                self.lifetime_loaded_samples += sample_count
                self.paths_loaded.add(path)
            elif self.lifetime_loaded_samples < initial_lifetime_loaded_samples:
                self._load_next_gamedata_file()
            else:
                break
            self.paths_loaded.add(path)

        # Initialize the lifetime trained samples to the point where the model left off.
        while self._current_sampling_ratio() < self.cfg["training"]["sampling_ratio"]:
            self.lifetime_trained_samples += 5

        if self.use_logging:
            log_event("training_initialized", {
                "lifetime_loaded_samples": self.lifetime_loaded_samples,
                "lifetime_trained_samples": self.lifetime_trained_samples,
                "current_sampling_ratio": self._current_sampling_ratio(),
                "target_sampling_ratio": self.cfg["training"]["sampling_ratio"],
                "window_size_total_samples": self.dataset.total_samples,
                "window_size_files": len(self.dataset.files),
                "in_memory_samples": self.dataset.loaded_samples_count,
                "max_samples_in_memory": self.dataset.max_samples_in_memory,
            })

    def _current_sampling_ratio(self):
        if self.lifetime_loaded_samples <= self.cfg["training"]["minimum_window_size"]:
            return 100000
        else:
            return self.lifetime_trained_samples / (self.lifetime_loaded_samples - self.cfg["training"]["minimum_window_size"])

    def _all_gamedata_paths(self):
        os.makedirs(self.gamedata_dir, exist_ok=True)
        return sorted([
            os.path.join(self.gamedata_dir, filename)
            for filename in os.listdir(self.gamedata_dir)
            if filename.endswith(".npz")
        ])

    def _load_next_gamedata_file(self):
        all_paths = self._all_gamedata_paths()
        next_file = min([path for path in all_paths if path not in self.paths_loaded], default=None)
        if next_file is None:
            return None
        self.paths_loaded.add(next_file)

        # Add the file to our dataset
        samples_added = self.dataset.add_file(next_file)
        if samples_added == 0:
            return 0
        
        self.lifetime_loaded_samples += samples_added

        # Remove oldest file if window is too large
        window_size = self.dataset.total_samples
        if window_size > self.cfg["training"]["sample_window"]:
            samples_removed = self.dataset.remove_oldest_file()
            if self.use_logging:
                log_event("training_dropped_from_window", {
                    "num_dropped_samples": samples_removed,
                    "num_samples_remaining": window_size - samples_removed,
                })

        return samples_added

    def _run_batch(self):
        boards, policies, values, valid_moves = self.dataset.get_batch(self.cfg["training"]["batch_size"])

        pred_values, pred_policy = self.model(boards)

        # Mask invalid moves.
        if self.cfg["training"]["exclude_invalid_moves_from_loss"]:
            pred_policy[~valid_moves] = -1e6

        value_loss = nn.CrossEntropyLoss()(pred_values, values)
        policy_loss = self.cfg["training"]["policy_loss_weight"] * nn.CrossEntropyLoss()(pred_policy, policies)
        loss = value_loss + policy_loss

        if self.compute_top_one:
            value_correct = (pred_values.argmax(dim=1) == values.argmax(dim=1)).sum().item() / len(boards)
            policy_correct = (pred_policy.argmax(dim=1) == policies.argmax(dim=1)).sum().item() / len(boards)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.lifetime_trained_samples += len(boards)

        result = {
            "value_loss": value_loss.item() / len(boards),
            "policy_loss": policy_loss.item() / len(boards),
            "loss": loss.item() / len(boards),
            "lifetime_loaded_samples": self.lifetime_loaded_samples,
            "lifetime_trained_samples": self.lifetime_trained_samples,
            "current_sampling_ratio": self._current_sampling_ratio(),
        }

        if self.compute_top_one:
            result["value_correct"] = value_correct
            result["policy_correct"] = policy_correct

        return result

    def run_iteration(self):
        if self._current_sampling_ratio() < self.cfg["training"]["sampling_ratio"]:
            # Run a batch, which will increase the sampling ratio.
            return "trained", self._run_batch()
        else:
            # Read in a file, which will decrease the sampling ratio.
            data_loaded = self._load_next_gamedata_file()
            if data_loaded is None:
                return "no_new_data", None
            return "read_new_data", {
                "count": data_loaded,
            }
