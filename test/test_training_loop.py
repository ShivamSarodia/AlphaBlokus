import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from alpha_blokus.training.helpers import TrainingLoop


class MockModel(nn.Module):
    def __init__(self, board_size=20, num_moves=100):
        super().__init__()

        # Simple network with minimal parameters
        self.conv = nn.Conv2d(4, 4, kernel_size=1)  # 1x1 conv just for parameters
        self.value_head = nn.Linear(4 * board_size * board_size, 4)
        self.policy_head = nn.Linear(4 * board_size * board_size, num_moves)

        self.board_size = board_size
        self.num_moves = num_moves

        # Tracking for test assertions
        self.forward_calls = 0
        self.inputs = []

    def forward(self, x):
        """Mock forward method that returns predictable outputs for testing."""
        self.forward_calls += 1
        self.inputs.append(x.clone().detach())

        batch_size = x.shape[0]

        # Do minimal computation - we don't care about the actual output values
        # Just need something that creates a gradient
        x = self.conv(x)
        x_flat = x.reshape(batch_size, -1)

        values = self.value_head(x_flat)
        policies = self.policy_head(x_flat)

        return values, policies

    def reset_tracking(self):
        """Reset the tracking counters for a fresh test."""
        self.forward_calls = 0
        self.inputs = []


class TestTrainingLoop:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "game": {
                "board_size": 20,
                "num_moves": 100
            },
            "training": {
                "sample_window": 100,
                "minimum_window_size": 25,
                "sampling_ratio": 2.0,
                "batch_size": 3,
                "policy_loss_weight": 1.0
            }
        }
    
    @pytest.fixture
    def game_data_dir(self, tmp_path):
        """Create a directory with mock game data files in chronological order."""
        data_dir = tmp_path / "game_data"
        data_dir.mkdir()
        
        # Create multiple mock game data files with timestamps in the names
        for i in range(100):
            # Game data naming convention: game_data_<timestamp>_<sample_count>.npz
            file_path = data_dir / f"game_data_1000{i}_5.npz"
            
            # Create mock data arrays (5 samples per file)
            num_samples = 5
            boards = np.random.rand(num_samples, 4, 20, 20).astype(np.float32)
            policies = np.random.rand(num_samples, 100).astype(np.float32)
            # Normalize policies for softmax
            policies = policies / policies.sum(axis=1, keepdims=True)
            values = np.zeros((num_samples, 4), dtype=np.float32)
            # Set one position to 1.0 for each sample (one-hot encoding)
            for j in range(num_samples):
                values[j, j % 4] = 1.0
            valid_moves_array = np.ones((num_samples, 100), dtype=bool)
            
            # Save as npz file
            np.savez(
                file_path,
                boards=boards,
                policies=policies,
                values=values,
                valid_moves_array=valid_moves_array,
                unused_pieces=np.ones((num_samples, 4, 21), dtype=np.uint8)  # Required by load_game_file
            )
        
        return str(data_dir)
    
    def test_training_loop_integration(self, mock_config, game_data_dir):
        """Integration test for TrainingLoop with file loading and iterations."""
        device = torch.device("cpu")

        # Create a mock model and optimizer
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Disable real logging during tests
        with patch('alpha_blokus.event_logger.log_event'):
            # Initialize with no samples loaded yet
            training_loop = TrainingLoop(
                initial_model=model,
                initial_lifetime_loaded_samples=0,
                optimizer=optimizer,
                device=device,
                gamedata_dir=game_data_dir,
                compute_top_one=True,
                use_logging=False,
                cfg=mock_config
            )

            # Verify initial state
            assert training_loop.lifetime_loaded_samples == 0
            assert training_loop.lifetime_trained_samples == 0
            assert model.forward_calls == 0  # No training has occurred yet

            # Initial window is built during the first five iterations because each
            # file has 5 samples, so until the 7th iteration we're not able to compute
            # the training ratio.
            for i in range(6):
                result_type, result_data = training_loop.run_iteration()
                assert result_type == "read_new_data"

            sampling_ratios = []
            while True:
                result_type, result_data = training_loop.run_iteration()
                if result_type != "trained":
                    break

                assert result_data["lifetime_loaded_samples"] == 30
                sampling_ratios.append(result_data["current_sampling_ratio"])

            # Verify that the sampling ratio was below 2.0 for all iterations except
            # the last one.
            for s in sampling_ratios[:-1]:
                assert s < 2.0
            assert sampling_ratios[-1] > 2.0

            # Now we'll read new data.
            while result_type == "read_new_data":
                result_type, result_data = training_loop.run_iteration()

            # Once we've finished reading new data, we should have a sampling ratio
            # below 2.0 again.
            assert result_type == "trained"
            assert result_data["current_sampling_ratio"] < 2.0

            # Verify that training occurred
            assert model.forward_calls > 0
            assert len(model.inputs) > 0

            # Check that the inputs to the model match the expected shape
            for input_batch in model.inputs:
                assert input_batch.shape[0] == mock_config["training"]["batch_size"]  # Batch size
                assert input_batch.shape[1] == 4  # 4 channels (player POV)
                assert input_batch.shape[2:] == (mock_config["game"]["board_size"], mock_config["game"]["board_size"])

            # Now, train several more times.
            while result_type != "no_new_data":
                result_type, result_data = training_loop.run_iteration()

            # Check that the sampling ratio is above 2.0, which is what triggered
            # us to try reading new data.
            assert training_loop._current_sampling_ratio() > 2.0

            # Check that sliding window is working and we dropped oldest files
            assert training_loop.dataset.total_samples <= mock_config["training"]["sample_window"]

            # Verify the oldest files were removed (FIFO)
            all_paths = sorted([
                os.path.join(game_data_dir, filename)
                for filename in os.listdir(game_data_dir)
                if filename.endswith(".npz")
            ])

            # Assert not all files are in the dataset.
            assert len(training_loop.dataset.files) < len(all_paths)

            # Current dataset should contain newest files, not oldest
            dataset_paths = [file_info.path for file_info in training_loop.dataset.files]
            for path in dataset_paths:
                # Check that we kept newer files, not older ones
                assert path in all_paths[-len(dataset_paths):]