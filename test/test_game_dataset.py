import os
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from alpha_blokus.training.helpers import GameDataset

class TestGameDataset:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "game": {
                "board_size": 20,  # Standard Blokus board size
                "num_moves": 100   # Arbitrary number for testing
            }
        }
    
    @pytest.fixture
    def mock_game_data(self, tmp_path):
        """Create mock game data files for testing."""
        # Create a temporary directory for test files
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Create two mock game data files
        for i in range(2):
            file_path = data_dir / f"game_data_{i}.npz"
            
            # Create mock data arrays
            num_samples = 5
            boards = np.random.rand(num_samples, 4, 20, 20).astype(np.float32)
            policies = np.random.rand(num_samples, 100).astype(np.float32)
            values = np.random.rand(num_samples, 4).astype(np.float32)
            valid_moves_array = np.random.randint(0, 2, (num_samples, 100)).astype(bool)
            
            # Save as npz file
            np.savez(
                file_path,
                boards=boards,
                policies=policies,
                values=values,
                valid_moves_array=valid_moves_array
            )
        
        return str(data_dir)
    
    def test_add_file(self, mock_config, mock_game_data):
        """Test adding a file to the dataset."""
        device = torch.device("cpu")
        dataset = GameDataset(device, mock_config)
        
        # Get the path to one of the mock files
        file_path = os.path.join(mock_game_data, "game_data_0.npz")
        
        # Add the file to the dataset
        added_samples = dataset.add_file(file_path)
        
        assert added_samples == 5
        assert len(dataset.files) == 1
        assert dataset.total_samples == 5
        
    def test_remove_oldest_file(self, mock_config, mock_game_data):
        """Test removing the oldest file from the dataset."""
        device = torch.device("cpu")
        dataset = GameDataset(device, mock_config)

        # Add two files
        file1 = os.path.join(mock_game_data, "game_data_0.npz")
        file2 = os.path.join(mock_game_data, "game_data_1.npz")

        dataset.add_file(file1)
        dataset.add_file(file2)

        assert len(dataset.files) == 2
        assert dataset.total_samples == 10

        # Store the original files to check which one gets removed
        first_file_path = dataset.files[0].path
        second_file_path = dataset.files[1].path

        # Remove the oldest file
        removed_samples = dataset.remove_oldest_file()

        assert removed_samples == 5
        assert len(dataset.files) == 1
        assert dataset.total_samples == 5

        # Verify the first file was removed (FIFO order)
        assert dataset.files[0].path == second_file_path
        assert first_file_path not in [f.path for f in dataset.files]
    
    def test_get_batch(self, mock_config, mock_game_data):
        """Test getting a batch of samples."""
        device = torch.device("cpu")
        max_samples_in_memory = 5
        dataset = GameDataset(device, mock_config, max_samples_in_memory=max_samples_in_memory)
        
        # Add two files
        file1 = os.path.join(mock_game_data, "game_data_0.npz")
        file2 = os.path.join(mock_game_data, "game_data_1.npz")
        
        dataset.add_file(file1)
        dataset.add_file(file2)

        assert dataset.loaded_samples_count == 0
        
        # Mock _rotate_files_in_memory to track calls
        rotate_calls = 0
        original_rotate = dataset._rotate_files_in_memory
        def mock_rotate():
            nonlocal rotate_calls
            rotate_calls += 1
            original_rotate()
        dataset._rotate_files_in_memory = mock_rotate
        
        # Get several batches. This will trigger _rotate_files_in_memory() internally
        # several times.
        batch_size = 2
        num_batches = 10
        for _ in range(num_batches):
            boards, policies, values, valid_moves = dataset.get_batch(batch_size)
        
        assert rotate_calls == num_batches // (max_samples_in_memory // batch_size)
        
        # Check types and shapes
        assert isinstance(boards, torch.Tensor)
        assert isinstance(policies, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(valid_moves, torch.Tensor)
        
        assert boards.shape == (batch_size, 4, 20, 20)
        assert policies.shape == (batch_size, 100)
        assert values.shape == (batch_size, 4)
        assert valid_moves.shape == (batch_size, 100)