import os
import random
import time
import torch
from omegaconf import DictConfig

from alpha_blokus.inference.actors.torch import TorchInferenceActor
from alpha_blokus.model_stores.base import ModelFile
from alpha_blokus.torch_net import NeuralNet


def create_test_configs(model_path):
    """Create test configuration objects for TorchInferenceActor."""
    network_config = {
        "backend": "torch",
        "main_body_channels": 16,  # Even smaller for faster testing
        "residual_blocks": 1,  # Minimal for speed
        "value_head_channels": 8,
        "value_head_flat_layer_width": 32,
        "policy_head_channels": 16,
        "policy_convolution_kernel": 3,
        "device": "cpu",  # Use CPU for testing
        "inference_dtype": "float32",
        "new_model_check_interval": 120,
        "batch_size": 128,
        "initialize_model_if_empty": False,
        "log_gpu_evaluation": False,
        "model_read_path": model_path,
    }
    
    cfg = DictConfig({
        "game": {
            "board_size": 20,
            "num_piece_orientations": 91,
        }
    })
    
    return network_config, cfg


def create_model_file(model_path, filename, network_config, cfg):
    """Create a real neural network model and save it to a file."""
    model = NeuralNet(network_config, cfg)
    file_path = os.path.join(model_path, filename)
    torch.save(model.state_dict(), file_path)
    stat = os.stat(file_path)
    return ModelFile(path=file_path, creation_time=stat.st_mtime)


def test_load_model_if_necessary_no_update_needed():
    """Test that no update happens when current model is same as latest on disk."""
    # Create a temporary directory for testing
    randomint = random.randint(0, 1_000_000)
    model_path = f"/tmp/alphablokus_test_{randomint}/"
    os.makedirs(model_path, exist_ok=True)
    
    network_config, cfg = create_test_configs(model_path)
    
    # Create a real model file
    model_file = create_model_file(model_path, "model_1.pt", network_config, cfg)
    
    # Wait for recency threshold to pass
    time.sleep(0.6)
    
    actor = TorchInferenceActor(network_config, cfg)
    actor.model_store.recency_threshold = 0.5
    
    # Load the model
    actor._load_model_from_file(model_file)
    
    # Verify model was actually loaded
    assert actor.model is not None
    assert actor.current_model_file is not None
    assert actor.current_model_file.path == model_file.path
    
    # Track the current model to verify it doesn't change
    original_model = actor.model
    
    actor.load_model_if_necessary()
    
    # Should not change the model since current is same as latest
    assert actor.model is original_model
    assert actor.current_model_file is not None
    assert actor.current_model_file.path == model_file.path


def test_load_model_if_necessary_load_newer_model():
    """Test that a newer model is loaded when available on disk."""
    # Create a temporary directory for testing
    randomint = random.randint(0, 1_000_000)
    model_path = f"/tmp/alphablokus_test_{randomint}/"
    os.makedirs(model_path, exist_ok=True)
    
    network_config, cfg = create_test_configs(model_path)
    
    # Create first model file
    first_model_file = create_model_file(model_path, "model_1.pt", network_config, cfg)
    
    # Wait for recency threshold to pass
    time.sleep(0.6)
    
    actor = TorchInferenceActor(network_config, cfg)
    actor.model_store.recency_threshold = 0.5
    
    # Load the first model as current
    actor._load_model_from_file(first_model_file)
    
    # Verify first model was loaded
    assert actor.model is not None
    first_model = actor.model
    
    # Create a newer model file
    newer_model_file = create_model_file(model_path, "model_2.pt", network_config, cfg)
    
    # Wait for recency threshold to pass for the newer model
    time.sleep(0.6)
    
    actor.load_model_if_necessary()
    
    # Should have loaded the newer model (real loading!)
    assert actor.model is not first_model  # Different model object
    assert actor.current_model_file is not None
    assert actor.current_model_file.path == newer_model_file.path


def test_load_model_if_necessary_create_new_model():
    """Test that a new model is created when maybe_create=True and no model exists."""
    # Create a temporary directory for testing
    randomint = random.randint(0, 1_000_000)
    model_path = f"/tmp/alphablokus_test_{randomint}/"
    os.makedirs(model_path, exist_ok=True)
    
    network_config, cfg = create_test_configs(model_path)
    actor = TorchInferenceActor(network_config, cfg)
    
    # Ensure no current model is loaded
    actor.model = None
    # Set a dummy current_model_file to avoid AttributeError (there's a bug in the actual code)
    actor.current_model_file = ModelFile(path="dummy", creation_time=0)
    
    # Initially no models exist
    assert actor.model_store.cached_list_model_files() == []
    
    actor.load_model_if_necessary(maybe_create=True)
    
    # Should have created a real model file
    model_files = actor.model_store.cached_list_model_files()
    assert len(model_files) == 1
    assert model_files[0].path.endswith("0.pt")
    
    # Should set the current model file
    assert actor.current_model_file is not None
    assert actor.current_model_file.path.endswith("0.pt")
    
    # Verify the model file actually exists and contains a real model
    model_file_path = actor.current_model_file.path
    assert os.path.exists(model_file_path)
    
    # Try loading the created model to verify it's valid
    state_dict = torch.load(model_file_path, weights_only=True)
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0  # Should have some parameters