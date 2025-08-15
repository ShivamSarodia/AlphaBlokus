"""
Centralized test configuration for AlphaBlokus tests.

This module provides shared test configurations to avoid conflicts
between different tests that use different board sizes or other parameters.
"""

from omegaconf import DictConfig

# Global setting: Choose which board size to use for all tests
# Options: "small" (10x10) or "large" (20x20)
BOARD_SIZE_CONFIG = "small"

# Board size configurations
_BOARD_CONFIGS = {
    "small": {
        "board_size": 10,
        "moves_data_path": "static/moves_10.npz",
        "num_moves": 6233,
        "num_piece_orientations": 91,
    },
    "large": {
        "board_size": 20,
        "moves_data_path": "static/moves_20.npz", 
        "num_moves": 30433,
        "num_piece_orientations": 91,
    }
}

def get_test_game_config():
    """Get the game configuration for tests based on the global BOARD_SIZE_CONFIG."""
    config_name = BOARD_SIZE_CONFIG
    if config_name not in _BOARD_CONFIGS:
        raise ValueError(f"Unknown board config: {config_name}. Available: {list(_BOARD_CONFIGS.keys())}")
    
    return _BOARD_CONFIGS[config_name]

def create_test_config():
    """Create a test configuration DictConfig using the global board size setting."""
    game_config = get_test_game_config()
    return DictConfig({
        "game": game_config
    })

def create_inference_test_configs(model_path):
    """Create test configuration objects for TorchInferenceActor tests."""
    game_config = get_test_game_config()
    
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
        "game": game_config
    })
    
    return network_config, cfg