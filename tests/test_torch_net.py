import pytest
import torch
from omegaconf import DictConfig

from alpha_blokus.torch_net import PolicyFlatten
from alpha_blokus.utils.moves_data import moves_data


@pytest.fixture
def test_config():
    """Create test configuration using the existing static moves data file."""
    return DictConfig({
        "game": {
            "moves_data_path": "static/moves_10.npz",
            "board_size": 10,
            "num_moves": 6233,
            "num_piece_orientations": 91  # Max piece orientation index is 90, so we need 91 total
        }
    })
    # Uncomment this to test on the larger board data -- but it takes longer to run.
    #
    # return DictConfig({
    #     "game": {
    #         "moves_data_path": "static/moves_20.npz",
    #         "board_size": 20,
    #         "num_moves": 30433,
    #         "num_piece_orientations": 91
    #     }
    # })


@pytest.fixture
def policy_flatten(test_config):
    """Create PolicyFlatten instance for testing."""
    return PolicyFlatten(test_config)


@pytest.fixture
def moves_data_dict(test_config):
    """Load moves data for testing."""
    return moves_data(test_config)


def test_policy_flatten_mapping_correctness(policy_flatten, test_config, moves_data_dict):
    """Test that PolicyFlatten maps coordinates correctly to move indices."""

    batch_size = 2
    num_piece_orientations = test_config.game.num_piece_orientations
    board_size = test_config.game.board_size
    
    # Create random input tensor.
    input_tensor = torch.randint(
        0,
        1000000,
        (batch_size, num_piece_orientations, board_size, board_size),
        dtype=torch.int32,
    )
    
    # Apply PolicyFlatten.
    output_tensor = policy_flatten(input_tensor)
    
    # Get the indexing arrays from moves data.
    piece_orientation_indices = moves_data_dict["piece_orientation_indices"]
    center_placement_x = moves_data_dict["center_placement_x"]
    center_placement_y = moves_data_dict["center_placement_y"]
    
    # Verify the mapping for each move.
    for move_idx in range(test_config.game.num_moves):
        expected_po = piece_orientation_indices[move_idx]
        expected_x = center_placement_x[move_idx]
        expected_y = center_placement_y[move_idx]

        # Confirm that the move index, when played, occupies (expected_x, expected_y) 
        # or one of the adjacent squares.
        expected_occupieds = moves_data_dict["new_occupieds"][move_idx]
        assert (
            expected_occupieds[expected_x, expected_y] or
            (expected_x < board_size - 1 and expected_occupieds[expected_x + 1, expected_y]) or
            (expected_x > 0 and expected_occupieds[expected_x - 1, expected_y]) or
            (expected_y < board_size - 1 and expected_occupieds[expected_x, expected_y + 1]) or
            (expected_y > 0 and expected_occupieds[expected_x, expected_y - 1])
        )
        
        for batch_idx in range(batch_size):
            # What we expect to find in the output
            expected_value = input_tensor[batch_idx, expected_po, expected_x, expected_y]
            
            # What we actually find in the output
            actual_value = output_tensor[batch_idx, move_idx].item()
            
            assert actual_value == expected_value, (
                f"Move {move_idx} (batch {batch_idx}): expected mapping from "
                f"[{batch_idx}, {expected_po}, {expected_x}, {expected_y}] = {expected_value}, "
                f"but got {actual_value}"
            )


def test_policy_flatten_with_gradient_flow(policy_flatten, test_config):
    """Test that PolicyFlatten preserves gradient flow."""
    batch_size = 2
    num_piece_orientations = test_config.game.num_piece_orientations
    board_size = test_config.game.board_size
    
    # Create input tensor with gradients enabled
    input_tensor = torch.randn(
        batch_size, num_piece_orientations, board_size, board_size, 
        requires_grad=True
    )
    
    # Apply PolicyFlatten
    output_tensor = policy_flatten(input_tensor)
    
    # Create a simple loss and backpropagate
    loss = output_tensor.sum()
    loss.backward()
    
    # Verify that gradients flowed back to the input
    assert input_tensor.grad is not None, "Gradients should flow back through PolicyFlatten"
    assert input_tensor.grad.shape == input_tensor.shape, "Gradient shape should match input shape"
    
    # Verify that the number of non-zero gradients in the input tensor is equal to the number of moves.
    assert torch.count_nonzero(input_tensor.grad) == batch_size * test_config.game.num_moves