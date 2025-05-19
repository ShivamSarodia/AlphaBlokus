import pytest
import pytest_asyncio
import numpy as np
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import OmegaConf

from alpha_blokus.mcts import MCTSAgent, MCTSValuesNode
from alpha_blokus.state import State
from alpha_blokus.data_recorder import DataRecorder
from alpha_blokus.inference.client import InferenceClient


class MockInferenceClient:
    """Mock InferenceClient for testing MCTS"""
    
    def __init__(self, values_response=None, policy_logits_response=None):
        """
        Initialize with predetermined responses
        
        Args:
            values_response: numpy array to return as values
            policy_logits_response: numpy array to return as policy logits
        """
        self.values_response = values_response if values_response is not None else np.array([0.5, 0.5, 0.5, 0.5])
        self.policy_logits_response = policy_logits_response if policy_logits_response is not None else np.ones(10)
        self.evaluate_calls = []
        
    def init_in_process(self, loop):
        pass
        
    async def evaluate(self, board, move_indices, turn):
        """Mock implementation that returns predetermined values"""
        self.evaluate_calls.append((board, move_indices, turn))
        # Return policy logits for the specific moves requested
        filtered_logits = self.policy_logits_response
        if len(filtered_logits) != len(move_indices):
            # If sizes don't match, create a new array of correct size
            filtered_logits = np.ones(len(move_indices))
        return self.values_response, filtered_logits


@pytest.fixture
def mock_moves_data():
    """Create mock moves data for testing"""
    # Create a simple data structure that mimics the shape and structure of moves_data
    # This dictionary maps player-relative move indices to universal move indices
    # For simplicity, we'll use a 1-to-1 mapping for the moves we're testing
    moves_data = {
        'player_pov_to_universal': {
            0: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            1: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            2: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            3: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
        },
        'universal_to_player_pov': {
            0: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            1: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            2: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
            3: {10: 10, 20: 20, 30: 30, 40: 40, 50: 50},
        }
    }
    return moves_data

@pytest.fixture
def mock_state():
    """Create a mock State object for testing"""
    state = MagicMock(spec=State)
    state.turn = 0
    state.player = 0
    state.occupancies = np.zeros((20, 20, 4), dtype=np.int8)

    # Set up valid_moves_array to return a specific pattern
    valid_moves = np.zeros(400, dtype=bool)
    valid_moves[[10, 20, 30, 40, 50]] = True
    state.valid_moves_array.return_value = valid_moves

    # Mock play_move to return False (game not over)
    state.play_move.return_value = False

    # Mock clone to return a copy of itself
    state.clone.return_value = state

    # Mock result to return a specific result
    state.result.return_value = np.array([1.0, 0.0, 0.0, 0.0])

    # Set up the cfg attribute
    state.cfg = OmegaConf.create({"game": {"num_moves": 400}})

    return state


@pytest.fixture
def mcts_config():
    """Create MCTS configuration for testing"""
    config = {
        "ucb_exploration": 1.0,
        "full_move_probability": 1.0,
        "full_move_rollouts": 10,
        "fast_move_rollouts": 5,
        "reuse_tree": True,
        "move_selection_temperature": 0.0,
        "temperature_turn_cutoff": 5,
        "total_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "forced_playouts_multiplier": 1.0,
        "log_ucb_report": False,
        "log_mcts_report_fraction": 0.0
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_data_recorder():
    """Create a mock DataRecorder"""
    recorder = MagicMock(spec=DataRecorder)
    recorder.record_rollout_result = MagicMock()
    return recorder


@pytest.mark.asyncio
async def test_mcts_agent_select_move_index(mock_state, mcts_config, mock_data_recorder, mock_moves_data):
    """Test MCTSAgent.select_move_index method"""
    # Create a mock InferenceClient with predetermined responses
    inference_client = MockInferenceClient(
        values_response=np.array([0.8, 0.2, 0.2, 0.2]),
        policy_logits_response=np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )

    # Create the MCTS agent
    agent = MCTSAgent(
        mcts_config=mcts_config,
        inference_client=inference_client,
        data_recorder=mock_data_recorder,
        recorder_game_id=1,
        config=OmegaConf.create({})
    )

    # Patch the moves_data function to return our mock data
    with patch('alpha_blokus.mcts.moves_data', return_value=mock_moves_data):
        # Test select_move_index with a fresh tree
        move_index = await agent.select_move_index(mock_state)

        # Verify inference client was called
        assert len(inference_client.evaluate_calls) > 0

        # Check that we got a valid move index
        assert move_index in [10, 20, 30, 40, 50]

        # Test with an existing tree
        move_index2 = await agent.select_move_index(mock_state)
        assert move_index2 in [10, 20, 30, 40, 50]


@pytest.mark.asyncio
async def test_mcts_agent_report_move(mock_state, mcts_config, mock_moves_data):
    """Test MCTSAgent.report_move method"""
    inference_client = MockInferenceClient()

    agent = MCTSAgent(
        mcts_config=mcts_config,
        inference_client=inference_client,
        data_recorder=None,
        recorder_game_id=None,
        config=OmegaConf.create({})
    )

    # Set up a tree root
    agent.next_move_tree_root = MagicMock()

    # Call report_move (no need to patch moves_data here since we're not using the neural network)
    await agent.report_move(mock_state, 10)

    # Verify the tree was cleared
    assert agent.next_move_tree_root is None


@pytest.mark.asyncio
async def test_mcts_values_node_select_child_by_ucb(mock_state, mcts_config, mock_moves_data):
    """Test MCTSValuesNode.select_child_by_ucb method"""
    node = MCTSValuesNode(mcts_config)

    # Set up the node with test data
    node.expanded_at_turn = 0
    node.num_valid_moves = 5
    node.values = np.array([0.8, 0.2, 0.2, 0.2])
    node.array_index_to_move_index = np.array([10, 20, 30, 40, 50])
    node.move_index_to_array_index = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
    node.children_value_sums = np.zeros((4, 5), dtype=float)
    node.children_visit_counts = np.array([5, 3, 0, 0, 0])
    node.children_priors = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    # Update some values to make the UCB scores more interesting
    node.children_value_sums[0, 0] = 4.0  # High value for player 0, move 10
    node.children_value_sums[0, 1] = 1.0  # Low value for player 0, move 20

    # Call select_child_by_ucb
    selected_move = node.select_child_by_ucb(mock_state)

    # Verify a valid move was selected
    assert selected_move in [10, 20, 30, 40, 50]

    # When it's an unexplored node, it should prefer to explore
    node.children_visit_counts = np.array([10, 0, 0, 0, 0])
    selected_move = node.select_child_by_ucb(mock_state)
    assert selected_move in [20, 30, 40, 50]  # Should not select 10


@pytest.mark.asyncio
async def test_mcts_values_node_get_value_and_expand_children(mock_state, mcts_config, mock_moves_data):
    """Test MCTSValuesNode.get_value_and_expand_children method"""
    node = MCTSValuesNode(mcts_config)

    # Create a mock inference client
    inference_client = MockInferenceClient(
        values_response=np.array([0.8, 0.2, 0.2, 0.2]),
        policy_logits_response=np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )

    # Call get_value_and_expand_children with moves_data patched
    with patch('alpha_blokus.mcts.moves_data', return_value=mock_moves_data):
        values = await node.get_value_and_expand_children(mock_state, inference_client, 0)

    # Verify the node was initialized properly
    assert node.expanded_at_turn == 0
    assert node.num_valid_moves == 5
    assert len(node.move_index_to_array_index) == 5
    assert len(node.array_index_to_move_index) == 5
    assert node.children_value_sums.shape == (4, 5)
    assert node.children_visit_counts.shape == (5,)

    # Test caching - calling with same turn should return cached values
    with patch('alpha_blokus.mcts.moves_data', return_value=mock_moves_data):
        old_eval_calls = len(inference_client.evaluate_calls)
        cached_values = await node.get_value_and_expand_children(mock_state, inference_client, 0)
        assert len(inference_client.evaluate_calls) == old_eval_calls
        assert np.array_equal(values, cached_values)

        # Test with a new turn - should reset visit counts
        node.children_visit_counts = np.array([5, 3, 1, 0, 0])
        await node.get_value_and_expand_children(mock_state, inference_client, 1)
        assert np.array_equal(node.children_visit_counts, np.zeros(5))


@pytest.mark.asyncio
async def test_mcts_values_node_add_noise(mcts_config, mock_moves_data):
    """Test MCTSValuesNode.add_noise method"""
    node = MCTSValuesNode(mcts_config)

    # Setup node with test data
    node.num_valid_moves = 5
    node.children_priors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Call add_noise with fixed random seed for reproducibility
    with patch('numpy.random.dirichlet', return_value=np.array([0.1, 0.2, 0.3, 0.2, 0.2])):
        node.add_noise()

    # Verify noise was added correctly
    assert node.noise_added_at_this_node == True

    # Check that priors were modified
    # new_priors = (1 - 0.25) * original_priors + 0.25 * noise
    expected = (1 - 0.25) * 0.2 + 0.25 * np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    assert np.allclose(node.children_priors, expected)


@pytest.mark.asyncio
async def test_mcts_values_node_get_policy(mock_state, mcts_config, mock_moves_data):
    """Test MCTSValuesNode.get_policy method"""
    node = MCTSValuesNode(mcts_config)

    # Setup node with test data
    node.num_valid_moves = 5
    node.array_index_to_move_index = np.array([10, 20, 30, 40, 50])
    node.move_index_to_array_index = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
    node.children_visit_counts = np.array([10, 5, 3, 1, 2])
    node.values = np.array([0.5, 0.5, 0.5, 0.5])
    node.children_priors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    node.children_value_sums = np.zeros((4, 5), dtype=float)

    # Call get_policy
    policy = node.get_policy(mock_state)

    # Verify policy has correct shape and is normalized
    assert policy.shape == (400,)
    assert np.isclose(np.sum(policy), 1.0)

    # Check that policy has non-zero values only at valid move indices
    policy_for_visited_moves = []
    for i in range(400):
        if i in [10, 20, 30, 40, 50]:
            policy_for_visited_moves.append(policy[i])
        else:
            assert policy[i] == 0

    # Check that the policy for visited moves is correct. One of the moves has been
    # reduced to 0 because of our forced playouts removal.
    assert np.allclose(policy_for_visited_moves, np.array([0.5, 0.25, 0.15, 0.0, 0.1]))
