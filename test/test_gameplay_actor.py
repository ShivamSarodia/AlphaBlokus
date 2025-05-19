import pytest
import pytest_asyncio
import ray
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import OmegaConf

# Import mock_moves_data from test_mcts
from test_mcts import mock_moves_data

from alpha_blokus.gameplay_actor import GameplayActor, generate_agent
from alpha_blokus.state import State
from alpha_blokus.data_recorder import DataRecorder
from alpha_blokus.agents import RandomAgent
from alpha_blokus.mcts import MCTSAgent
from alpha_blokus.inference.client import InferenceClient


class MockInferenceClient:
    """Mock InferenceClient for testing"""
    
    def __init__(self, values_response=None, policy_logits_response=None):
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


class MockAgent:
    """Mock agent for testing"""
    
    def __init__(self):
        self.selected_moves = []
        self.reported_moves = []
    
    async def select_move_index(self, state):
        """Return a predetermined move"""
        # Return valid moves in sequence
        move = 10 + (len(self.selected_moves) * 10) % 50
        self.selected_moves.append(move)
        return move
    
    async def report_move(self, state, move_index):
        """Record reported moves"""
        self.reported_moves.append((state.player, move_index))


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
    
    # Mock clone to return a copy of itself
    state.clone.return_value = state
    
    # Mock result to return a specific result
    state.result.return_value = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Set up the cfg attribute
    state.cfg = OmegaConf.create({"game": {"num_moves": 400}})
    
    return state


@pytest.fixture
def gameplay_config():
    """Create gameplay configuration for testing"""
    config = {
        "gameplay": {
            "architecture": {
                "coroutines_per_process": 2
            },
            "agents": {
                "agent1": {
                    "type": "mcts",
                    "name": "agent1",
                    "network": "network1",
                    "ucb_exploration": 1.0,
                    "full_move_probability": 1.0,
                    "full_move_rollouts": 2,
                    "fast_move_rollouts": 1,
                    "reuse_tree": True,
                    "move_selection_temperature": 0.0,
                    "temperature_turn_cutoff": 5,
                    "total_dirichlet_alpha": 0.3,
                    "root_exploration_fraction": 0.25,
                    "forced_playouts_multiplier": 1.0,
                    "log_ucb_report": False,
                    "log_mcts_report_fraction": 0.0
                },
                "agent2": {
                    "type": "random",
                    "name": "agent2"
                }
            }
        },
        "use_profiler": False,
        "log_made_move": False
    }
    return OmegaConf.create(config)


@pytest.mark.asyncio
async def test_generate_agent(mock_moves_data):
    """Test the generate_agent function"""
    # Test MCTS agent creation
    mcts_config = OmegaConf.create({
        "type": "mcts",
        "network": "test_network"
    })
    inference_clients = {"test_network": MagicMock(spec=InferenceClient)}
    data_recorder = MagicMock(spec=DataRecorder)

    # Since this doesn't actually use the MCTS code that needs moves_data, we don't need to patch
    agent = generate_agent(mcts_config, inference_clients, data_recorder, 1, OmegaConf.create({}))
    assert isinstance(agent, MCTSAgent)

    # Test Random agent creation
    random_config = OmegaConf.create({"type": "random"})
    agent = generate_agent(random_config, {}, None, None, OmegaConf.create({}))
    assert isinstance(agent, RandomAgent)

    # Test invalid agent type
    with pytest.raises(BaseException):
        generate_agent(OmegaConf.create({"type": "invalid"}), {}, None, None, OmegaConf.create({}))


@pytest.fixture
def mock_gameplay_actor(gameplay_config):
    """Create a GameplayActor for testing without Ray initialization"""
    # Mock Ray remote to return the original class
    with patch('ray.remote', return_value=lambda cls: cls):
        # Create mock inference clients
        inference_clients = {
            "network1": MockInferenceClient()
        }
        
        # Create the actor
        actor = GameplayActor(0, inference_clients, gameplay_config)
        
        # Replace data_recorder with a mock
        actor.data_recorder = MagicMock(spec=DataRecorder)
        actor.data_recorder.start_game.return_value = 1
        
        return actor


@pytest.mark.asyncio
async def test_play_game(mock_gameplay_actor, mock_moves_data):
    """Test the play_game method of GameplayActor"""
    # Patch the generate_agent function to return mock agents
    with patch('alpha_blokus.gameplay_actor.generate_agent') as mock_generate_agent:
        # Create mock agents
        mock_agents = {
            "agent1": MockAgent(),
            "agent2": MockAgent()
        }
        mock_generate_agent.side_effect = lambda config, *args, **kwargs: mock_agents[config["name"]]

        # Patch State to control game flow
        with patch('alpha_blokus.gameplay_actor.State') as MockState:
            # Setup mock state behavior
            state_instance = MagicMock()
            MockState.return_value = state_instance

            # Make play_move return game_over=True after 4 moves
            state_instance.play_move.side_effect = lambda move: (state_instance.player >= 3)

            # Make player cycle through 0-3
            type(state_instance).player = MockPlayer()

            # Set up result
            state_instance.result.return_value = [0.5, 0.2, 0.2, 0.1]

            # Add needed configuration
            state_instance.cfg = OmegaConf.create({"game": {"num_moves": 400}})

            # Patch moves_data to avoid loading actual data
            with patch('alpha_blokus.mcts.moves_data', return_value=mock_moves_data):
                # Call play_game
                await mock_gameplay_actor.play_game()

                # Verify:
                # 1. State was initialized
                MockState.assert_called_once()

                # 2. Agents were created for each player type
                assert mock_generate_agent.call_count == 2

                # 3. Each agent was asked to select moves
                assert len(mock_agents["agent1"].selected_moves) > 0

                # 4. Other agents were notified of moves
                assert len(mock_agents["agent1"].reported_moves) > 0 or len(mock_agents["agent2"].reported_moves) > 0

                # 5. Data recorder saved the game result
                mock_gameplay_actor.data_recorder.record_game_end.assert_called_once_with(1, [0.5, 0.2, 0.2, 0.1])


class MockPlayer:
    """Mock property for player that cycles through 0-3"""
    def __init__(self):
        self.value = 0
    
    def __get__(self, obj, objtype=None):
        value = self.value
        self.value = (self.value + 1) % 4
        return value


@pytest.mark.asyncio
async def test_continuously_play_games(mock_gameplay_actor, mock_moves_data):
    """Test the continuously_play_games method"""
    # Make play_game raise an exception after first call to break the loop
    mock_gameplay_actor.play_game = AsyncMock(side_effect=[None, Exception("Stop")])

    # Call continuously_play_games - it should loop until exception
    with patch('alpha_blokus.mcts.moves_data', return_value=mock_moves_data):
        with pytest.raises(Exception):
            await mock_gameplay_actor.continuously_play_games()

        # Verify play_game was called
        assert mock_gameplay_actor.play_game.call_count == 2