import time
from omegaconf import DictConfig
from io import StringIO
import sys
from contextlib import contextmanager

from alpha_blokus.utils.moves_data import moves_data


@contextmanager
def capture_stdout():
    """Context manager to capture stdout for testing print statements."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout


def test_moves_data_cache_usage():
    """Test that _MOVES_DATA_CACHE prevents multiple file reads."""
    
    # Reset the global cache before testing
    import alpha_blokus.utils.moves_data as moves_data_module
    moves_data_module._MOVES_DATA_CACHE = None
    
    # Use existing static file
    cfg = DictConfig({
        "game": {
            "moves_data_path": "static/moves_10.npz",
            "board_size": 10,
            "num_moves": 6233
        }
    })
    
    # First call should load data from file
    with capture_stdout() as output1:
        start_time1 = time.time()
        result1 = moves_data(cfg)
        end_time1 = time.time()
    first_call_duration = end_time1 - start_time1
    
    # Second call should use cached data
    with capture_stdout() as output2:
        start_time2 = time.time()
        result2 = moves_data(cfg)
        end_time2 = time.time()
    second_call_duration = end_time2 - start_time2
    
    # Verify both calls return the same cached object
    assert result1 is result2, "Second call should return the exact same cached object"
    
    # Verify loading print statement appears only in first call
    output1_str = output1.getvalue()
    output2_str = output2.getvalue()
    
    assert "Loading moves data..." in output1_str, "First call should load data from file"
    assert "Loading moves data..." not in output2_str, "Second call should not load data from file"
    
    # Verify first call takes significantly longer than second call
    assert first_call_duration > 50 * second_call_duration
    
    # Reset cache after test
    moves_data_module._MOVES_DATA_CACHE = None
