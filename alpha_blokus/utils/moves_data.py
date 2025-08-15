import numpy as np

_MOVES_DATA_CACHE = None

def _load_moves_data(cfg):
    print("Loading moves data...")
    moves_data_path = cfg["game"]["moves_data_path"]
    compressed_moves = np.load(moves_data_path)
    moves = {key: compressed_moves[key].copy() for key in compressed_moves.files}

    assert moves["new_occupieds"].shape[0] == cfg["game"]["num_moves"]
    assert moves["new_occupieds"].shape[1] == cfg["game"]["board_size"]

    compressed_moves.close()
    
    print("Moves data loaded.")

    return moves

def moves_data(cfg):
    global _MOVES_DATA_CACHE
    if _MOVES_DATA_CACHE is None:
        _MOVES_DATA_CACHE = _load_moves_data(cfg)
    return _MOVES_DATA_CACHE