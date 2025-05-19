from zipfile import BadZipFile
import numpy as np
from tqdm import tqdm
from alpha_blokus.event_logger import log_event

def load_games(game_file_paths):
    boards = []
    policies = []
    values = []
    # valid_moves = []
    # unused_pieces = []
    
    for game_file in game_file_paths:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                boards.append(npz["boards"])
                policies.append(npz["policies"])
                values.append(npz["values"])
                # valid_moves.append(npz["valid_moves_array"])
                # unused_pieces.append(npz["unused_pieces"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    if len(boards) == 0:
        return None

    return (
        np.concatenate(boards),
        np.concatenate(policies),
        np.concatenate(values),
        # np.concatenate(valid_moves),
        # np.concatenate(unused_pieces),
    )

def load_games_new(game_file_paths, with_tqdm=False, skip_concat=False):
    game_ids = []
    boards = []
    policies = []
    values = []
    valid_moves = []
    unused_pieces = []
    players = []

    iterator = tqdm(game_file_paths) if with_tqdm else game_file_paths
    
    for game_file in iterator:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                boards.append(npz["boards"])
                policies.append(npz["policies"])
                values.append(npz["values"])
                if "game_ids" in npz:
                    game_ids.append(npz["game_ids"])
                valid_moves.append(npz["valid_moves_array"])
                unused_pieces.append(npz["unused_pieces"])
                if "players" in npz:
                    players.append(npz["players"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    if len(boards) == 0:
        return None
    
    if not skip_concat:
        boards = np.concatenate(boards)
        policies = np.concatenate(policies)
        values = np.concatenate(values)
        valid_moves = np.concatenate(valid_moves)
        unused_pieces = np.concatenate(unused_pieces)

    result = {
        "boards": boards,
        "policies": policies,
        "values": values,
        "valid_moves": valid_moves,
        "unused_pieces": unused_pieces,
    }

    if players:
        result["players"] = np.concatenate(players)

    if game_ids:
        result["game_ids"] = np.concatenate(game_ids)

    return result


def load_game_file(game_file_path):
    with open(game_file_path, "rb") as f:
        try:
            npz = np.load(f)
            result = {
                "boards": npz["boards"],
                "policies": npz["policies"],
                "values": npz["values"],
                "valid_moves": npz["valid_moves_array"],
                "unused_pieces": npz["unused_pieces"],
            }
            if "game_ids" in npz:
                result["game_ids"] = npz["game_ids"]
            if "players" in npz:
                result["players"] = npz["players"]
            return result
        except BadZipFile:
            log_event("bad_game_file", {"path": game_file_path})
            return None