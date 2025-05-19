import time
import os
import numpy as np
import random
from omegaconf import OmegaConf

from alpha_blokus import player_pov_helpers
from alpha_blokus.moves_data import moves_data
from alpha_blokus.state import State

class DataRecorder:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.directory = self.cfg.gameplay.game_save_directory

        os.makedirs(self.directory, exist_ok=True)

        # This object is a map from a randomly generated game ID to a dictionary of the form:
        # {
        #     "game_ids": [],
        #     "boards": [],
        #     "policies": [],
        #     "players": [],
        #     "valid_moves_array": [],
        #     "values": [],
        #     "unused_pieces": [],
        # }
        self.games = {}
        self.finished_games = set()

    def start_game(self) -> int:
        game_id = random.randint(0, (1<<31) - 1)
        self.games[game_id] = {
            "game_ids": [],
            "boards": [],
            "policies": [],
            "players": [],
            "valid_moves_array": [],
            "values": [],
            "unused_pieces": [],
        }
        return game_id

    def record_rollout_result(self, game_id: int, state: State, policy: np.ndarray):
        """
        After each MCTS rollout completes, record the occupancy state did the rollout on and the final
        visit distribution of the children of that node.
        """
        game = self.games[game_id]
        game["game_ids"].append(game_id)
        game["boards"].append(
            player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)
        )
        game["policies"].append(
            player_pov_helpers.moves_array_to_player_pov(
                policy,
                state.player,
                moves_data(self.cfg),
            )
        )
        game["valid_moves_array"].append(
            player_pov_helpers.moves_array_to_player_pov(
                state.valid_moves_array(),
                state.player,
                moves_data(self.cfg),
            )
        )
        game["unused_pieces"].append(
            player_pov_helpers.unused_pieces_to_player_pov(
                state.unused_pieces,
                state.player
            )
        )
        game["players"].append(state.player)
    
    def record_game_end(self, game_id: int, values: np.ndarray):
        game = self.games[game_id]
        for player in game["players"]:
            game["values"].append(player_pov_helpers.values_to_player_pov(values, player))

        self.finished_games.add(game_id)
        if (
            # A negative value for GAME_FLUSH_THRESHOLD means we never flush.
            self.cfg.gameplay.game_flush_threshold >= 0 and 
            len(self.finished_games) >= self.cfg.gameplay.game_flush_threshold
        ):
            self.flush()

    def flush(self):
        game_ids = []
        boards = []
        policies = []
        values = []
        valid_moves_array = []
        unused_pieces = []
        players = []
        for game_id in self.finished_games:
            game = self.games[game_id]

            # Some small number of games will have no data, because they were composed
            # only of fast rollouts that didn't report any states.
            if len(game["boards"]) > 0:
                game_ids.append(np.array(game["game_ids"]))
                boards.append(np.array(game["boards"]))
                policies.append(np.array(game["policies"]))
                valid_moves_array.append(np.array(game["valid_moves_array"]))
                values.append(np.array(game["values"]))
                unused_pieces.append(np.array(game["unused_pieces"]))
                players.append(np.array(game["players"]))
            del self.games[game_id]
        
        self.finished_games = set()

        if not boards:
            return
        
        game_ids = np.concatenate(game_ids)
        boards = np.concatenate(boards)
        policies = np.concatenate(policies)
        valid_moves_array = np.concatenate(valid_moves_array)
        values = np.concatenate(values)
        unused_pieces = np.concatenate(unused_pieces)
        players = np.concatenate(players)
        
        # Save the files to disk with the number of samples included, so that the
        # training script can tell from just the filename how many samples are in
        # the file.
        np.savez_compressed(
            os.path.join(self.directory, f"{int(time.time() * 1000)}_{len(boards)}.npz"),
            game_ids=game_ids,
            boards=boards,
            policies=policies,
            players=players,
            values=values,
            valid_moves_array=valid_moves_array,
            unused_pieces=unused_pieces,
        )
