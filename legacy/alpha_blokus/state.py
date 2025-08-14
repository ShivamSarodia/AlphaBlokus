import numpy as np
import functools

from alpha_blokus.display import Display
from alpha_blokus.moves_data import moves_data
_initial_moves_enabled_cache = None

class State:
    def __init__(self, cfg):
        self.cfg = cfg

        # Moves for each player that are permissible because they intersect
        # with an exposed corner.
        self.moves_enabled = self._get_initial_moves_enabled().copy()

        # Moves for each player that are ruled out by conflict
        # with an occupied square, conflict with a player's own adjacent
        # square, or having used the piece already.
        self.moves_ruled_out = np.zeros((4, cfg["game"]["num_moves"]), dtype=bool)

        # Compute the score as we go.
        self.accumulated_scores = np.zeros(4, dtype=int)

        # Track occupancies as the board state.
        self.occupancies = np.zeros((4, cfg["game"]["board_size"], cfg["game"]["board_size"]), dtype=bool)

        # Track unused pieces per player.
        self.unused_pieces = np.ones((4, cfg["game"]["num_pieces"]), dtype=bool)
        
        # Player 0 starts.
        self.player = 0

        # Track the number of turns played.
        self.turn = 0

        # Index of the previous move played, useful for rendering the game.
        self.last_move_index = None

    def _get_initial_moves_enabled(self):
        global _initial_moves_enabled_cache
        if _initial_moves_enabled_cache is not None:
            return _initial_moves_enabled_cache
        
        # First-time computation
        board_size = self.cfg["game"]["board_size"]
        print("Computing initial moves enabled...")
        # Precompute corner mask
        start_corners = np.zeros((4, board_size, board_size), dtype=bool)
        start_corners[0, 0, 0] = True
        start_corners[1, 0, board_size - 1] = True
        start_corners[2, board_size - 1, board_size - 1] = True
        start_corners[3, board_size - 1, 0] = True
        mask = np.any(
            moves_data(self.cfg)["new_occupieds"] & start_corners[:, np.newaxis, :, :],
            axis=(2, 3),
        )
        _initial_moves_enabled_cache = mask
        return mask

    def clone(self):
        new_state = State(self.cfg)
        new_state.moves_enabled = self.moves_enabled.copy()
        new_state.moves_ruled_out = self.moves_ruled_out.copy()
        new_state.accumulated_scores = self.accumulated_scores.copy()
        new_state.occupancies = self.occupancies.copy()
        new_state.unused_pieces = self.unused_pieces.copy()
        new_state.player = self.player
        new_state.turn = self.turn
        new_state.last_move_index = self.last_move_index

        return new_state

    def play_move(self, move_index) -> bool:
        """
        Play the given move and update self.player to the new player (skipping any players 
        who are out of moves). Return whether the game is over.

        This method assumes the provided move is valid.
        """
        if self.cfg["gameplay"]["check_move_validity"]:
            if not self.valid_moves_array()[move_index]:
                raise "Playing an invalid move!"
            if not self.unused_pieces[self.player][moves_data(self.cfg)["piece_indices"][move_index]]:
                raise "Playing a piece that has already been used!"

        # Update occupancies.
        self.occupancies[self.player] |= moves_data(self.cfg)["new_occupieds"][move_index]

        # Update unused pieces.
        self.unused_pieces[self.player][moves_data(self.cfg)["piece_indices"][move_index]] = False

        # Rule out some moves.
        self.moves_ruled_out |= moves_data(self.cfg)["moves_ruled_out_for_all"][move_index]
        self.moves_ruled_out[self.player] |= moves_data(self.cfg)["moves_ruled_out_for_player"][move_index]

        # Enable new moves for player based on corners.
        self.moves_enabled[self.player] |= moves_data(self.cfg)["moves_enabled_for_player"][move_index]

        # Compute scores.
        self.accumulated_scores[self.player] += moves_data(self.cfg)["scores"][move_index]

        self.last_move_index = move_index

        # Increment the number of turns played.
        self.turn += 1
        
        # Find the next player who has a valid move.
        for _ in range(4):
            self.player = (self.player + 1) % 4
            if self.valid_moves_array().any():
                return False
            
        return True

    def valid_moves_array(self):
        return ~self.moves_ruled_out[self.player] & self.moves_enabled[self.player]
    
    def result(self):
        r = np.where(self.accumulated_scores == np.max(self.accumulated_scores), 1, 0)
        return r / np.sum(r)
    
    def pretty_print_board(self):
        Display(
            self.occupancies,
            moves_data(self.cfg)["new_occupieds"][self.last_move_index],
        ).show()
