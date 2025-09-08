use crate::config::{GameConfig, NUM_PLAYERS};
use crate::game::Board;
use crate::game::MovesBitSet;

pub struct State<'c> {
    board: Board,
    player: usize,
    turn: u8,
    moves_enabled: [MovesBitSet; NUM_PLAYERS],
    moves_ruled_out: [MovesBitSet; NUM_PLAYERS],
    game_config: &'c GameConfig,
}

impl<'c> State<'c> {
    pub fn new(game_config: &'c GameConfig) -> Self {
        State {
            board: Board::new(game_config),
            player: 0,
            turn: 0,
            moves_enabled: std::array::from_fn(|_| MovesBitSet::new(game_config.num_moves)),
            moves_ruled_out: std::array::from_fn(|_| MovesBitSet::new(game_config.num_moves)),
            game_config,
        }
    }

    pub fn apply_move(&mut self, move_index: usize) {
        let move_profile = self.game_config.move_profiles().get(move_index);

        // Update the board occupancies with the newly occupied cells of the
        // selected move.
        self.board.add(self.player, &move_profile.occupied_cells);

        // Update moves enabled for player.
        self.moves_enabled[self.player].add(&move_profile.moves_enabled_for_self);

        // Update moves ruled out for the player.
        self.moves_ruled_out[self.player].add(&move_profile.moves_ruled_out_for_self);

        // Update moves ruled out for other players.
        for other_player in 0..NUM_PLAYERS {
            if other_player != self.player {
                self.moves_ruled_out[other_player].add(&move_profile.moves_ruled_out_for_others);
            }
        }

        // Update turn
        self.turn += 1;

        // TODO: Update player -- e.g. if a player can't make a move.
        // TODO: Write test!
    }
}
