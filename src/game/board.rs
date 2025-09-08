use crate::config::GameConfig;
use crate::config::NUM_PLAYERS;
use crate::game::BoardSlice;

/// Structure representing a board state, with each player
/// on a different slice of the provided values.
pub struct Board {
    slices: [BoardSlice; NUM_PLAYERS],
    #[allow(dead_code)]
    size: usize,
}

impl Board {
    pub fn new(game_config: &GameConfig) -> Self {
        Board {
            slices: std::array::from_fn(|_| BoardSlice::new(game_config.board_size)),
            size: game_config.board_size,
        }
    }

    pub fn add(&mut self, player: usize, slice: &BoardSlice) {
        self.slices[player].add(slice);
    }
}
