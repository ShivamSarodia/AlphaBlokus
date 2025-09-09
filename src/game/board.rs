use crate::config::GameConfig;
use crate::config::NUM_PLAYERS;
use crate::game::BoardSlice;
use crate::game::display::{BoardDisplay, BoardDisplayLayer};
use std::fmt;

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

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = BoardDisplay::new(vec![
            BoardDisplayLayer {
                color: "blue",
                board_slice: &self.slices[0],
            },
            BoardDisplayLayer {
                color: "yellow",
                board_slice: &self.slices[1],
            },
            BoardDisplayLayer {
                color: "red",
                board_slice: &self.slices[2],
            },
            BoardDisplayLayer {
                color: "green",
                board_slice: &self.slices[3],
            },
        ]);
        f.write_str(&display.render())?;
        Ok(())
    }
}
