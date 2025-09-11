use crate::config::GameConfig;
use crate::config::NUM_PLAYERS;
use crate::game::BoardSlice;
use crate::game::display::BoardDisplayShape;
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

    pub fn slice(&self, player: usize) -> &BoardSlice {
        &self.slices[player]
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = BoardDisplay::new(vec![
            BoardDisplayLayer {
                color: BoardDisplay::player_to_color(0),
                shape: BoardDisplayShape::Primary,
                board_slice: &self.slices[0],
            },
            BoardDisplayLayer {
                color: BoardDisplay::player_to_color(1),
                shape: BoardDisplayShape::Primary,
                board_slice: &self.slices[1],
            },
            BoardDisplayLayer {
                color: BoardDisplay::player_to_color(2),
                shape: BoardDisplayShape::Primary,
                board_slice: &self.slices[2],
            },
            BoardDisplayLayer {
                color: BoardDisplay::player_to_color(3),
                shape: BoardDisplayShape::Primary,
                board_slice: &self.slices[3],
            },
        ]);
        f.write_str(&display.render())?;
        Ok(())
    }
}
