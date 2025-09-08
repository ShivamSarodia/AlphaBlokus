use anyhow::Result;
use serde::Deserialize;

use crate::game::MovesArray;
use crate::move_profile::MoveProfile;

#[derive(Deserialize, Debug)]
pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: usize,
    // Path to the file containing the static move profiles.
    pub move_profiles_file: String,
    // Number of valid moves.
    pub num_moves: usize,
    // Number of pieces that can be played. (For standard Blokus, this is 21)
    pub num_pieces: usize,
    // Number of (piece, orientation) tuples that produce a unique shape. (For standard Blokus, this is 91)
    pub num_piece_orientations: usize,

    // Move profiles themselves. These are loaded from the provided file when load_move_profiles is
    // called.
    pub move_profiles: Option<MovesArray<MoveProfile>>,
}

impl GameConfig {
    pub fn board_area(&self) -> usize {
        self.board_size * self.board_size
    }

    pub fn move_profiles(&self) -> &MovesArray<MoveProfile> {
        self.move_profiles.as_ref().unwrap()
    }

    pub fn load_move_profiles(&mut self) -> Result<()> {
        panic!("Not implemented");
    }
}
