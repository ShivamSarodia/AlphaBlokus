use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

use crate::config::NUM_PLAYERS;
use crate::game::move_data;
use crate::game::move_data::{MoveData, MoveProfile};
use crate::game::{MovesArray, MovesBitSet};

#[derive(Deserialize, Debug)]
pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: usize,
    // Path to the file containing the static move profiles.
    pub move_data_file: PathBuf,
    // Number of valid moves.
    pub num_moves: usize,
    // Number of pieces that can be played. (For standard Blokus, this is 21)
    pub num_pieces: usize,
    // Number of (piece, orientation) tuples that produce a unique shape. (For standard Blokus, this is 91)
    pub num_piece_orientations: usize,

    // Move data. This is loaded from the provided file when load_move_data is called.
    pub move_data: Option<MoveData>,
}

impl GameConfig {
    pub fn board_area(&self) -> usize {
        self.board_size * self.board_size
    }

    pub fn move_profiles(&self) -> Result<&MovesArray<MoveProfile>> {
        self.move_data
            .as_ref()
            .map(|data| &data.profiles)
            .context("Move data is not loaded")
    }

    pub fn cloned_initial_moves_enabled(&self) -> Result<[MovesBitSet; NUM_PLAYERS]> {
        self.move_data
            .as_ref()
            .map(|data| data.initial_moves_enabled.clone())
            .context("Move data is not loaded")
    }

    pub fn load_move_profiles(&mut self) -> Result<()> {
        self.move_data = Some(move_data::load(self.move_data_file.as_path()).with_context(
            || {
                format!(
                    "Failed to load move profiles from file: {}",
                    self.move_data_file.display()
                )
            },
        )?);
        Ok(())
    }
}
