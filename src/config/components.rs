use anyhow::{Context, Result};
use serde::Deserialize;

use crate::config::NUM_PLAYERS;
use crate::game::move_data;
use crate::game::move_data::{MoveData, MoveProfile};
use crate::game::{MovesArray, MovesBitSet};

#[derive(Deserialize, Debug)]
pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: usize,
    // Path to the file containing the static move profiles.
    pub move_data_file: String,
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

    pub fn move_profiles(&self) -> &MovesArray<MoveProfile> {
        &self.move_data.as_ref().unwrap().profiles
    }

    pub fn cloned_initial_moves_enabled(&self) -> [MovesBitSet; NUM_PLAYERS] {
        self.move_data
            .as_ref()
            .unwrap()
            .initial_moves_enabled
            .clone()
    }

    pub fn load_move_profiles(&mut self) -> Result<()> {
        self.move_data = Some(move_data::load(&self.move_data_file).with_context(|| {
            format!(
                "Failed to load move profiles from file: {}",
                self.move_data_file
            )
        })?);
        Ok(())
    }
}

pub struct MCTSConfig {
    pub num_rollouts: u32,
    pub total_dirichlet_noise_alpha: f32,
    pub root_dirichlet_noise_fraction: f32,
    pub ucb_exploration_factor: f32,
    pub temperature_turn_cutoff: u16,
    pub move_selection_temperature: f32,
}
