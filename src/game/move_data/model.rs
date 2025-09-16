use crate::config::NUM_PLAYERS;
use crate::game::BoardSlice;
use crate::game::{MovesArray, MovesBitSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
    pub rotated_move_indexes: [usize; 4],
    pub moves_ruled_out_for_self: MovesBitSet,
    pub moves_ruled_out_for_others: MovesBitSet,
    pub moves_enabled_for_self: MovesBitSet,
}

// I don't love the naming here :/
#[derive(Debug, Serialize, Deserialize)]
pub struct MoveData {
    pub profiles: MovesArray<MoveProfile>,
    pub initial_moves_enabled: [MovesBitSet; NUM_PLAYERS],
}

pub fn move_index_to_player_pov(
    move_index: usize,
    player: usize,
    move_profiles: &MovesArray<MoveProfile>,
) -> usize {
    let move_profile = move_profiles.get(move_index);
    move_profile.rotated_move_indexes[(NUM_PLAYERS - player) % NUM_PLAYERS]
}
