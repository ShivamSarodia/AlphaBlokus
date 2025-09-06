use crate::game::BoardSlice;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
    pub rotated_move_indexes: [usize; 4],
    pub moves_ruled_out_for_player: Vec<usize>,
    pub moves_ruled_out_for_all: Vec<usize>,
    pub moves_enabled_for_player: Vec<usize>,
}
