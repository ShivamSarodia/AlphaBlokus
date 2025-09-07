use crate::game::BoardSlice;
use bit_set::BitSet;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
    pub rotated_move_indexes: [usize; 4],
    pub moves_ruled_out_for_self: BitSet,
    pub moves_ruled_out_for_others: BitSet,
    pub moves_enabled_for_self: BitSet,
}
