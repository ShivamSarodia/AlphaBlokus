use crate::game::BoardSlice;
use crate::game::MovesBitSet;
use anyhow::Result;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use crate::move_data::stage_3::Stage3MoveProfile;

pub struct Stage4MoveProfile {
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

/// Stage 4 is responsible for computing which moves are ruled out or enabled for each other
/// player.
pub fn compute_stage_4_move_profiles(
    stage_3_move_profiles: Vec<Stage3MoveProfile>,
) -> Result<Vec<Stage4MoveProfile>> {
    let num_moves = stage_3_move_profiles.len();
    let stage_4_move_profiles = stage_3_move_profiles
        .par_iter()
        .progress_count(stage_3_move_profiles.len() as u64)
        .map(|stage_3_move| {
            let mut moves_ruled_out_for_self = MovesBitSet::new(num_moves);
            let mut moves_ruled_out_for_others = MovesBitSet::new(num_moves);
            let mut moves_enabled_for_self = MovesBitSet::new(num_moves);

            for other_stage_3_move in &stage_3_move_profiles {
                let mut ruled_out_for_self = false;
                let mut ruled_out_for_others = false;
                let mut enabled_for_self = false;

                // If the other move occupies any of this move's edge cells, that other move is
                // now ruled out for the player.
                if other_stage_3_move
                    .occupied_cells
                    .overlaps(&stage_3_move.edge_cells)
                {
                    ruled_out_for_self = true;
                }

                // If the other move uses the same piece as this move, that other move is ruled
                // out for the player.
                if other_stage_3_move.piece_index == stage_3_move.piece_index {
                    ruled_out_for_self = true;
                }

                // If the other move occupies any of this move's occupied cells, that other move
                // is now ruled out for both the player and all other players.
                if other_stage_3_move
                    .occupied_cells
                    .overlaps(&stage_3_move.occupied_cells)
                {
                    ruled_out_for_self = true;
                    ruled_out_for_others = true;
                }

                // If the other move occupies any corner of this move, it's now enabled for the player.
                // For convenience, we exclude any moves that are already ruled out, but that isn't
                // strictly necessary because the game only permits moves that are both enabled AND
                // not ruled out.
                if !ruled_out_for_self
                    && other_stage_3_move
                        .occupied_cells
                        .overlaps(&stage_3_move.corner_cells)
                {
                    enabled_for_self = true;
                }

                if ruled_out_for_self {
                    moves_ruled_out_for_self.insert(other_stage_3_move.index);
                }
                if ruled_out_for_others {
                    moves_ruled_out_for_others.insert(other_stage_3_move.index);
                }
                if enabled_for_self {
                    moves_enabled_for_self.insert(other_stage_3_move.index);
                }
            }

            Stage4MoveProfile {
                index: stage_3_move.index,
                occupied_cells: stage_3_move.occupied_cells.clone(),
                center: stage_3_move.center,
                piece_orientation_index: stage_3_move.piece_orientation_index,
                piece_index: stage_3_move.piece_index,
                rotated_move_indexes: stage_3_move.rotated_move_indexes,
                moves_ruled_out_for_self,
                moves_ruled_out_for_others,
                moves_enabled_for_self,
            }
        })
        .collect::<Vec<Stage4MoveProfile>>();

    Ok(stage_4_move_profiles)
}
