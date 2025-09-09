use crate::game::BoardSlice;
use anyhow::Result;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use crate::move_data::stage_1::Stage1MoveProfile;

pub struct Stage2MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
    pub rotated_move_indexes: [usize; 4],
}

/// Stage 2 is responsible for computing rotation information for each move. That is,
/// for each move, what is the rotated move it corresponds to.
pub fn compute_stage_2_move_profiles(
    stage_1_move_profiles: Vec<Stage1MoveProfile>,
) -> Result<Vec<Stage2MoveProfile>> {
    let stage_2_move_profiles = stage_1_move_profiles
        .par_iter()
        .progress_count(stage_1_move_profiles.len() as u64)
        .map(|stage_1_move| {
            let mut rotated_move_indexes: [usize; 4] = [0, 0, 0, 0];
            for turns in 0..4 {
                let rotated_board = stage_1_move.occupied_cells.rotate(turns);
                for comparison_move in &stage_1_move_profiles {
                    if comparison_move.occupied_cells == rotated_board {
                        rotated_move_indexes[turns as usize] = comparison_move.index;
                        break;
                    }
                }
            }

            Stage2MoveProfile {
                index: stage_1_move.index,
                occupied_cells: stage_1_move.occupied_cells.clone(),
                center: stage_1_move.center,
                piece_orientation_index: stage_1_move.piece_orientation_index,
                piece_index: stage_1_move.piece_index,
                rotated_move_indexes,
            }
        })
        .collect::<Vec<Stage2MoveProfile>>();

    Ok(stage_2_move_profiles)
}
