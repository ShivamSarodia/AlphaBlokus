use crate::game::BoardSlice;
use anyhow::Result;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use crate::move_profile::stage_2::Stage2MoveProfile;

pub struct Stage3MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
    pub rotated_move_indexes: [usize; 4],
    pub corner_cells: BoardSlice,
    pub edge_cells: BoardSlice,
}

/// Stage 3 is responsible for computing for each move a BoardSlice of the associated
/// corner and edge cells.
pub fn compute_stage_3_move_profiles(
    stage_2_move_profiles: Vec<Stage2MoveProfile>,
    size: usize,
) -> Result<Vec<Stage3MoveProfile>> {
    let stage_3_move_profiles = stage_2_move_profiles
        .par_iter()
        .progress_count(stage_2_move_profiles.len() as u64)
        .map(|stage_2_move| {
            let mut corner_cells = BoardSlice::new(size);
            let mut edge_cells = BoardSlice::new(size);

            for x in 0..size {
                for y in 0..size {
                    // An edge cell is horizontally or vertically adjacent to an occupied cell,
                    // but not itself occupied by this piece.
                    let is_edge_cell = ((x > 0 && stage_2_move.occupied_cells.get((x - 1, y)))
                        || (x < size - 1 && stage_2_move.occupied_cells.get((x + 1, y)))
                        || (y > 0 && stage_2_move.occupied_cells.get((x, y - 1)))
                        || (y < size - 1 && stage_2_move.occupied_cells.get((x, y + 1))))
                        && !stage_2_move.occupied_cells.get((x, y));

                    // A corner cell is diagonally adjacent to an occupied cell, but is not an
                    // edge cell or itself occupied by this piece.
                    let is_corner_cell =
                        ((x > 0 && y > 0 && stage_2_move.occupied_cells.get((x - 1, y - 1)))
                            || (x > 0
                                && y < size - 1
                                && stage_2_move.occupied_cells.get((x - 1, y + 1)))
                            || (x < size - 1
                                && y > 0
                                && stage_2_move.occupied_cells.get((x + 1, y - 1)))
                            || (x < size - 1
                                && y < size - 1
                                && stage_2_move.occupied_cells.get((x + 1, y + 1))))
                            && !is_edge_cell
                            && !stage_2_move.occupied_cells.get((x, y));

                    if is_corner_cell {
                        corner_cells.set((x, y), true);
                    }
                    if is_edge_cell {
                        edge_cells.set((x, y), true);
                    }
                }
            }

            Stage3MoveProfile {
                index: stage_2_move.index,
                occupied_cells: stage_2_move.occupied_cells.clone(),
                center: stage_2_move.center,
                piece_orientation_index: stage_2_move.piece_orientation_index,
                piece_index: stage_2_move.piece_index,
                rotated_move_indexes: stage_2_move.rotated_move_indexes,
                corner_cells,
                edge_cells,
            }
        })
        .collect::<Vec<Stage3MoveProfile>>();

    Ok(stage_3_move_profiles)
}
