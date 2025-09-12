use crate::{config, game::BoardSlice};
use anyhow::{Result, bail};
use config::GameConfig;
use std::collections::HashSet;

use crate::game::move_data::pieces::Coord;
use crate::game::move_data::pieces::Piece;

pub struct Stage1MoveProfile {
    pub index: usize,
    pub occupied_cells: BoardSlice,
    pub center: (usize, usize),
    pub piece_orientation_index: usize,
    pub piece_index: usize,
}

/// Stage 1 is responsible for accumulating information about each move alone, without
/// any context about its relationship with other stages.
pub fn compute_stage_1_move_profiles(
    pieces: &[Piece],
    game_config: &GameConfig,
) -> Result<Vec<Stage1MoveProfile>> {
    let mut piece_orientations_visited = HashSet::new();
    let mut stage_1_move_profiles = Vec::<Stage1MoveProfile>::new();
    let mut move_index = 0;

    for (piece_index, original_piece) in pieces.iter().enumerate() {
        for rotation in 0..4 {
            for flip in [false, true] {
                let mut piece = original_piece.clone();

                piece.rotate(rotation);
                piece.flip(flip);

                // Recenter and sort the piece to ensure that any two
                // piece-orientations that differ only by a translation or
                // reordering of coordinates map to the same coordinate list.
                piece.recenter();

                // Skip any piece orientations we've already seen in the loop
                // We add only the coordinates to the set because the center can vary
                // even for pieces with the same coordinates. For example, the 1x4 piece
                // can have a piece on either of the two center cells, depending on the
                // orientation, but those don't constitute different piece-orientations.
                if piece_orientations_visited.contains(piece.coords()) {
                    continue;
                }
                piece_orientations_visited.insert(piece.coords().clone());

                // At this point, we know this is a new piece-orientation.
                let top_right = piece.top_right();
                for x in 0..(game_config.board_size - top_right.x) {
                    for y in 0..(game_config.board_size - top_right.y) {
                        // We have found a valid move!

                        // Compute what cells this move will occupy.
                        let mut occupied_cells = BoardSlice::new(game_config.board_size);
                        for Coord {
                            x: c_x,
                            y: c_y,
                            board_size: _,
                        } in piece.coords()
                        {
                            occupied_cells.set(((*c_x + x), (*c_y + y)), true);
                        }

                        // Push move details to the relevant vectors.
                        stage_1_move_profiles.push(Stage1MoveProfile {
                            index: move_index,
                            occupied_cells,
                            center: (piece.center().x + x, piece.center().y + y),
                            piece_orientation_index: piece_orientations_visited.len() - 1,
                            piece_index,
                        });

                        move_index += 1;
                    }
                }
            }
        }
    }

    if piece_orientations_visited.len() != game_config.num_piece_orientations {
        bail!(
            "Number of piece-orientations does not match config: {}",
            piece_orientations_visited.len()
        )
    }
    if stage_1_move_profiles.len() != game_config.num_moves {
        bail!(
            "Number of moves ({}) does not match config ({})",
            stage_1_move_profiles.len(),
            game_config.num_moves,
        )
    }
    for (i, move_profile) in stage_1_move_profiles.iter().enumerate() {
        if move_profile.index != i {
            bail!(
                "Expected the indices to match the indices of the stage in the output vector at index {}.",
                i
            )
        }
    }

    Ok(stage_1_move_profiles)
}
