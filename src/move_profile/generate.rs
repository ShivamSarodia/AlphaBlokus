use anyhow::Result;
use bincode;
use std::fs::File;
use std::io::{BufWriter, Write};
use zstd::stream::write::Encoder;

use crate::config::{GameConfig, PreprocessMovesConfig};
use crate::game::MovesArray;
use crate::move_profile::MoveProfile;
use crate::move_profile::pieces::piece_list;
use crate::move_profile::stage_1::compute_stage_1_move_profiles;
use crate::move_profile::stage_2::compute_stage_2_move_profiles;
use crate::move_profile::stage_3::compute_stage_3_move_profiles;
use crate::move_profile::stage_4::compute_stage_4_move_profiles;

pub fn generate(config: &GameConfig) -> Result<MovesArray<MoveProfile>> {
    println!("Generating piece list...");
    let pieces = piece_list(config.num_pieces, config.board_size)?;
    println!("Generated piece list!");

    println!("Computing stage 1 profiles...");
    let stage_1_move_profiles = compute_stage_1_move_profiles(&pieces, config)?;
    println!("Computed stage 1 profiles!");

    println!("Computing stage 2 profiles...");
    let stage_2_move_profiles = compute_stage_2_move_profiles(stage_1_move_profiles)?;
    println!("Computed stage 2 profiles!");

    println!("Computing stage 3 profiles...");
    let stage_3_move_profiles =
        compute_stage_3_move_profiles(stage_2_move_profiles, config.board_size)?;
    println!("Computed stage 3 profiles!");

    println!("Computing stage 4 profiles...");
    let stage_4_move_profiles = compute_stage_4_move_profiles(stage_3_move_profiles)?;
    println!("Computed stage 4 profiles!");

    let move_profile_vec = stage_4_move_profiles
        .into_iter()
        .map(|stage_4_move_profile| MoveProfile {
            index: stage_4_move_profile.index,
            occupied_cells: stage_4_move_profile.occupied_cells,
            center: stage_4_move_profile.center,
            piece_orientation_index: stage_4_move_profile.piece_orientation_index,
            piece_index: stage_4_move_profile.piece_index,
            rotated_move_indexes: stage_4_move_profile.rotated_move_indexes,
            moves_ruled_out_for_self: stage_4_move_profile.moves_ruled_out_for_self,
            moves_ruled_out_for_others: stage_4_move_profile.moves_ruled_out_for_others,
            moves_enabled_for_self: stage_4_move_profile.moves_enabled_for_self,
        })
        .collect::<Vec<MoveProfile>>();

    Ok(MovesArray::new_from_vec(move_profile_vec, config))
}

pub fn save(move_profiles: MovesArray<MoveProfile>, config: &PreprocessMovesConfig) -> Result<()> {
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(&config.output_file)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    let bincode_config = bincode::config::standard().with_variable_int_encoding();
    bincode::serde::encode_into_std_write(&move_profiles, &mut enc, bincode_config)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    println!("Wrote move profiles to disk at {}", config.output_file);

    println!("Finished!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BoardSlice;
    use std::collections::HashMap;

    #[test]
    fn test_generate() -> Result<()> {
        let game_config = GameConfig {
            board_size: 5,
            num_moves: 958,
            num_pieces: 21,
            num_piece_orientations: 91,
            moves_file_path: "".to_string(),
        };
        let move_profiles = generate(&game_config)?;
        let pieces = piece_list(game_config.num_pieces, game_config.board_size).unwrap();

        let mut center_occupied_count = 0;
        let mut piece_orientation_to_piece = HashMap::<usize, usize>::new();

        for i in 0..game_config.num_moves {
            let move_profile = move_profiles.get(i);

            // Confirm the stored index matches the move's position in the list.
            assert_eq!(move_profile.index, i);

            // Confirm the occupied cells match the piece's size.
            let piece_size = pieces[move_profile.piece_index].coords().len();
            assert_eq!(move_profile.occupied_cells.count(), piece_size);

            // Help confirm the center is usually occupied.
            if move_profile
                .occupied_cells
                .get((move_profile.center.0, move_profile.center.1))
            {
                center_occupied_count += 1;
            }

            // Confirm that a piece orientation index only ever has exactly one associated
            // piece index.
            let expected_piece_index =
                piece_orientation_to_piece.get(&move_profile.piece_orientation_index);
            if expected_piece_index.is_none() {
                piece_orientation_to_piece.insert(
                    move_profile.piece_orientation_index,
                    move_profile.piece_index,
                );
            } else {
                assert_eq!(*expected_piece_index.unwrap(), move_profile.piece_index);
            }

            // Confirm that each rotate move uses the same piece as the original move.
            for turns in 1..4 {
                let rotated_move_profile =
                    move_profiles.get(move_profile.rotated_move_indexes[turns]);
                assert_eq!(
                    rotated_move_profile.piece_index,
                    move_profile.piece_index,
                    "Move {} uses piece {}, which doesn't match its rotated move {} which uses piece {}",
                    move_profile.index,
                    move_profile.piece_index,
                    rotated_move_profile.index,
                    rotated_move_profile.piece_index,
                )
            }

            // Confirm that the rotated move indexes cycle as expected.
            assert_eq!(
                move_profile.rotated_move_indexes[0], move_profile.index,
                "The unrotated move should match the original move. Index: {}",
                move_profile.index,
            );
            assert_eq!(
                move_profiles
                    .get(
                        move_profiles
                            .get(
                                move_profiles
                                    .get(move_profile.rotated_move_indexes[1])
                                    .rotated_move_indexes[1],
                            )
                            .rotated_move_indexes[1]
                    )
                    .rotated_move_indexes[1],
                move_profile.index,
                "Rotating the move four times by 90 degrees should return the original move. Index: {}",
                move_profile.index
            );
            assert_eq!(
                move_profiles
                    .get(move_profile.rotated_move_indexes[2])
                    .rotated_move_indexes[2],
                move_profile.index,
                "Rotating the move twice by 180 degrees should return the original move. Index: {}",
                move_profile.index
            );
            assert_eq!(
                move_profiles
                    .get(
                        move_profiles
                            .get(
                                move_profiles
                                    .get(move_profile.rotated_move_indexes[3])
                                    .rotated_move_indexes[3],
                            )
                            .rotated_move_indexes[3]
                    )
                    .rotated_move_indexes[3],
                move_profile.index,
                "Rotating the move four times by 270 degrees should return the original move. Index: {}",
                move_profile.index
            );

            for move_index in 0..game_config.num_moves {
                let other_move_profile = move_profiles.get(move_index);

                // If a move is ruled out for self but NOT others, it must either share a piece index with this move
                // or one must occupy the edges of another.
                if move_profile.moves_ruled_out_for_self.contains(move_index)
                    && !move_profile.moves_ruled_out_for_others.contains(move_index)
                {
                    let same_piece_index =
                        other_move_profile.piece_index == move_profile.piece_index;
                    let edges_overlap = edge_overlap(
                        &move_profile.occupied_cells,
                        &other_move_profile.occupied_cells,
                    );

                    assert!(same_piece_index || edges_overlap);
                }

                // If a move is ruled out for others, it must overlap occupied cells.
                if move_profile.moves_ruled_out_for_others.contains(move_index) {
                    assert!(
                        move_profile
                            .occupied_cells
                            .overlaps(&other_move_profile.occupied_cells)
                    );
                }

                // If a move is enabled, it must overlap a corner cell.
                if move_profile.moves_enabled_for_self.contains(move_index) {
                    assert!(corner_overlap(
                        &move_profile.occupied_cells,
                        &other_move_profile.occupied_cells
                    ));
                }
            }
        }

        // Confirm the center was almost always occupied.
        // For a few pieces, it isn't: e.g. the L piece.
        assert!((center_occupied_count as f64) / (game_config.num_moves as f64) > 0.95);

        Ok(())
    }

    fn edge_overlap(board_slice_1: &BoardSlice, board_slice_2: &BoardSlice) -> bool {
        for x in 0..board_slice_1.size() {
            for y in 0..board_slice_1.size() {
                if !board_slice_1.get((x, y)) {
                    continue;
                }

                if board_slice_2.get_padded((x as i32 - 1, y as i32)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32, y as i32 - 1)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32 + 1, y as i32)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32, y as i32 + 1)) {
                    return true;
                }
            }
        }
        false
    }

    fn corner_overlap(board_slice_1: &BoardSlice, board_slice_2: &BoardSlice) -> bool {
        for x in 0..board_slice_1.size() {
            for y in 0..board_slice_1.size() {
                if !board_slice_1.get((x, y)) {
                    continue;
                }

                if board_slice_2.get_padded((x as i32 - 1, y as i32 - 1)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32 - 1, y as i32 + 1)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32 + 1, y as i32 - 1)) {
                    return true;
                }
                if board_slice_2.get_padded((x as i32 + 1, y as i32 + 1)) {
                    return true;
                }
            }
        }
        false
    }
}
