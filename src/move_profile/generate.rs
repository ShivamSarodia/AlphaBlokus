use anyhow::Result;
use bincode;
use std::fs::File;

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
            moves_ruled_out_for_player: stage_4_move_profile.moves_ruled_out_for_player,
            moves_ruled_out_for_all: stage_4_move_profile.moves_ruled_out_for_all,
            moves_enabled_for_player: stage_4_move_profile.moves_enabled_for_player,
        })
        .collect::<Vec<MoveProfile>>();

    Ok(MovesArray::new_from_vec(move_profile_vec, config))
}

pub fn save(move_profiles: MovesArray<MoveProfile>, config: &PreprocessMovesConfig) -> Result<()> {
    println!("Saving move profiles...");
    let mut file = File::create(&config.output_file)?;
    let bincode_config = bincode::config::standard().with_variable_int_encoding();
    bincode::serde::encode_into_std_write(&move_profiles, &mut file, bincode_config)?;
    println!("Wrote move profiles to disk at {}", config.output_file);

    println!("Finished!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_generate() -> Result<()> {
        // Run generation on a 5-by-5 board.
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
        }

        // Confirm the center is almost always occupied.
        // For some pieces, it may not be -- e.g. the L piece.
        assert!((center_occupied_count as f64) / (game_config.num_moves as f64) > 0.95);

        Ok(())
    }
}
