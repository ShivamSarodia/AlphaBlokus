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

    #[test]
    fn test_generate() -> Result<()> {
        let num_moves = 958;
        let game_config = GameConfig {
            board_size: 5,
            num_moves: num_moves,
            num_pieces: 21,
            num_piece_orientations: 91,
            moves_file_path: "".to_string(),
        };
        let move_profiles = generate(&game_config)?;

        // Confirm the output indices are correct.
        for i in 0..num_moves {
            assert_eq!(move_profiles.get(i).index, i);
        }

        Ok(())
    }
}
