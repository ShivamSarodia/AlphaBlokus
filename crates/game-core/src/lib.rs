//! Target-portable AlphaBlokus rules and move-table types.

#[cfg(not(target_arch = "wasm32"))]
mod config_native;

pub mod config;
pub mod game;

#[cfg(test)]
pub mod testing {
    use std::path::PathBuf;

    use crate::config::GameConfig;

    fn tiny_move_data_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../static/move_data/tiny.bin")
    }

    pub fn create_game_config_without_data() -> &'static GameConfig {
        Box::leak(Box::new(GameConfig {
            board_size: 5,
            num_moves: 958,
            num_pieces: 21,
            num_piece_orientations: 91,
            move_data_file: tiny_move_data_path(),
            move_data: None,
        }))
    }

    pub fn create_game_config() -> &'static GameConfig {
        let mut config = GameConfig {
            board_size: 5,
            num_moves: 958,
            num_pieces: 21,
            num_piece_orientations: 91,
            move_data_file: tiny_move_data_path(),
            move_data: None,
        };
        config.load_move_profiles().unwrap();
        Box::leak(Box::new(config))
    }
}
