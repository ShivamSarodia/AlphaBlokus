use std::sync::Arc;

use crate::config::GameConfig;
use once_cell::sync::Lazy;

/// Creates a base GameConfig for testing with board size 5
fn create_base_game_config() -> GameConfig {
    GameConfig {
        board_size: 5,
        num_moves: 958,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data_file: "static/move_data/tiny.bin".to_string(),
        move_data: None,
    }
}

fn create_base_half_game_config() -> GameConfig {
    GameConfig {
        board_size: 10,
        num_moves: 6233,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data_file: "static/move_data/half.bin".to_string(),
        move_data: None,
    }
}

/// Static GameConfig without move data, cached using once_cell
static GAME_CONFIG_WITHOUT_DATA: Lazy<Arc<GameConfig>> =
    Lazy::new(|| Arc::new(create_base_game_config()));

/// Static GameConfig with move data loaded, cached using once_cell
static GAME_CONFIG_WITH_DATA: Lazy<Arc<GameConfig>> = Lazy::new(|| {
    let mut config = create_base_game_config();
    config.load_move_profiles().unwrap();
    Arc::new(config)
});

static HALF_GAME_CONFIG_WITHOUT_DATA: Lazy<Arc<GameConfig>> =
    Lazy::new(|| Arc::new(create_base_half_game_config()));

static HALF_GAME_CONFIG_WITH_DATA: Lazy<Arc<GameConfig>> = Lazy::new(|| {
    let mut config = create_base_half_game_config();
    config.load_move_profiles().unwrap();
    Arc::new(config)
});

/// Creates a GameConfig for testing with board size 5 without loading move profiles.
/// This function returns the same cached instance on every call.
pub fn create_game_config_without_data() -> &'static GameConfig {
    &GAME_CONFIG_WITHOUT_DATA
}

/// Creates a GameConfig for testing with board size 5 with move profiles loaded.
/// This function calls create_game_config_without_data() and then loads move profiles.
/// Returns the same cached instance on every call to avoid reloading.
pub fn create_game_config() -> &'static GameConfig {
    &GAME_CONFIG_WITH_DATA
}

pub fn create_half_game_config_without_data() -> &'static GameConfig {
    &HALF_GAME_CONFIG_WITHOUT_DATA
}

pub fn create_half_game_config() -> &'static GameConfig {
    &HALF_GAME_CONFIG_WITH_DATA
}
