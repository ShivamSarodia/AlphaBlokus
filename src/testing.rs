use std::path::PathBuf;
use std::sync::Arc;

use crate::config::{GameConfig, MCTSConfig};
use once_cell::sync::Lazy;
use rand::Rng;

/// Creates a base GameConfig for testing with board size 5
fn create_base_game_config() -> GameConfig {
    GameConfig {
        board_size: 5,
        num_moves: 958,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data_file: PathBuf::from("static/move_data/tiny.bin"),
        move_data: None,
    }
}

fn create_base_half_game_config() -> GameConfig {
    GameConfig {
        board_size: 10,
        num_moves: 6233,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data_file: PathBuf::from("static/move_data/half.bin"),
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

pub fn create_mcts_config(num_rollouts: u32, temperature: f32) -> &'static MCTSConfig {
    Box::leak(Box::new(MCTSConfig {
        fast_move_probability: 0.0,
        fast_move_num_rollouts: num_rollouts,
        full_move_num_rollouts: num_rollouts,
        total_dirichlet_noise_alpha: 1.0,
        root_dirichlet_noise_fraction: 0.0,
        ucb_exploration_factor: 1.0,
        temperature_turn_cutoff: 10,
        move_selection_temperature: temperature,
        inference_config_name: "".to_string(),
    }))
}

pub fn create_tmp_directory() -> String {
    let path = std::env::temp_dir().join(format!(
        "alphablokus_test_{}",
        rand::rng().random_range(0..10000000)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path.to_string_lossy().to_string()
}
