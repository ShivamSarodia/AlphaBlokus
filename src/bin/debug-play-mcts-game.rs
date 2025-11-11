use std::sync::Arc;

use alpha_blokus::agents::Agent;
use alpha_blokus::inference::OrtExecutor;
use anyhow::Result;
use log::debug;
use std::path::{Path, PathBuf};

use alpha_blokus::agents::MCTSAgent;
use alpha_blokus::config::GameConfig;
use alpha_blokus::config::MCTSConfig;
use alpha_blokus::game::{GameStatus, State};
use alpha_blokus::inference;
use alpha_blokus::utils;

fn main() -> Result<()> {
    debug!("Debug logging enabled.");

    let game_config: &'static mut GameConfig = Box::leak(Box::new(GameConfig {
        board_size: 10,
        num_moves: 6233,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data: None,
        move_data_file: PathBuf::from("static/move_data/half.bin"),
    }));
    game_config.load_move_profiles()?;

    let mcts_config: &'static mut MCTSConfig = Box::leak(Box::new(MCTSConfig {
        name: "debug_mcts".to_string(),
        fast_move_probability: 0.0,
        fast_move_num_rollouts: 10,
        full_move_num_rollouts: 10,
        total_dirichlet_noise_alpha: 10.83,
        root_dirichlet_noise_fraction: 0.25,
        ucb_exploration_factor: 1.05,
        temperature_turn_cutoff: 24,
        move_selection_temperature: 1.0,
        inference_config_name: "".to_string(),
    }));

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let cancel_token = utils::setup_cancel_token();

        let executor = OrtExecutor::build(
            Path::new("static/networks/trivial_net_half.onnx"),
            game_config,
        )?;

        let inference_client = Arc::new(inference::DefaultClient::build_and_start(
            executor,
            1,
            cancel_token,
        ));

        let mut agent = MCTSAgent::new(mcts_config, game_config, Arc::clone(&inference_client));
        let mut state = State::new(game_config);
        loop {
            let move_index = agent.choose_move(&state).await;
            let status = state.apply_move(move_index);
            println!("{}", state);
            if status == GameStatus::GameOver {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}
