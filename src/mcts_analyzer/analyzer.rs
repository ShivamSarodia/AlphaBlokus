use std::{path::PathBuf, sync::Arc};

use tokio::fs;
use tokio_util::sync::CancellationToken;

use crate::{
    agents::{MCTSAgent, MCTSNodeAnalysis},
    config::{GameConfig, MCTSAnalyzerConfig},
    game::{SerializableState, State},
    gameplay::build_inference_clients,
};

pub fn run(config: &'static MCTSAnalyzerConfig) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let inference_clients =
            build_inference_clients(&config.inference, &config.game, CancellationToken::new())
                .await;
        let inference_client =
            Arc::clone(&inference_clients[&config.mcts_config.inference_config_name]);

        let mut mcts_agent = MCTSAgent::new(&config.mcts_config, &config.game, inference_client);

        let state = read_state_from_file(&config.state_file, &config.game).await;

        println!("Beginning MCTS...");
        let (move_index, search_root) = mcts_agent.choose_move_with_node(&state).await;
        println!("MCTS complete.");

        let _analysis = MCTSNodeAnalysis::from_node(&search_root);

        println!("Move index: {}", move_index);
    });
}

async fn read_state_from_file(path: &PathBuf, game_config: &'static GameConfig) -> State {
    let data = fs::read_to_string(path)
        .await
        .expect("Failed to read state file");
    let state: SerializableState = serde_json::from_str(&data).expect("Failed to parse state file");
    state.to_state(game_config)
}
