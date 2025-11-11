use crate::utils;
use crate::{
    config::SelfPlayConfig, gameplay::Engine, inference::DefaultClient, recorder::Recorder,
};
use ahash::AHashMap as HashMap;
use std::sync::Arc;

pub fn run_selfplay(config: &'static SelfPlayConfig) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let mut inference_clients = HashMap::<String, Arc<DefaultClient>>::new();

        let cancel_token = utils::setup_cancel_token();

        for inference_config in &config.inference {
            let cancel_token = cancel_token.clone();

            let client =
                DefaultClient::from_inference_config(inference_config, &config.game, cancel_token)
                    .await;

            inference_clients.insert(inference_config.name.clone(), Arc::new(client));
        }

        let (recorder, recorder_background_task) = match &config.mcts_recorder {
            crate::config::MCTSRecorderConfig::Disabled => Recorder::disabled(),
            crate::config::MCTSRecorderConfig::Directory {
                data_directory,
                flush_row_count,
            } => Recorder::build_and_start(*flush_row_count, data_directory.clone()),
        };

        let mut engine = Engine::new(
            config.num_concurrent_games,
            config.num_total_games,
            inference_clients,
            &config.game,
            &config.agents,
            recorder,
        );

        // Start a timer that cancels the token after the specified duration
        if config.duration_seconds > 0 {
            let cancel_token_clone = cancel_token.clone();
            let duration_seconds = config.duration_seconds;
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(duration_seconds)).await;
                tracing::info!(
                    "Duration limit of {} seconds reached, stopping self-play.",
                    duration_seconds
                );
                cancel_token_clone.cancel();
            });
        }

        tokio::select! {
            _ = cancel_token.cancelled() => { }
            _ = engine.play_games() => { }
        };

        // Dropping the engine causes the recorder to be flushed, so the
        // background task can finish writing remaining data.
        drop(engine);

        recorder_background_task.await.unwrap();

        tracing::info!("Self-play complete.");
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AgentConfig, AgentGroupConfig, ExecutorConfig, GameConfig, InferenceConfig,
        MCTSRecorderConfig, SelfPlayConfig,
    };
    use std::path::PathBuf;

    #[test]
    fn test_run_selfplay() {
        let config = SelfPlayConfig {
            game: GameConfig {
                board_size: 5,
                move_data_file: PathBuf::from("static/move_data/tiny.bin"),
                num_moves: 958,
                num_pieces: 21,
                num_piece_orientations: 91,
                move_data: None,
            },
            agents: AgentGroupConfig::Single(AgentConfig::Random(crate::config::RandomConfig {
                name: "test_random".to_string(),
            })),
            inference: vec![InferenceConfig {
                name: "default".to_string(),
                batch_size: 2,
                model_path: "static/networks/trivial_net_tiny.onnx".to_string(),
                executor: ExecutorConfig::Ort,
                reload: None,
            }],
            num_concurrent_games: 10,
            num_total_games: 100,
            duration_seconds: 0,
            mcts_recorder: MCTSRecorderConfig::Directory {
                data_directory: "/tmp/alphablokus_test_fixture".to_string(),
                flush_row_count: 10,
            },
            observability: Default::default(),
        };

        let config: &'static mut SelfPlayConfig = Box::leak(Box::new(config));
        config.game.load_move_profiles().unwrap();

        run_selfplay(config);
    }

    #[test]
    fn test_run_selfplay_with_duration_limit() {
        let config = SelfPlayConfig {
            game: GameConfig {
                board_size: 5,
                move_data_file: PathBuf::from("static/move_data/tiny.bin"),
                num_moves: 958,
                num_pieces: 21,
                num_piece_orientations: 91,
                move_data: None,
            },
            agents: AgentGroupConfig::Single(AgentConfig::Random(crate::config::RandomConfig {
                name: "test_random".to_string(),
            })),
            inference: vec![InferenceConfig {
                name: "default".to_string(),
                batch_size: 2,
                model_path: "static/networks/trivial_net_tiny.onnx".to_string(),
                executor: ExecutorConfig::Ort,
                reload: None,
            }],
            num_concurrent_games: 10,
            num_total_games: 0, // Infinite games - would run forever without duration limit
            duration_seconds: 2, // Should stop after 2 seconds
            mcts_recorder: MCTSRecorderConfig::Directory {
                data_directory: "/tmp/alphablokus_test_fixture_duration".to_string(),
                flush_row_count: 10,
            },
            observability: Default::default(),
        };

        let config: &'static mut SelfPlayConfig = Box::leak(Box::new(config));
        config.game.load_move_profiles().unwrap();

        let start = std::time::Instant::now();
        run_selfplay(config);
        let elapsed = start.elapsed();

        // Verify it took approximately 2 seconds (within a reasonable margin)
        // Allow 1.8 to 3.0 seconds to account for startup time and shutdown time
        assert!(
            elapsed.as_secs_f64() >= 1.8 && elapsed.as_secs_f64() <= 3.0,
            "Expected duration to be around 2 seconds, but it took {:.2} seconds",
            elapsed.as_secs_f64()
        );
    }
}
