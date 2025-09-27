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
            let channel_size = (config.num_concurrent_games * 2) as usize;
            let cancel_token = cancel_token.clone();

            let client = DefaultClient::from_inference_config(
                inference_config,
                &config.game,
                channel_size,
                cancel_token,
            )
            .await;

            inference_clients.insert(inference_config.name.clone(), Arc::new(client));
        }

        let (recorder, recorder_background_task) = Recorder::build_and_start(
            config.mcts_recorder.flush_row_count,
            config.mcts_recorder.data_directory.clone(),
        );

        let mut engine = Engine::new(
            config.num_concurrent_games,
            config.num_total_games,
            inference_clients,
            &config.game,
            &config.agents,
            recorder,
        );

        tokio::select! {
            _ = cancel_token.cancelled() => { }
            _ = engine.play_games() => { }
        };

        // Dropping the engine causes the recorder to be flushed, so the
        // background task can finish writing remaining data.
        drop(engine);

        recorder_background_task.await.unwrap();

        println!("Self-play complete.");
    })
}

#[cfg(test)]
mod tests {
    use crate::config::LoadableConfig;
    use std::path::Path;

    use super::*;

    #[test]
    fn test_run_selfplay() {
        let path = Path::new("tests/fixtures/configs/self_play.toml");
        let config: &'static mut SelfPlayConfig = SelfPlayConfig::from_file(path).unwrap();
        config.game.load_move_profiles().unwrap();
        run_selfplay(config);
    }
}
