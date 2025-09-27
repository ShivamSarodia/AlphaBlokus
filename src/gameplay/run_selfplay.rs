use crate::{
    config::SelfPlayConfig, gameplay::Engine, inference::DefaultClient, recorder::Recorder,
};
use ahash::AHashMap as HashMap;
use std::sync::Arc;

pub fn run_selfplay(config: &'static SelfPlayConfig) -> u32 {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let mut inference_clients = HashMap::<String, Arc<DefaultClient>>::new();

        for inference_config in &config.inference {
            let channel_size = (config.num_concurrent_games * 2) as usize;

            let client =
                DefaultClient::from_inference_config(inference_config, &config.game, channel_size)
                    .await;

            inference_clients.insert(inference_config.name.clone(), Arc::new(client));
        }

        let mut engine = Engine::new(
            config.num_concurrent_games,
            config.num_total_games,
            inference_clients,
            &config.game,
            &config.agents,
            Recorder::build_and_start(
                config.mcts_recorder.flush_row_count,
                config.mcts_recorder.data_directory.clone(),
            ),
        );

        engine.play_games().await
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

        let expected_num_finished_games = config.num_total_games;
        let num_finished_games = run_selfplay(config);

        assert_eq!(num_finished_games, expected_num_finished_games);
    }
}
