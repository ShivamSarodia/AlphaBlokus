use crate::{
    config::{AgentConfig, AgentGroupConfig, SelfPlayConfig},
    gameplay::Engine,
};

pub fn run_selfplay(config: &'static SelfPlayConfig) -> u32 {
    let mut engine = Engine::new(
        config.num_concurrent_games,
        config.num_total_games,
        &config.game,
        &AgentGroupConfig::Single(AgentConfig::Random),
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(engine.play_games())
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
