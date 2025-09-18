use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::config::{AgentGroupConfig, GameConfig, InferenceConfig, MCTSRecorderConfig};

pub trait LoadableConfig: Sized + DeserializeOwned {
    /// Given a string, parse it into a config and produce a static
    /// reference to it.
    fn from_string(string: &str) -> Result<&'static mut Self> {
        let config = toml::from_str::<Self>(string).context("Failed to parse config")?;
        Ok(Box::leak(Box::new(config)))
    }

    /// Given a path, read the file into a string and produce a static
    /// reference to it.
    fn from_file(path: &Path) -> Result<&'static mut Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from path: {}", path.display()))?;
        Self::from_string(&contents)
    }
}

#[derive(Deserialize, Debug)]
pub struct SelfPlayConfig {
    pub game: GameConfig,
    pub agents: AgentGroupConfig,
    pub inference: Vec<InferenceConfig>,
    pub num_concurrent_games: u32,
    pub num_total_games: u32,
    pub mcts_recorder: MCTSRecorderConfig,
}

impl LoadableConfig for SelfPlayConfig {}

#[derive(Deserialize, Debug)]
pub struct PreprocessMovesConfig {
    pub game: GameConfig,
}

impl LoadableConfig for PreprocessMovesConfig {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_from_file() -> Result<()> {
        let path = Path::new("tests/fixtures/configs/self_play.toml");
        let self_play_config = SelfPlayConfig::from_file(path)?;
        assert_eq!(self_play_config.game.board_size, 5);
        Ok(())
    }
}
