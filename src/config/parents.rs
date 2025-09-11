use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::config::components::GameConfig;

pub trait LoadableConfig: Sized + DeserializeOwned {
    fn from_string(string: &str) -> Result<Self> {
        let config = toml::from_str::<Self>(string).context("Failed to parse config")?;
        Ok(config)
    }

    fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from path: {}", path.display()))?;
        Self::from_string(&contents)
    }
}

#[derive(Deserialize, Debug)]
pub struct SelfPlayConfig {
    pub game: GameConfig,
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
        assert_eq!(self_play_config.game.board_size, 20);
        Ok(())
    }
}
