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

#[derive(Deserialize, Debug)]
pub struct BenchmarkInferenceConfig {
    /// Number of concurrent threads to use for generating inference requests.
    pub num_concurrent_threads: u32,
    /// Duration of the benchmark in seconds.
    pub duration_seconds: u64,
    pub inference: InferenceConfig,
    pub game: GameConfig,
}

impl LoadableConfig for BenchmarkInferenceConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use std::fs;

    fn assert_configs_load<T: LoadableConfig + 'static>(directory: &str) -> Result<()> {
        let mut loaded = 0;
        for entry in fs::read_dir(directory)
            .with_context(|| format!("Failed to read directory {directory}"))?
        {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }

            T::from_file(&path)
                .with_context(|| format!("Failed to load config from {}", path.display()))?;
            loaded += 1;
        }

        assert!(loaded > 0, "No TOML configs found in {}", directory);
        Ok(())
    }

    #[test]
    fn benchmark_inference_configs_load() -> Result<()> {
        assert_configs_load::<BenchmarkInferenceConfig>("configs/benchmark_inference")
    }

    #[test]
    fn generate_move_data_configs_load() -> Result<()> {
        assert_configs_load::<PreprocessMovesConfig>("configs/generate_move_data")
    }

    #[test]
    fn self_play_configs_load() -> Result<()> {
        assert_configs_load::<SelfPlayConfig>("configs/self_play")
    }
}
