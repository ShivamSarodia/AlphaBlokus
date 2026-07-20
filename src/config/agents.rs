use std::path::PathBuf;

use mcts_core::MCTSConfig;
use serde::Deserialize;

fn default_agent_name() -> String {
    "unnamed".to_string()
}

/// An agent group config describes the methodology for selecting agents for
/// each game.
#[derive(Deserialize, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum AgentGroupConfig {
    /// The same agent is used by all four players.
    Single(AgentConfig),
    /// Four different agents are used, one for each player. The order is
    /// randomized for each game.
    QuadArena([AgentConfig; 4]),
    /// Two different agents are used, with each agent playing twice. The order
    /// is randomized for each game.
    DuoArena([AgentConfig; 2]),
}

/// An agent config describes the type and behavior of a particular agent.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentConfig {
    #[serde(rename = "mcts")]
    MCTS(MCTSConfig),
    Random(RandomConfig),
    PolicySampling(PolicySamplingConfig),
    Pentobi(PentobiConfig),
}

#[derive(Deserialize, Debug)]
pub struct RandomConfig {
    #[serde(default = "default_agent_name")]
    pub name: String,
    /// If true, sample only from moves with the largest occupied cell count.
    #[serde(default)]
    pub from_largest: bool,
}

#[derive(Deserialize, Debug)]
pub struct PolicySamplingConfig {
    #[serde(default = "default_agent_name")]
    pub name: String,
    /// The name of the inference config that the engine should pass to the
    /// policy sampling agent. The config file must contain an inference config with this
    /// name.
    pub inference_config_name: String,
    /// Temperature used to scale the policy probabilities before sampling.
    pub temperature: f32,
}

#[derive(Deserialize, Debug)]
pub struct PentobiConfig {
    #[serde(default = "default_agent_name")]
    pub name: String,
    pub binary_path: PathBuf,
    pub opening_book: PathBuf,
    pub level: u8,
}
