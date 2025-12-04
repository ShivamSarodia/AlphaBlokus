use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;

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
pub struct MCTSConfig {
    #[serde(default = "default_agent_name")]
    pub name: String,
    pub fast_move_probability: f32,
    pub fast_move_num_rollouts: u32,
    pub full_move_num_rollouts: u32,
    pub total_dirichlet_noise_alpha: f32,
    pub root_dirichlet_noise_fraction: f32,
    pub ucb_exploration_factor: f32,
    pub temperature_turn_cutoff: u16,
    pub move_selection_temperature: f32,
    #[serde(default)]
    pub default_exploitation_value: DefaultExploitationValue,
    /// The name of the inference config that the engine should pass to the
    /// MCTS agent. The config file must contain an inference config with this
    /// name.
    pub inference_config_name: String,
    /// If provided, saves debug information for each move as a new file in this directory.
    #[serde(default)]
    pub trace_file: Option<PathBuf>,
}

impl MCTSConfig {
    pub fn tracing_enabled(&self) -> bool {
        self.trace_file.is_some()
    }
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

#[derive(Deserialize, Debug, Serialize, Clone, Copy, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DefaultExploitationValue {
    #[default]
    NetworkValue,
    FixedValue {
        value: f32,
    },
}
