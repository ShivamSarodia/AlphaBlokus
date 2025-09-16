use serde::Deserialize;

/// An agent group config describes the methodology for selecting agents for
/// each game.
#[derive(Deserialize, Debug)]
pub enum AgentGroupConfig {
    /// The same agent is used by all four players.
    Single(AgentConfig),
}

/// An agent config describes the type and behavior of a particular agent.
#[derive(Deserialize, Debug)]
pub enum AgentConfig {
    MCTS(MCTSConfig),
    Random,
}

#[derive(Deserialize, Debug)]
pub struct MCTSConfig {
    pub num_rollouts: u32,
    pub total_dirichlet_noise_alpha: f32,
    pub root_dirichlet_noise_fraction: f32,
    pub ucb_exploration_factor: f32,
    pub temperature_turn_cutoff: u16,
    pub move_selection_temperature: f32,
    /// The name of the inference config that the engine should pass to the
    /// MCTS agent. The config file must contain an inference config with this
    /// name.
    pub inference_config_name: String,
}
