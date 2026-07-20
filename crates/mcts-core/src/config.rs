use serde::{Deserialize, Serialize};

fn default_agent_name() -> String {
    "unnamed".to_string()
}

fn default_empty_string() -> String {
    String::new()
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
    /// Native inference configuration used by `MCTSAgent`.
    pub inference_config_name: String,
    #[serde(default = "default_empty_string")]
    pub policy_inference_config_name: String,
    #[serde(default = "default_empty_string")]
    pub value_inference_config_name: String,
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
