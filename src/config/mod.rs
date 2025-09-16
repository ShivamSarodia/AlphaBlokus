mod agents;
mod game;
mod inference;
mod parents;

pub const NUM_PLAYERS: usize = 4;
pub use agents::{AgentConfig, AgentGroupConfig, MCTSConfig};
pub use game::GameConfig;
pub use inference::InferenceConfig;
pub use parents::{LoadableConfig, PreprocessMovesConfig, SelfPlayConfig};
