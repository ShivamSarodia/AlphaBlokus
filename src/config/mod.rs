mod components;
mod parents;

pub const NUM_PLAYERS: usize = 4;
pub use components::GameConfig;
pub use components::MCTSConfig;
pub use parents::{LoadableConfig, PreprocessMovesConfig, SelfPlayConfig};
