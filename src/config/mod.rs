mod game;
mod mcts;
mod parents;

pub const NUM_PLAYERS: usize = 4;
pub use game::GameConfig;
pub use mcts::MCTSConfig;
pub use parents::{LoadableConfig, PreprocessMovesConfig, SelfPlayConfig};
