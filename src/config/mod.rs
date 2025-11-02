mod agents;
mod game;
mod inference;
mod observability;
mod parents;
mod recorder;

pub const NUM_PLAYERS: usize = 4;
pub use agents::{AgentConfig, AgentGroupConfig, MCTSConfig};
pub use game::GameConfig;
pub use inference::{ExecutorConfig, InferenceConfig, ReloadConfig};
pub use observability::{LoggingConfig, MetricsConfig, ObservabilityConfig};
pub use parents::{
    BenchmarkInferenceConfig, LoadableConfig, PreprocessMovesConfig, SelfPlayConfig,
};
pub use recorder::MCTSRecorderConfig;
