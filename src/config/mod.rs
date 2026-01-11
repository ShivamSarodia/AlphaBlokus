mod agents;
mod game;
mod inference;
mod observability;
mod parents;
mod recorder;

pub const NUM_PLAYERS: usize = 4;
pub use agents::{
    AgentConfig, AgentGroupConfig, DefaultExploitationValue, MCTSConfig, PentobiConfig,
    PolicySamplingConfig, RandomConfig,
};
pub use game::GameConfig;
pub use inference::{
    ExecutorConfig, InferenceCacheConfig, InferenceConfig, OrtExecutionProvider, ReloadConfig,
};
pub use observability::{LoggingConfig, MetricsConfig, ObservabilityConfig};
pub use parents::{
    BenchmarkInferenceConfig, LoadableConfig, MCTSAnalyzerConfig, PreprocessMovesConfig,
    SelfPlayConfig, WebPlayConfig,
};
pub use recorder::MCTSRecorderConfig;
