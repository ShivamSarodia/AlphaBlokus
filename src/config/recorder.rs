use serde::Deserialize;

/// Configuration for MCTS data recording.
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCTSRecorderConfig {
    /// No data recording (disabled).
    Disabled,
    /// Record data to a directory (local or S3).
    Directory {
        data_directory: String,
        flush_row_count: usize,
    },
}
