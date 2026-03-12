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

fn default_game_result_flush_row_count() -> usize {
    10
}

/// Configuration for finished game result recording.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameResultRecorderConfig {
    /// No data recording (disabled).
    #[default]
    Disabled,
    /// Record compact finished-game rows to a local JSONL file.
    JsonlFile {
        path: String,
        #[serde(default = "default_game_result_flush_row_count")]
        flush_row_count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_result_recorder_defaults_to_disabled_when_missing() {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(default)]
            game_result_recorder: GameResultRecorderConfig,
        }

        let wrapper: Wrapper = toml::from_str("").unwrap();
        assert!(matches!(
            wrapper.game_result_recorder,
            GameResultRecorderConfig::Disabled
        ));
    }

    #[test]
    fn game_result_recorder_jsonl_defaults_flush_count_to_ten() {
        let config: GameResultRecorderConfig = toml::from_str(
            r#"
type = "jsonl_file"
path = "/tmp/results.jsonl"
"#,
        )
        .unwrap();

        match config {
            GameResultRecorderConfig::JsonlFile {
                path,
                flush_row_count,
            } => {
                assert_eq!(path, "/tmp/results.jsonl");
                assert_eq!(flush_row_count, 10);
            }
            GameResultRecorderConfig::Disabled => panic!("expected jsonl_file config"),
        }
    }
}
