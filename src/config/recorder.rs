use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct MCTSRecorderConfig {
    pub data_directory: String,
    pub flush_row_count: usize,
}
