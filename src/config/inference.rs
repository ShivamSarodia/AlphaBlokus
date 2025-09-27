use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Clone)]
pub struct InferenceConfig {
    pub name: String,
    pub batch_size: usize,
    pub model_path: PathBuf,
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub reload: Option<ReloadConfig>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutorConfig {
    Ort,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ReloadConfig {
    pub poll_interval_seconds: u64,
}
