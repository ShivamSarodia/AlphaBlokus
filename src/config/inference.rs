use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct InferenceConfig {
    pub name: String,
    pub batch_size: usize,
    pub model_path: String,
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub reload: Option<ReloadConfig>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutorConfig {
    Ort,
    Tensorrt {
        max_batch_size: usize,
        pool_size: usize,
    },
}

#[derive(Deserialize, Debug, Clone)]
pub struct ReloadConfig {
    pub poll_interval_seconds: u64,
    #[serde(default = "default_s3_cache_size")]
    pub s3_cache_size: usize,
}

fn default_s3_cache_size() -> usize {
    3
}
