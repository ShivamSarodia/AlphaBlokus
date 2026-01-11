use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct InferenceConfig {
    pub name: String,
    pub batch_size: usize,
    pub model_path: String,
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub reload: Option<ReloadConfig>,
    #[serde(default)]
    pub cache: InferenceCacheConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct InferenceCacheConfig {
    #[serde(default = "default_cache_max_entries")]
    pub max_entries: usize,
}

impl Default for InferenceCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: default_cache_max_entries(),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutorConfig {
    Ort {
        execution_provider: OrtExecutionProvider,
    },
    Random {
        sleep_duration_ms: u64,
    },
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

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrtExecutionProvider {
    Cpu,
    #[serde(alias = "core_ml")]
    Coreml,
}

fn default_s3_cache_size() -> usize {
    3
}

fn default_cache_max_entries() -> usize {
    0
}
