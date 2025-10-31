use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

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
    Tensorrt {
        max_batch_size: usize,
        pool_size: usize,
    },
}

#[derive(Deserialize, Debug, Clone)]
pub struct ReloadConfig {
    #[serde(
        rename = "poll_interval_seconds",
        with = "crate::config::inference::duration_seconds"
    )]
    pub poll_interval: Duration,
    #[serde(default = "ReloadConfig::default_s3_keep_last_models")]
    pub s3_keep_last_models: usize,
}

impl ReloadConfig {
    fn default_s3_keep_last_models() -> usize {
        1
    }
}

pub mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    #[allow(dead_code)]
    pub fn serialize<S>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(value.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(seconds))
    }
}
