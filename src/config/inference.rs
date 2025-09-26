use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct InferenceConfig {
    pub name: String,
    pub model_path: std::path::PathBuf,
    pub batch_size: usize,
}
