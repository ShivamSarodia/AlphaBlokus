use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct InferenceConfig {
    pub name: String,
    pub model_path: String,
    pub batch_size: usize,
}
