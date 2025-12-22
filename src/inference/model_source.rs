use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::s3::S3ModelDownloader;

/// Trait for abstracting model discovery from different sources (local filesystem or S3).
#[async_trait]
pub trait ModelSource: Send + Sync {
    /// Returns the path to the latest model file.
    /// For local sources, this is the actual model path.
    /// For S3 sources, this is the path to the cached model.
    async fn get_latest_model(&self) -> Result<PathBuf>;
}

/// Model source that reads from a local directory.
pub struct LocalModelSource {
    model_dir: PathBuf,
}

impl LocalModelSource {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    async fn latest_model_file_async(&self) -> Result<PathBuf> {
        let mut entries = fs::read_dir(&self.model_dir).await?;
        let mut candidates = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "onnx") {
                candidates.push(path);
            }
        }

        candidates.retain(|path| path.file_name().is_some());
        candidates.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        let path = candidates.pop().ok_or_else(|| {
            anyhow::anyhow!(
                "No .onnx files found in directory: {}",
                self.model_dir.display()
            )
        })?;

        Ok(path.canonicalize().unwrap_or(path))
    }
}

#[async_trait]
impl ModelSource for LocalModelSource {
    async fn get_latest_model(&self) -> Result<PathBuf> {
        self.latest_model_file_async().await
    }
}

/// Model source that reads from S3 and caches locally.
pub struct S3ModelSource {
    downloader: S3ModelDownloader,
}

impl S3ModelSource {
    pub fn new(downloader: S3ModelDownloader) -> Self {
        Self { downloader }
    }
}

#[async_trait]
impl ModelSource for S3ModelSource {
    async fn get_latest_model(&self) -> Result<PathBuf> {
        self.downloader.sync_latest_model().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::fs;

    #[tokio::test]
    async fn test_local_model_source_finds_latest() {
        // Create a temporary directory with test .onnx files
        let temp_dir = std::env::temp_dir().join("test_model_source");
        let _ = fs::create_dir_all(&temp_dir).await;

        // Create some test .onnx files
        fs::write(temp_dir.join("model_001.onnx"), b"test1")
            .await
            .unwrap();
        fs::write(temp_dir.join("model_002.onnx"), b"test2")
            .await
            .unwrap();
        fs::write(temp_dir.join("model_003.onnx"), b"test3")
            .await
            .unwrap();

        let source = LocalModelSource::new(&temp_dir);
        let latest = source.get_latest_model().await.unwrap();

        // Should get the lexicographically last file
        assert!(
            latest
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("model_003")
        );

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_local_model_source_no_files() {
        let temp_dir = std::env::temp_dir().join("test_model_source_empty");
        let _ = fs::create_dir_all(&temp_dir).await;

        let source = LocalModelSource::new(&temp_dir);
        let result = source.get_latest_model().await;

        // Should error when no .onnx files found
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("No .onnx files found in directory:"));
        assert!(error_message.contains("test_model_source_empty"));

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }
}
