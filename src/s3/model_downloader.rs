use anyhow::{Context, Result};
use aws_sdk_s3::Client;
use rand::Rng;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::s3::{S3Uri, create_s3_client};

/// Manages downloading and caching of model files from S3.
pub struct S3ModelDownloader {
    s3_uri: S3Uri,
    cache_dir: PathBuf,
    cache_size: usize,
    client: Client,
}

impl S3ModelDownloader {
    /// Creates a new S3ModelDownloader.
    ///
    /// # Arguments
    /// * `s3_path` - S3 URI (e.g., "s3://bucket/path/to/models")
    /// * `cache_size` - Maximum number of models to keep in cache
    pub async fn new(s3_path: String, cache_size: usize) -> Result<Self> {
        let s3_uri = S3Uri::new(s3_path)?;
        let client = create_s3_client().await?;

        // Create a temporary cache directory with random suffix to avoid conflicts
        let random_suffix: u32 = rand::rng().random();
        let cache_dir =
            std::env::temp_dir().join(format!("alphablokus_model_cache_{}", random_suffix));
        fs::create_dir_all(&cache_dir).await?;

        tracing::info!("S3 model cache directory: {}", cache_dir.display());

        Ok(Self {
            s3_uri,
            cache_dir,
            cache_size,
            client,
        })
    }

    /// Syncs the latest model from S3 to local cache and returns the path.
    pub async fn sync_latest_model(&self) -> Result<PathBuf> {
        // If the S3Uri is a specific file, just download that single file
        if let Some(ref filename) = self.s3_uri.filename {
            return self.download_model(filename).await;
        }

        // Otherwise, we need to list all models in S3 and find the latest one.

        // List all models in S3
        let s3_models = self.list_s3_models().await?;

        // Get the latest model (lexicographically last)
        let latest_model = s3_models.last().ok_or_else(|| {
            anyhow::anyhow!(
                "No .onnx models found in S3 path: s3://{}/{}",
                self.s3_uri.bucket,
                self.s3_uri.key()
            )
        })?;

        // Download the model
        let local_path = self.download_model(latest_model).await?;

        // Clean up old models
        self.cleanup_old_models().await?;

        Ok(local_path)
    }

    /// Lists all .onnx files in the S3 bucket, sorted lexicographically.
    async fn list_s3_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        let prefix = if self.s3_uri.key().is_empty() {
            None
        } else {
            Some(self.s3_uri.key())
        };

        let mut continuation_token = None;

        loop {
            let mut request = self.client.list_objects_v2().bucket(&self.s3_uri.bucket);

            if let Some(ref prefix) = prefix {
                request = request.prefix(prefix);
            }

            if let Some(ref token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;

            for object in response.contents() {
                if let Some(key) = object.key()
                    && key.ends_with(".onnx")
                {
                    // Extract just the filename from the key
                    let filename = key.split('/').next_back().unwrap_or(key);
                    models.push(filename.to_string());
                }
            }

            if response.is_truncated().unwrap_or(false) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        models.sort();
        Ok(models)
    }

    /// Downloads a model (both .onnx and .onnx.data if exists) from S3.
    async fn download_model(&self, model_name: &str) -> Result<PathBuf> {
        // Check if we already have this model cached. If so, just return that directly.
        let local_path = self.cache_dir.join(model_name);
        if local_path.exists() {
            return Ok(local_path);
        }

        // Download the .onnx file
        tracing::info!("Downloading model from S3: {}", model_name);
        self.download_file(model_name, true).await?;

        // Try to download the .onnx.data file (it may not exist)
        let data_filename = format!("{}.data", model_name);
        self.download_file(&data_filename, false).await?;

        Ok(local_path)
    }

    /// Downloads a single file from S3 to the cache directory.
    async fn download_file(&self, filename: &str, required: bool) -> Result<()> {
        let final_s3_uri = self.s3_uri.with_filename(filename.to_string())?;
        let key = final_s3_uri.key();

        let response_result = self
            .client
            .get_object()
            .bucket(&self.s3_uri.bucket)
            .key(&key)
            .send()
            .await;

        let response = match response_result {
            Ok(response) => response,
            Err(e) if required => {
                return Err(e).context(format!("Failed to download {} from S3", filename));
            }
            Err(_) => {
                return Ok(());
            }
        };

        let local_path = self.cache_dir.join(filename);
        let mut file = fs::File::create(&local_path).await?;

        let mut body = response.body.into_async_read();
        tokio::io::copy(&mut body, &mut file).await?;
        file.flush().await?;

        tracing::debug!("Downloaded {} to {}", filename, local_path.display());
        Ok(())
    }

    /// Removes old models from cache, keeping only the most recent N models.
    async fn cleanup_old_models(&self) -> Result<()> {
        let mut entries = fs::read_dir(&self.cache_dir).await?;
        let mut onnx_files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "onnx") {
                onnx_files.push(path);
            }
        }

        // Sort by filename (lexicographic)
        onnx_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        // Remove old models if we exceed cache_size
        if onnx_files.len() > self.cache_size {
            let to_remove = onnx_files.len() - self.cache_size;
            for path in onnx_files.iter().take(to_remove) {
                tracing::info!("Removing old cached model: {}", path.display());

                // Remove the .onnx file
                fs::remove_file(path).await?;

                // Try to remove the associated .onnx.data file if it exists
                let data_path = path.with_extension("onnx.data");
                if data_path.exists() {
                    fs::remove_file(&data_path).await?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s3::S3Uri;

    // Test helper to create an S3ModelDownloader without needing S3 setup
    async fn create_test_downloader(cache_dir: PathBuf, cache_size: usize) -> S3ModelDownloader {
        // Set env var required by create_s3_client (not actually used in cleanup tests)
        unsafe {
            std::env::set_var("AWS_ENDPOINT_URL", "http://localhost:9000");
        }

        S3ModelDownloader {
            s3_uri: S3Uri::new("s3://test-bucket/models".to_string()).unwrap(),
            cache_dir,
            cache_size,
            client: create_s3_client().await.unwrap(),
        }
    }

    #[test]
    fn test_s3_uri_parsing() {
        // Test that S3Uri can be created from valid S3 paths
        let uri = S3Uri::new("s3://bucket/models".to_string()).unwrap();
        assert_eq!(uri.bucket, "bucket");
        assert_eq!(uri.key(), "models/");
    }

    #[test]
    fn test_cache_dir_creation() {
        // Test that we can instantiate the cache directory path
        let cache_dir = std::env::temp_dir().join("alphablokus_model_cache");
        assert!(
            cache_dir
                .to_string_lossy()
                .contains("alphablokus_model_cache")
        );
    }

    #[tokio::test]
    async fn test_cleanup_old_models() {
        use tokio::fs;

        // Create a temporary test cache directory
        let test_cache_dir = std::env::temp_dir().join(format!(
            "test_cleanup_models_{}",
            rand::rng().random::<u32>()
        ));
        fs::create_dir_all(&test_cache_dir).await.unwrap();

        // Create 5 test model files (lexicographically ordered)
        let model_files = vec![
            "model_001.onnx",
            "model_002.onnx",
            "model_003.onnx",
            "model_004.onnx",
            "model_005.onnx",
        ];

        for file in &model_files {
            let path = test_cache_dir.join(file);
            fs::File::create(&path).await.unwrap();

            let data_path = path.with_extension("onnx.data");
            fs::File::create(&data_path).await.unwrap();
        }

        // Create a downloader with cache_size = 3 (will keep 3 newest, delete 2 oldest)
        let downloader = create_test_downloader(test_cache_dir.clone(), 3).await;

        // Run cleanup
        downloader.cleanup_old_models().await.unwrap();

        // Verify the 2 oldest .onnx files were deleted
        assert!(!test_cache_dir.join("model_001.onnx").exists());
        assert!(!test_cache_dir.join("model_002.onnx").exists());

        // Verify the 3 newest .onnx files still exist
        assert!(test_cache_dir.join("model_003.onnx").exists());
        assert!(test_cache_dir.join("model_004.onnx").exists());
        assert!(test_cache_dir.join("model_005.onnx").exists());

        // Verify associated .onnx.data files were also deleted for old models
        assert!(!test_cache_dir.join("model_001.onnx.data").exists());
        assert!(!test_cache_dir.join("model_002.onnx.data").exists());

        // Verify .onnx.data files still exist for kept models
        assert!(test_cache_dir.join("model_003.onnx.data").exists());
        assert!(test_cache_dir.join("model_004.onnx.data").exists());
        assert!(test_cache_dir.join("model_005.onnx.data").exists());

        // Cleanup test directory
        let _ = fs::remove_dir_all(&test_cache_dir).await;
    }
}
