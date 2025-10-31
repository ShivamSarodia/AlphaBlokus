use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use aws_sdk_s3::Client;
use log::{info, warn};
use rand::{Rng, distr::Alphanumeric};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::time::sleep;

pub struct S3MirrorConfig {
    pub poll_interval: Duration,
    pub keep_last_models: usize,
}

pub async fn ensure_local_mirror(uri: &str, config: S3MirrorConfig) -> Result<PathBuf> {
    let parsed = crate::s3::ParsedUri::parse(uri)?;

    let client = crate::s3::build_client().await;
    let directory = parsed.directory.clone();
    let local_dir = create_local_directory(&parsed.bucket_name, directory.as_deref())?;

    let initial_key = match parsed.object_key() {
        Some(explicit_key) => explicit_key,
        None => fetch_latest_onnx_key(&client, &parsed.bucket_name, directory.as_deref()).await?,
    };
    let initial_path =
        download_object(&client, &parsed.bucket_name, &initial_key, &local_dir).await?;

    cleanup_old_models(
        &local_dir,
        initial_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
        config.keep_last_models,
    )
    .await?;

    spawn_background_sync(
        client,
        parsed.bucket_name.clone(),
        directory,
        local_dir.clone(),
        initial_key,
        config,
    );

    Ok(local_dir)
}

fn create_local_directory(bucket: &str, prefix: Option<&str>) -> Result<PathBuf> {
    let mut base = std::env::temp_dir();
    base.push("alpha_blokus_models");
    std::fs::create_dir_all(&base).with_context(|| {
        format!(
            "Failed to create base model cache directory at {}",
            base.display()
        )
    })?;

    let sanitized_prefix = prefix.unwrap_or("").replace('/', "_");
    let mut rng = rand::rng();
    let random_suffix: String = (&mut rng)
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();

    let mut directory = base;
    directory.push(format!("{}_{}_{}", bucket, sanitized_prefix, random_suffix));
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "Failed to create local model cache directory at {}",
            directory.display()
        )
    })?;

    Ok(directory)
}

async fn fetch_latest_onnx_key(
    client: &Client,
    bucket: &str,
    prefix: Option<&str>,
) -> Result<String> {
    let mut keys = Vec::new();
    let mut continuation_token = None;

    loop {
        let mut request = client.list_objects_v2().bucket(bucket);

        if let Some(prefix) = prefix {
            let trimmed = prefix.trim_matches('/');
            if !trimmed.is_empty() {
                request = request.prefix(format!("{}/", trimmed));
            }
        }

        if let Some(token) = &continuation_token {
            request = request.continuation_token(token);
        }

        let response = request.send().await?;

        for object in response.contents() {
            if let Some(key) = object.key() {
                if key.ends_with(".onnx") {
                    keys.push(key.to_string());
                }
            }
        }

        if response.is_truncated().unwrap_or(false) {
            continuation_token = response
                .next_continuation_token()
                .map(|token| token.to_string());
        } else {
            break;
        }
    }

    if keys.is_empty() {
        bail!(
            "No .onnx files found for s3://{}/{}",
            bucket,
            prefix.unwrap_or("")
        );
    }

    keys.sort();
    Ok(keys.pop().unwrap())
}

async fn download_object(
    client: &Client,
    bucket: &str,
    key: &str,
    local_dir: &Path,
) -> Result<PathBuf> {
    let response = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .with_context(|| format!("Failed to download s3://{}/{}", bucket, key))?;

    let file_name = Path::new(key)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("Could not determine file name for key {}", key))?;

    let final_path = local_dir.join(file_name);
    let tmp_path = local_dir.join(format!("{}.download", file_name));

    let mut body = response.body.into_async_read();
    let mut file = fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("Failed to create {}", tmp_path.display()))?;

    tokio::io::copy(&mut body, &mut file)
        .await
        .with_context(|| format!("Failed to write to {}", tmp_path.display()))?;
    file.flush().await?;

    drop(file);

    fs::rename(&tmp_path, &final_path).await.with_context(|| {
        format!(
            "Failed to rename {} to {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    Ok(final_path)
}

async fn cleanup_old_models(
    local_dir: &Path,
    current_file_name: &str,
    keep_last_models: usize,
) -> Result<()> {
    let keep_last = keep_last_models.max(1);

    let mut entries = fs::read_dir(local_dir)
        .await
        .with_context(|| format!("Failed to read {}", local_dir.display()))?;

    let mut old_files = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("onnx") {
            continue;
        }

        if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name == current_file_name)
            .unwrap_or(false)
        {
            continue;
        }

        old_files.push(path);
    }

    old_files.sort();

    if old_files.len() <= keep_last - 1 {
        return Ok(());
    }

    let files_to_remove = old_files.len().saturating_sub(keep_last - 1);
    for path in old_files.into_iter().take(files_to_remove) {
        if let Err(err) = fs::remove_file(&path).await {
            warn!("Failed to remove cached model {}: {}", path.display(), err);
        }
    }

    Ok(())
}

fn spawn_background_sync(
    client: Client,
    bucket: String,
    prefix: Option<String>,
    local_dir: PathBuf,
    initial_key: String,
    config: S3MirrorConfig,
) {
    let poll_interval = config.poll_interval;
    let keep_last_models = config.keep_last_models.max(1);

    tokio::spawn(async move {
        let mut current_key = initial_key;

        loop {
            sleep(poll_interval).await;

            match fetch_latest_onnx_key(&client, &bucket, prefix.as_deref()).await {
                Ok(latest_key) if latest_key != current_key => {
                    match download_object(&client, &bucket, &latest_key, &local_dir).await {
                        Ok(new_path) => {
                            let new_file_name = new_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .map(|name| name.to_string())
                                .unwrap_or_default();

                            info!(
                                "Downloaded new model {} from s3://{}/{}",
                                new_file_name, bucket, latest_key
                            );

                            if let Err(err) =
                                cleanup_old_models(&local_dir, &new_file_name, keep_last_models)
                                    .await
                            {
                                warn!(
                                    "Failed to cleanup cached models in {}: {}",
                                    local_dir.display(),
                                    err
                                );
                            }

                            current_key = latest_key;
                        }
                        Err(err) => warn!("{}", err),
                    }
                }
                Ok(_) => {}
                Err(err) => warn!(
                    "Failed to check for new models in s3://{}/{}: {}",
                    bucket,
                    prefix.as_deref().unwrap_or(""),
                    err
                ),
            }
        }
    });
}
