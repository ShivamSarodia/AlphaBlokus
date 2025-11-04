// Integration test for S3 model reloading with Cloudflare R2
//
// You can also use `source .env` to load environment variables, then run the test.

use alpha_blokus::inference::{ModelSource, S3ModelSource};
use alpha_blokus::s3::S3ModelDownloader;
use std::time::Duration;
use tokio::fs;

#[tokio::test]
#[ignore]
async fn test_s3_reload_integration() {
    dotenvy::dotenv().unwrap();

    // Test configuration - adjust this to match your R2 bucket
    let test_bucket = "alpha-blokus-staging";
    let test_prefix = format!(
        "integration-test-model-{}",
        chrono::Utc::now().timestamp_millis()
    );
    let s3_uri = format!("s3://{}/{}", test_bucket, test_prefix);

    println!("Testing S3 URI: {}", s3_uri);

    // Step 1: Create test model files locally
    let test_dir = std::env::temp_dir().join("s3_integration_test_upload");
    fs::create_dir_all(&test_dir).await.unwrap();

    let model1_path = test_dir.join("model_001.onnx");
    let model2_path = test_dir.join("model_002.onnx");
    fs::write(&model1_path, b"fake model 1 data").await.unwrap();
    fs::write(&model2_path, b"fake model 2 data").await.unwrap();

    println!("Created test model files");

    // Step 2: Upload test models to S3
    println!("Uploading test models to S3...");
    upload_test_models_to_s3(
        &test_bucket,
        &test_prefix,
        &[
            ("model_001.onnx", &model1_path),
            ("model_002.onnx", &model2_path),
        ],
    )
    .await
    .unwrap();

    println!("Successfully uploaded test models");

    // Step 3: Test S3ModelSource and S3ModelDownloader
    println!("\nTesting S3ModelSource...");
    let downloader = S3ModelDownloader::new(s3_uri.clone(), 2).await.unwrap();
    let model_source = S3ModelSource::new(downloader);

    let model_path = model_source.get_latest_model().await.unwrap();

    println!("Downloaded model to: {}", model_path.display());
    assert!(model_path.exists(), "Downloaded model file should exist");
    assert!(
        model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("model_002"),
        "Should download the lexicographically latest model (model_002)"
    );

    // Step 4: Simulate reload by uploading a newer model
    println!("\nSimulating model update...");
    let model3_path = test_dir.join("model_003.onnx");
    fs::write(&model3_path, b"fake model 3 data - newer!")
        .await
        .unwrap();

    upload_test_models_to_s3(
        &test_bucket,
        &test_prefix,
        &[("model_003.onnx", &model3_path)],
    )
    .await
    .unwrap();

    println!("Uploaded model_003.onnx");

    // Wait a moment and check that we can fetch the new model
    tokio::time::sleep(Duration::from_secs(1)).await;

    let latest_model_after_update = model_source.get_latest_model().await.unwrap();

    println!(
        "After update, latest model: {}",
        latest_model_after_update.display()
    );
    assert!(
        latest_model_after_update
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("model_003"),
        "Should now have model_003 as the latest"
    );

    // Cleanup
    println!("\nCleaning up...");
    cleanup_test_models_from_s3(
        &test_bucket,
        &test_prefix,
        &["model_001.onnx", "model_002.onnx", "model_003.onnx"],
    )
    .await
    .unwrap();
    let _ = fs::remove_dir_all(&test_dir).await;

    println!("\n✅ S3 reload integration test passed!");
}

async fn upload_test_models_to_s3(
    bucket: &str,
    prefix: &str,
    models: &[(&str, &std::path::Path)],
) -> Result<(), anyhow::Error> {
    use alpha_blokus::s3::create_s3_client;

    let client = create_s3_client().await;

    for (name, path) in models {
        let data = fs::read(path).await?;
        let key = format!("{}/{}", prefix, name);

        client
            .put_object()
            .bucket(bucket)
            .key(&key)
            .body(data.into())
            .send()
            .await?;

        println!("  Uploaded: {}", key);
    }

    Ok(())
}

async fn cleanup_test_models_from_s3(
    bucket: &str,
    prefix: &str,
    models: &[&str],
) -> Result<(), anyhow::Error> {
    use alpha_blokus::s3::create_s3_client;

    let client = create_s3_client().await;

    for name in models {
        let key = format!("{}/{}", prefix, name);

        match client.delete_object().bucket(bucket).key(&key).send().await {
            Ok(_) => println!("  Deleted: {}", key),
            Err(e) => println!("  Failed to delete {}: {}", key, e),
        }
    }

    Ok(())
}
