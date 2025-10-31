use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::fs;
use tokio::sync::RwLock;
use tokio::time::sleep;

use crate::config;
use crate::inference;
use crate::inference::batcher::Executor;
use crate::inference::s3_model_mirror::{S3MirrorConfig, ensure_local_mirror};

/// Generic wrapper that hot-reloads an executor whenever a newer model file
/// appears in a watched directory.
pub struct ReloadExecutor<E>
where
    E: Executor,
{
    executor: Arc<RwLock<E>>,
    current_model_path: Arc<RwLock<PathBuf>>,
}

fn is_s3_path(path: &Path) -> bool {
    path.to_str()
        .map(|value| value.starts_with("s3://"))
        .unwrap_or(false)
}

impl<E> ReloadExecutor<E>
where
    E: Executor,
{
    pub async fn build<P, F>(model_dir: P, reload_config: &config::ReloadConfig, builder: F) -> Self
    where
        P: AsRef<Path>,
        F: Fn(&Path) -> E + Send + Sync + 'static,
    {
        let poll_interval = reload_config.poll_interval;
        let keep_last_models = reload_config.s3_keep_last_models.max(1);

        let mut model_dir = model_dir.as_ref().to_path_buf();

        if is_s3_path(&model_dir) {
            let s3_uri = model_dir.to_string_lossy().to_string();
            model_dir = ensure_local_mirror(
                &s3_uri,
                S3MirrorConfig {
                    poll_interval,
                    keep_last_models,
                },
            )
            .await
            .expect("Failed to initialize S3 model mirror");
        }

        let executor_builder = Arc::new(builder);

        let initial_model_path = Self::latest_model_file_async(&model_dir).await;
        let initial_executor = executor_builder(&initial_model_path);

        let executor = Arc::new(RwLock::new(initial_executor));
        let current_model_path = Arc::new(RwLock::new(initial_model_path));

        tokio::spawn({
            let executor = Arc::clone(&executor);
            let current_model_path = Arc::clone(&current_model_path);
            let executor_builder = Arc::clone(&executor_builder);
            let model_dir = model_dir.clone();

            async move {
                Self::reload_loop(
                    model_dir,
                    poll_interval,
                    executor_builder,
                    executor,
                    current_model_path,
                )
                .await;
            }
        });

        Self {
            executor,
            current_model_path,
        }
    }

    async fn reload_loop<F>(
        model_dir: PathBuf,
        poll_interval: Duration,
        builder: Arc<F>,
        executor: Arc<RwLock<E>>,
        current_model_path: Arc<RwLock<PathBuf>>,
    ) where
        F: Fn(&Path) -> E + Send + Sync + 'static,
    {
        loop {
            sleep(poll_interval).await;

            let latest_model = Self::latest_model_file_async(&model_dir).await;

            let current_path = current_model_path.read().await.clone();

            if latest_model == current_path {
                continue;
            }

            println!("Reloading executor with model {}", latest_model.display());

            {
                let mut guard = executor.write().await;
                *guard = builder(&latest_model);
            }

            {
                let mut guard = current_model_path.write().await;
                *guard = latest_model;
            }
        }
    }

    async fn latest_model_file_async(model_dir: &Path) -> PathBuf {
        let mut entries = fs::read_dir(model_dir).await.unwrap();
        let mut candidates = Vec::new();

        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();

            if path.is_file() && path.extension().unwrap() == "onnx" {
                candidates.push(path);
            }
        }

        candidates.sort_by(|a, b| a.file_name().unwrap().cmp(b.file_name().unwrap()));

        let path = candidates.pop().unwrap();

        path.canonicalize().unwrap_or(path)
    }

    #[allow(dead_code)]
    pub fn current_model_path(&self) -> PathBuf {
        self.current_model_path.blocking_read().clone()
    }
}

impl<E> Executor for ReloadExecutor<E>
where
    E: Executor,
{
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
        self.executor.blocking_read().execute(requests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NUM_PLAYERS, ReloadConfig};
    use crate::game::Board;
    use crate::inference;
    use crate::testing;
    use std::fs;
    use std::path::Path;

    #[derive(Clone)]
    struct MockSubExecutor {
        id: usize,
    }

    impl Executor for MockSubExecutor {
        fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
            requests
                .into_iter()
                .map(|_| inference::Response {
                    value: [self.id as f32; NUM_PLAYERS],
                    policy: vec![],
                })
                .collect()
        }
    }

    fn make_request(game_config: &'static crate::config::GameConfig) -> inference::Request {
        inference::Request {
            board: Board::new(game_config),
            valid_move_indexes: vec![],
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reloads_when_newer_model_appears() {
        let directory = testing::create_tmp_directory();
        let directory_path = Path::new(&directory);

        let first_model_path = directory_path.join("1.onnx");
        fs::write(&first_model_path, "first").unwrap();

        let reload_config = ReloadConfig {
            poll_interval: Duration::from_millis(25),
            s3_keep_last_models: 1,
        };

        let executor = Arc::new(
            ReloadExecutor::build(&directory, &reload_config, move |path| {
                let file_name = path.file_stem().unwrap().to_str().unwrap();
                let id = file_name.parse::<usize>().unwrap();
                MockSubExecutor { id }
            })
            .await,
        );

        let game_config = testing::create_game_config();

        // The call to execute needs to be in a blocking task (like how it's called in
        // the actual batcher).
        let executor_clone = Arc::clone(&executor);
        let initial_response = tokio::task::spawn_blocking(move || {
            executor_clone.execute(vec![make_request(game_config)])
        })
        .await
        .unwrap();
        assert_eq!(initial_response[0].value[0], 1.0);

        tokio::time::sleep(Duration::from_millis(150)).await;
        let second_model_path = directory_path.join("2.onnx");
        fs::write(&second_model_path, "second").unwrap();

        tokio::time::sleep(Duration::from_millis(75)).await;

        let executor_clone = Arc::clone(&executor);
        let updated_response = tokio::task::spawn_blocking(move || {
            executor_clone.execute(vec![make_request(game_config)])
        })
        .await
        .unwrap();
        assert_eq!(updated_response[0].value[0], 2.0);
    }
}
