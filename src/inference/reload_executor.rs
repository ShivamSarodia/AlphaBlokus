use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::time::sleep;

use crate::inference;
use crate::inference::batcher::Executor;
use crate::inference::client::ResponseCache;
use crate::inference::model_source::ModelSource;
use anyhow::{Context, Result};

/// Generic wrapper that hot-reloads an executor whenever a newer model file
/// appears from a model source.
pub struct ReloadExecutor<E>
where
    E: Executor,
{
    executor: Arc<RwLock<E>>,
    current_model_path: Arc<RwLock<PathBuf>>,
}

impl<E> ReloadExecutor<E>
where
    E: Executor,
{
    pub async fn build<F, M>(
        model_source: M,
        poll_interval: Duration,
        builder: F,
        cache: Option<ResponseCache>,
    ) -> Result<Self>
    where
        F: Fn(&Path) -> Result<E> + Send + Sync + 'static,
        M: ModelSource + 'static,
    {
        let executor_builder = Arc::new(builder);
        let model_source = Arc::new(model_source);

        let initial_model_path = model_source
            .get_latest_model()
            .await
            .context("Failed to get initial model")?;
        let initial_executor = executor_builder(&initial_model_path).with_context(|| {
            format!(
                "Failed to build executor for {}",
                initial_model_path.display()
            )
        })?;

        let executor = Arc::new(RwLock::new(initial_executor));
        let current_model_path = Arc::new(RwLock::new(initial_model_path));

        tokio::spawn({
            let executor = Arc::clone(&executor);
            let current_model_path = Arc::clone(&current_model_path);
            let executor_builder = Arc::clone(&executor_builder);
            let model_source = Arc::clone(&model_source);
            let cache = cache.clone();

            async move {
                Self::reload_loop(
                    model_source,
                    poll_interval,
                    executor_builder,
                    executor,
                    current_model_path,
                    cache,
                )
                .await;
            }
        });

        Ok(Self {
            executor,
            current_model_path,
        })
    }

    async fn reload_loop<F, M>(
        model_source: Arc<M>,
        poll_interval: Duration,
        builder: Arc<F>,
        executor: Arc<RwLock<E>>,
        current_model_path: Arc<RwLock<PathBuf>>,
        cache: Option<ResponseCache>,
    ) where
        F: Fn(&Path) -> Result<E> + Send + Sync + 'static,
        M: ModelSource,
    {
        loop {
            sleep(poll_interval).await;

            let latest_model = match model_source.get_latest_model().await {
                Ok(path) => path,
                Err(e) => {
                    tracing::error!("Failed to get latest model: {}", e);
                    continue;
                }
            };

            let current_path = current_model_path.read().await.clone();

            if latest_model == current_path {
                metrics::counter!("reload_executor.model_not_changed_total").increment(1);
                continue;
            }

            tracing::info!("Reloading executor with model {}", latest_model.display());

            {
                let mut guard = executor.write().await;
                let next_executor = match builder(&latest_model) {
                    Ok(executor) => executor,
                    Err(err) => {
                        // If the executor fails to build, log the error but continue the loop.
                        tracing::error!(
                            "Failed to build executor for {}: {}",
                            latest_model.display(),
                            err
                        );
                        continue;
                    }
                };
                *guard = next_executor;
            }

            tracing::info!("Successfully loaded new model: {}", latest_model.display());
            metrics::counter!("reload_executor.model_reloaded_total").increment(1);

            {
                let mut guard = current_model_path.write().await;
                *guard = latest_model;
            }
            if let Some(cache) = &cache {
                sleep(Duration::from_secs(1)).await;
                cache.invalidate_all();
            }
        }
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
    fn execute(&self, requests: Vec<inference::Request>) -> Result<Vec<inference::Response>> {
        self.executor.blocking_read().execute(requests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NUM_PLAYERS;
    use crate::game::Board;
    use crate::inference;
    use crate::inference::LocalModelSource;
    use crate::inference::client::ResponseCache;
    use crate::testing;
    use moka::policy::EvictionPolicy;
    use moka::sync::Cache;
    use nohash_hasher::BuildNoHashHasher;
    use std::fs;
    use std::path::Path;
    #[derive(Clone)]
    struct MockSubExecutor {
        id: usize,
    }

    impl Executor for MockSubExecutor {
        fn execute(&self, requests: Vec<inference::Request>) -> Result<Vec<inference::Response>> {
            let responses = requests
                .into_iter()
                .map(|_| inference::Response {
                    value: [self.id as f32; NUM_PLAYERS],
                    policy: vec![],
                })
                .collect();
            Ok(responses)
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

        let model_source = LocalModelSource::new(&directory);
        let executor = Arc::new(
            ReloadExecutor::build(
                model_source,
                Duration::from_millis(25),
                move |path| {
                    let file_name = path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .ok_or_else(|| anyhow::anyhow!("Invalid model filename"))?;
                    let id = file_name.parse::<usize>().map_err(anyhow::Error::new)?;
                    Ok(MockSubExecutor { id })
                },
                None,
            )
            .await
            .unwrap(),
        );

        let game_config = testing::create_game_config();

        // The call to execute needs to be in a blocking task (like how it's called in
        // the actual batcher).
        let executor_clone = Arc::clone(&executor);
        let initial_response = tokio::task::spawn_blocking(move || {
            executor_clone.execute(vec![make_request(game_config)])
        })
        .await
        .unwrap()
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
        .unwrap()
        .unwrap();
        assert_eq!(updated_response[0].value[0], 2.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reload_clears_cache() {
        let directory = testing::create_tmp_directory();
        let directory_path = Path::new(&directory);

        let first_model_path = directory_path.join("1.onnx");
        fs::write(&first_model_path, "first").unwrap();

        let cache: ResponseCache = Cache::builder()
            .max_capacity(10)
            .eviction_policy(EvictionPolicy::lru())
            .build_with_hasher(BuildNoHashHasher::default());
        cache.insert(
            1,
            inference::Response {
                value: [0.0; NUM_PLAYERS],
                policy: vec![],
            },
        );

        let model_source = LocalModelSource::new(&directory);
        let _executor = ReloadExecutor::build(
            model_source,
            Duration::from_millis(25),
            move |path| {
                let file_name = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid model filename"))?;
                let id = file_name.parse::<usize>().map_err(anyhow::Error::new)?;
                Ok(MockSubExecutor { id })
            },
            Some(cache.clone()),
        )
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(150)).await;
        let second_model_path = directory_path.join("2.onnx");
        fs::write(&second_model_path, "second").unwrap();

        tokio::time::sleep(Duration::from_millis(1250)).await;
        assert!(cache.get(&1).is_none());
    }
}
