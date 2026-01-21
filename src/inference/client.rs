use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

#[cfg(cuda)]
use crate::inference::TensorRtExecutor;
use crate::{
    config::{ExecutorConfig, GameConfig, InferenceCacheConfig, InferenceConfig, NUM_PLAYERS},
    game::Board,
    inference::{
        Executor, LocalModelSource, OrtExecutor, RandomExecutor, ReloadExecutor, S3ModelSource,
        batcher::Batcher,
    },
    s3::S3ModelDownloader,
};
use ahash::AHasher;
use anyhow::{Context, Result};
use moka::policy::EvictionPolicy;
use moka::sync::Cache;
use nohash_hasher::BuildNoHashHasher;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Hash)]
pub struct Request {
    pub board: Board,
    pub valid_move_indexes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Response {
    pub value: [f32; NUM_PLAYERS],
    pub policy: Vec<f32>,
}

pub struct RequestChannelMessage {
    pub request: Request,
    pub response_sender: oneshot::Sender<Result<Response>>,
}

pub type ResponseCache = Cache<u64, Response, BuildNoHashHasher<u64>>;

pub struct DefaultClient {
    request_sender: mpsc::UnboundedSender<RequestChannelMessage>,
    cache: Option<ResponseCache>,
}

pub struct PolicyValueClient {
    policy_client: Arc<DefaultClient>,
    value_client: Arc<DefaultClient>,
}

pub trait Client {
    fn evaluate(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Response>> + Send;
}

/// DefaultClient is the only production implementation of the Client trait. We separate the
/// two to simplify testing.
impl DefaultClient {
    pub async fn from_inference_config(
        inference_config: &InferenceConfig,
        game_config: &'static GameConfig,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let cache = Self::build_cache(&inference_config.cache);
        let client = match &inference_config.reload {
            Some(reload_config) => {
                Self::build_with_reload(
                    inference_config,
                    reload_config,
                    game_config,
                    cancel_token,
                    cache,
                )
                .await
            }
            None => {
                Self::build_without_reload(inference_config, game_config, cancel_token, cache).await
            }
        }?;
        Ok(client)
    }

    async fn build_with_reload(
        inference_config: &InferenceConfig,
        reload_config: &crate::config::ReloadConfig,
        game_config: &'static GameConfig,
        cancel_token: CancellationToken,
        cache: Option<ResponseCache>,
    ) -> Result<Self> {
        let base_executor_config = inference_config.executor.clone();
        let poll_interval = Duration::from_secs(reload_config.poll_interval_seconds);

        let executor = if Self::is_s3_path(&inference_config.model_path) {
            let model_source = Self::create_s3_model_source(
                &inference_config.model_path,
                reload_config.s3_cache_size,
            )
            .await?;
            ReloadExecutor::build(
                model_source,
                poll_interval,
                move |path| build_executor(&base_executor_config, game_config, path),
                cache.clone(),
            )
            .await
        } else {
            let model_source = LocalModelSource::new(Path::new(&inference_config.model_path));
            ReloadExecutor::build(
                model_source,
                poll_interval,
                move |path| build_executor(&base_executor_config, game_config, path),
                cache.clone(),
            )
            .await
        };

        Ok(Self::build_with_executor(
            executor?,
            inference_config.batch_size,
            cancel_token,
            cache,
        ))
    }

    async fn build_without_reload(
        inference_config: &InferenceConfig,
        game_config: &'static GameConfig,
        cancel_token: CancellationToken,
        cache: Option<ResponseCache>,
    ) -> Result<Self> {
        let model_path = if Self::is_s3_path(&inference_config.model_path) {
            Self::download_model_from_s3(&inference_config.model_path).await?
        } else {
            PathBuf::from(&inference_config.model_path)
        };

        let executor = build_executor(&inference_config.executor, game_config, &model_path)?;
        Ok(Self::build_with_executor(
            executor,
            inference_config.batch_size,
            cancel_token,
            cache,
        ))
    }

    fn build_cache(cache_config: &InferenceCacheConfig) -> Option<ResponseCache> {
        if cache_config.max_entries == 0 {
            return None;
        }
        Some(
            Cache::builder()
                .max_capacity(cache_config.max_entries as u64)
                .eviction_policy(EvictionPolicy::lru())
                // We hash outside the cache because I don't want Moka to save the key itself
                // in order to save memory.
                .build_with_hasher(BuildNoHashHasher::<u64>::default()),
        )
    }

    fn build_with_executor<T: Executor>(
        executor: T,
        batch_size: usize,
        cancel_token: CancellationToken,
        cache: Option<ResponseCache>,
    ) -> Self {
        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let mut batcher = Batcher::new(batch_size, executor, request_receiver, cancel_token);
        tokio::spawn(async move { batcher.run().await });
        Self {
            request_sender,
            cache,
        }
    }

    fn is_s3_path(path: &str) -> bool {
        path.starts_with("s3://")
    }

    async fn create_s3_model_source(s3_uri: &str, cache_size: usize) -> Result<S3ModelSource> {
        let downloader = S3ModelDownloader::new(s3_uri.to_string(), cache_size)
            .await
            .with_context(|| format!("Failed to create S3ModelDownloader for {}", s3_uri))?;
        Ok(S3ModelSource::new(downloader))
    }

    async fn download_model_from_s3(s3_uri: &str) -> Result<std::path::PathBuf> {
        let downloader = S3ModelDownloader::new(s3_uri.to_string(), 1)
            .await
            .with_context(|| format!("Failed to create S3ModelDownloader for {}", s3_uri))?;
        downloader
            .sync_latest_model()
            .await
            .with_context(|| format!("Failed to download model from S3 {}", s3_uri))
    }
}

impl PolicyValueClient {
    pub fn new(policy_client: Arc<DefaultClient>, value_client: Arc<DefaultClient>) -> Self {
        Self {
            policy_client,
            value_client,
        }
    }
}

impl Client for PolicyValueClient {
    async fn evaluate(&self, request: Request) -> Result<Response> {
        let policy_request = request.clone();
        let value_request = request;
        let (policy_response, value_response) = tokio::try_join!(
            self.policy_client.evaluate(policy_request),
            self.value_client.evaluate(value_request)
        )?;

        Ok(Response {
            value: value_response.value,
            policy: policy_response.policy,
        })
    }
}

fn build_executor(
    executor_config: &ExecutorConfig,
    game_config: &'static GameConfig,
    model_path: &Path,
) -> Result<Box<dyn Executor>> {
    let executor: Box<dyn Executor> = match executor_config {
        ExecutorConfig::Ort { execution_provider } => Box::new(OrtExecutor::build(
            model_path,
            game_config,
            *execution_provider,
        )?),
        ExecutorConfig::Random { sleep_duration_ms } => Box::new(RandomExecutor::build(
            Duration::from_millis(*sleep_duration_ms),
        )),
        #[cfg(cuda)]
        ExecutorConfig::Tensorrt {
            max_batch_size,
            pool_size,
        } => Box::new(TensorRtExecutor::build(
            model_path,
            game_config,
            *max_batch_size,
            *pool_size,
        )?),
        #[cfg(not(cuda))]
        ExecutorConfig::Tensorrt { .. } => {
            panic!("TensorRT executor is only available when built with CUDA support.")
        }
    };
    Ok(executor)
}

impl Client for DefaultClient {
    async fn evaluate(&self, request: Request) -> Result<Response> {
        if let Some(cache) = &self.cache {
            let cache_key = self.cache_key(&request);
            if let Some(response) = cache.get(&cache_key) {
                metrics::counter!("inference_cache_hit_total").increment(1);
                return Ok(response);
            }
            metrics::counter!("inference_cache_miss_total").increment(1);

            let response = self.send_request(request).await?;
            cache.insert(cache_key, response.clone());
            return Ok(response);
        }

        self.send_request(request).await
    }
}

impl DefaultClient {
    fn cache_key(&self, request: &Request) -> u64 {
        let mut hasher = AHasher::default();
        request.hash(&mut hasher);
        hasher.finish()
    }

    async fn send_request(&self, request: Request) -> Result<Response> {
        // Generate a sender/receiver pair for the oneshot channel
        // used to pass back a response.
        let (response_sender, response_receiver) = oneshot::channel();

        // Send the request on the channel.
        self.request_sender
            .send(RequestChannelMessage {
                request,
                response_sender,
            })
            .context("Failed to send inference request")?;

        // Wait for a response on the generated channel.
        response_receiver
            .await
            .context("Inference response channel closed")?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::{config::GameConfig, inference::batcher::Executor, testing};

    use super::*;

    struct MockExecutor {
        pub game_config: &'static GameConfig,
    }
    impl Executor for MockExecutor {
        fn execute(&self, requests: Vec<Request>) -> Result<Vec<Response>> {
            if requests.len() != 3 {
                panic!(
                    "MockExecutor should only receive 3 requests, got {}",
                    requests.len()
                );
            }
            let responses = requests
                .into_iter()
                .map(|request| Response {
                    value: [
                        request.board.slice(0).count() as f32,
                        request.board.slice(1).count() as f32,
                        request.board.slice(2).count() as f32,
                        request.board.slice(3).count() as f32,
                    ],
                    policy: vec![1.0; self.game_config.num_moves],
                })
                .collect();
            Ok(responses)
        }
    }

    #[tokio::test]
    async fn test_evaluate() {
        // Create a game config.
        let game_config = testing::create_game_config();

        let client = Arc::new(DefaultClient::build_with_executor(
            MockExecutor {
                game_config: game_config,
            },
            3,
            CancellationToken::new(),
            DefaultClient::build_cache(&crate::config::InferenceCacheConfig::default()),
        ));

        // Generate four requests.
        let requests = (0..4)
            .map(|i| {
                let mut board = Board::new(&game_config);
                board.slice_mut(i).set((0, 0), true);
                Request {
                    board,
                    valid_move_indexes: vec![i],
                }
            })
            .collect::<Vec<_>>();

        let handle_0 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[0].clone();
            async move {
                let response = client.evaluate(request).await.unwrap();
                response
            }
        });
        let handle_1 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[1].clone();
            async move {
                let response = client.evaluate(request).await.unwrap();
                response
            }
        });

        // Sleep a bit to let any processing occur.
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Evaluations 0 and 1 should not be finished yet.
        assert!(!handle_0.is_finished());
        assert!(!handle_1.is_finished());

        // Now, evaluate 2 and 3.
        let handle_2 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[2].clone();
            async move {
                let response = client.evaluate(request).await.unwrap();
                response
            }
        });
        let handle_3 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[3].clone();
            async move {
                let response = client.evaluate(request).await.unwrap();
                response
            }
        });

        // Sleep a bit more to let any processing occur.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Evaluation 0, 1, and 2 should be finished now.
        assert!(handle_0.is_finished());
        assert!(handle_1.is_finished());
        assert!(handle_2.is_finished());
        assert!(handle_0.await.unwrap().value == [1.0, 0.0, 0.0, 0.0]);
        assert!(handle_1.await.unwrap().value == [0.0, 1.0, 0.0, 0.0]);
        assert!(handle_2.await.unwrap().value == [0.0, 0.0, 1.0, 0.0]);

        // Evaluation 3 should still be waiting, because it didn't make it
        // into the batch.
        assert!(!handle_3.is_finished());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cache_hits_return_cached_response() {
        let game_config = testing::create_game_config();
        let call_count = Arc::new(AtomicUsize::new(0));

        struct CountingExecutor {
            call_count: Arc<AtomicUsize>,
        }

        impl Executor for CountingExecutor {
            fn execute(&self, requests: Vec<Request>) -> Result<Vec<Response>> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(requests
                    .into_iter()
                    .map(|request| Response {
                        value: [0.25; NUM_PLAYERS],
                        policy: vec![0.0; request.valid_move_indexes.len()],
                    })
                    .collect())
            }
        }

        let client = DefaultClient::build_with_executor(
            CountingExecutor {
                call_count: Arc::clone(&call_count),
            },
            1,
            CancellationToken::new(),
            DefaultClient::build_cache(&crate::config::InferenceCacheConfig { max_entries: 10 }),
        );

        let request = Request {
            board: Board::new(&game_config),
            valid_move_indexes: vec![1, 2, 3],
        };

        let first = client.evaluate(request.clone()).await.unwrap();
        let second = client.evaluate(request).await.unwrap();

        assert_eq!(first.value, [0.25; NUM_PLAYERS]);
        assert_eq!(second.value, [0.25; NUM_PLAYERS]);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
