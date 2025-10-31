use std::path::Path;

#[cfg(cuda)]
use crate::inference::TensorRtExecutor;
use crate::{
    config::{ExecutorConfig, GameConfig, InferenceConfig, NUM_PLAYERS},
    game::Board,
    inference::{Executor, OrtExecutor, ReloadExecutor, batcher::Batcher},
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct Request {
    pub board: Board,
    pub valid_move_indexes: Vec<usize>,
}

#[derive(Debug)]
pub struct Response {
    pub value: [f32; NUM_PLAYERS],
    pub policy: Vec<f32>,
}

pub struct RequestChannelMessage {
    pub request: Request,
    pub response_sender: oneshot::Sender<Response>,
}

pub struct DefaultClient {
    request_sender: mpsc::UnboundedSender<RequestChannelMessage>,
}

pub trait Client {
    fn evaluate(&self, request: Request) -> impl std::future::Future<Output = Response> + Send;
}

/// DefaultClient is the only production implementation of the Client trait. We separate the
/// two to simplify testing.
impl DefaultClient {
    /// Builds a client and starts the batcher.
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor to use for inference.
    /// * `batch_size` - The size of batch at which the batcher executes requests.
    /// * `cancel_token` - Signals shutdown to background tasks.
    pub fn build_and_start<T: Executor>(
        executor: T,
        batch_size: usize,
        cancel_token: CancellationToken,
    ) -> Self {
        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let mut batcher = Batcher::new(batch_size, executor, request_receiver, cancel_token);
        tokio::spawn(async move { batcher.run().await });
        Self { request_sender }
    }

    pub async fn from_inference_config(
        inference_config: &InferenceConfig,
        game_config: &'static GameConfig,
        cancel_token: CancellationToken,
    ) -> Self {
        match &inference_config.reload {
            Some(reload_config) => {
                let base_executor_config = inference_config.executor.clone();
                let executor = ReloadExecutor::build(
                    &inference_config.model_path,
                    reload_config,
                    move |path| build_executor(&base_executor_config, game_config, path),
                )
                .await;

                Self::build_and_start(executor, inference_config.batch_size, cancel_token)
            }
            None => {
                let executor = build_executor(
                    &inference_config.executor,
                    game_config,
                    &inference_config.model_path,
                );
                Self::build_and_start(executor, inference_config.batch_size, cancel_token)
            }
        }
    }
}

fn build_executor(
    executor_config: &ExecutorConfig,
    game_config: &'static GameConfig,
    model_path: &Path,
) -> Box<dyn Executor> {
    match executor_config {
        ExecutorConfig::Ort => Box::new(OrtExecutor::build(model_path, game_config).unwrap()),
        #[cfg(cuda)]
        ExecutorConfig::Tensorrt {
            max_batch_size,
            pool_size,
        } => Box::new(
            TensorRtExecutor::build(model_path, game_config, *max_batch_size, *pool_size).unwrap(),
        ),
        #[cfg(not(cuda))]
        ExecutorConfig::Tensorrt { .. } => {
            panic!("TensorRT executor is only available when built with CUDA support.")
        }
    }
}

impl Client for DefaultClient {
    async fn evaluate(&self, request: Request) -> Response {
        // Generate a sender/receiver pair for the oneshot channel
        // used to pass back a response.
        let (response_sender, response_receiver) = oneshot::channel();

        // Send the request on the channel.
        self.request_sender
            .send(RequestChannelMessage {
                request,
                response_sender,
            })
            .unwrap();

        // Wait for a response on the generated channel.
        response_receiver.await.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{config::GameConfig, inference::batcher::Executor, testing};

    use super::*;

    struct MockExecutor {
        pub game_config: &'static GameConfig,
    }
    impl Executor for MockExecutor {
        fn execute(&self, requests: Vec<Request>) -> Vec<Response> {
            if requests.len() != 3 {
                panic!(
                    "MockExecutor should only receive 3 requests, got {}",
                    requests.len()
                );
            }
            requests
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
                .collect()
        }
    }

    #[tokio::test]
    async fn test_evaluate() {
        // Create a game config.
        let game_config = testing::create_game_config();

        let client = Arc::new(DefaultClient::build_and_start(
            MockExecutor {
                game_config: game_config,
            },
            3,
            CancellationToken::new(),
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
                let response = client.evaluate(request).await;
                response
            }
        });
        let handle_1 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[1].clone();
            async move {
                let response = client.evaluate(request).await;
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
                let response = client.evaluate(request).await;
                response
            }
        });
        let handle_3 = tokio::spawn({
            let client = Arc::clone(&client);
            let request = requests[3].clone();
            async move {
                let response = client.evaluate(request).await;
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
}
