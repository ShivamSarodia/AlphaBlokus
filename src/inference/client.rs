use crate::{
    config::NUM_PLAYERS,
    game::{Board, MovesArray},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct Request {
    pub board: Board,
}

#[derive(Debug)]
pub struct Response {
    pub value: [f32; NUM_PLAYERS],
    pub policy: MovesArray<f32>,
}

pub struct RequestChannelMessage {
    pub request: Request,
    pub response_sender: oneshot::Sender<Response>,
}

pub struct Client {
    request_sender: mpsc::Sender<RequestChannelMessage>,
}

impl Client {
    pub fn new(sender: mpsc::Sender<RequestChannelMessage>) -> Self {
        Self {
            request_sender: sender,
        }
    }

    pub async fn evaluate(&self, request: Request) -> Response {
        // Generate a sender/receiver pair for the oneshot channel
        // used to pass back a response.
        let (response_sender, response_receiver) = oneshot::channel();

        // Send the request on the channel.
        self.request_sender
            .send(RequestChannelMessage {
                request,
                response_sender,
            })
            .await
            .unwrap();

        // Wait for a response on the generated channel.
        response_receiver.await.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        config::GameConfig,
        inference::{batcher::Batcher, batcher::Executor},
        testing,
    };

    use super::*;

    struct MockExecutor {
        pub game_config: Arc<GameConfig>,
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
                    policy: MovesArray::new_with(0.0, &self.game_config),
                })
                .collect()
        }
    }

    #[tokio::test]
    async fn test_evaluate() {
        // Create a game config.
        let game_config = testing::create_game_config();

        // Create the executor.
        let executor = MockExecutor {
            game_config: Arc::clone(&game_config),
        };

        // Create a channel for communication.
        let (request_sender, request_receiver) = mpsc::channel(100);

        // Create the batcher and start it.
        let mut batcher = Batcher::new(3, executor, request_receiver);
        tokio::spawn(async move { batcher.run().await });

        // Create a client that publishes to the request sender.
        let client = Arc::new(Client::new(request_sender));

        // Generate four requests.
        let requests = (0..4)
            .map(|i| {
                let mut board = Board::new(&game_config);
                board.slice_mut(i).set((0, 0), true);
                Request { board }
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
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

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
