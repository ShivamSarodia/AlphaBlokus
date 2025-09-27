use crate::inference;
use crate::inference::client::RequestChannelMessage;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub trait Executor: Send + Sync + 'static {
    /// Accept a vector of requests and return a vector of response in the same order.
    ///
    /// Each Request includes a vector of valid move indexes. The Response's policy should
    /// be returned in the same order as the specified move indexes.
    ///
    /// The method may be called concurrently from multiple threads, so if locking is needed
    /// it should be done internally. The method is not async and the call can block.
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response>;
}

impl<T: Executor + ?Sized> Executor for Box<T> {
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
        self.as_ref().execute(requests)
    }
}

pub struct Batcher<T: Executor> {
    batch_size: usize,
    executor: Arc<T>,
    request_receiver: mpsc::UnboundedReceiver<RequestChannelMessage>,
    cancel_token: CancellationToken,
}

impl<T: Executor> Batcher<T> {
    pub fn new(
        batch_size: usize,
        executor: T,
        receiver: mpsc::UnboundedReceiver<RequestChannelMessage>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            batch_size,
            executor: Arc::new(executor),
            request_receiver: receiver,
            cancel_token,
        }
    }

    pub async fn run(&mut self) {
        // Responsible for receiving requests up to the batch size, then sending them
        // off to an executor.
        let mut requests = Vec::new();
        let mut response_senders = Vec::new();

        while let Some(request_message) = self.request_receiver.recv().await {
            requests.push(request_message.request);
            response_senders.push(request_message.response_sender);

            if requests.len() == self.batch_size {
                // Spawn a task responsible for executing the requests we've collected.
                // This needs to be in a separate task so that the batcher itself can continue
                // collecting requests.
                let executor = Arc::clone(&self.executor);
                let cancel_token = self.cancel_token.clone();
                tokio::spawn(async move {
                    // move requests and response senders into here
                    let execution_result =
                        tokio::task::spawn_blocking(move || executor.execute(requests))
                            .await
                            .unwrap();

                    execution_result.into_iter().zip(response_senders).for_each(
                        |(response, response_sender)| {
                            response_sender.send(response).unwrap_or_else(|_| {
                                if !cancel_token.is_cancelled() {
                                    println!("Error sending inference response");
                                }
                            });
                        },
                    );
                });

                // Reset the requests and response senders.
                requests = Vec::new();
                response_senders = Vec::new();
            } else if requests.len() > self.batch_size {
                panic!("Batcher should never have more requests than the batch size enqueued");
            }
        }
    }
}
