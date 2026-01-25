use crate::inference;
use crate::inference::client::RequestChannelMessage;
use anyhow::Result;
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
    fn execute(&self, requests: Vec<inference::Request>) -> Result<Vec<inference::Response>>;
}

impl<T: Executor + ?Sized> Executor for Box<T> {
    fn execute(&self, requests: Vec<inference::Request>) -> Result<Vec<inference::Response>> {
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
                        tokio::task::spawn_blocking(move || executor.execute(requests)).await;

                    let responses = match execution_result {
                        Ok(Ok(responses)) => Ok(responses),
                        Ok(Err(err)) => Err(err),
                        Err(err) => Err(anyhow::anyhow!("Executor task failed: {}", err)),
                    };

                    match responses {
                        // If we have responses, send them to the response senders.
                        Ok(responses) => {
                            responses.into_iter().zip(response_senders).for_each(
                                |(response, response_sender)| {
                                    if response_sender.send(Ok(response)).is_err()
                                        && !cancel_token.is_cancelled()
                                    {
                                        tracing::error!("Error sending inference response");
                                    }
                                },
                            );
                        }
                        // Otherwise, send the error to all the waiting senders.
                        Err(err) => {
                            let err = err.context("Error in inference executor");
                            // Include the full error chain so TensorRT failures are visible upstream.
                            let err_msg = format!("{:#}", err);

                            for response_sender in response_senders {
                                if response_sender
                                    .send(Err(anyhow::anyhow!(err_msg.clone())))
                                    .is_err()
                                {
                                    // Receiver dropped; only log if this wasn't due to cancellation
                                    if !cancel_token.is_cancelled() {
                                        tracing::error!("Error sending inference response");
                                    }
                                }
                            }
                        }
                    }
                });

                // Reset the requests and response senders.
                requests = Vec::new();
                response_senders = Vec::new();
            } else if requests.len() > self.batch_size {
                tracing::error!(
                    "Batcher had more requests {} than the batch size {}",
                    requests.len(),
                    self.batch_size
                );
                response_senders.into_iter().for_each(|response_sender| {
                    let err = anyhow::anyhow!("Batcher had more requests than the batch size ");
                    if response_sender.send(Err(err)).is_err() {
                        tracing::error!("Error sending batch size overflow message");
                    }
                });

                // Reset the requests and response senders.
                requests = Vec::new();
                response_senders = Vec::new();
            }
        }
    }
}
