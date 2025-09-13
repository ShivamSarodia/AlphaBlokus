use crate::inference::{client::RequestChannelMessage, executor::Executor};
use std::sync::Arc;
use tokio::sync::mpsc;

#[allow(dead_code)]
pub struct Batcher<T: Executor> {
    batch_size: usize,
    executor: Arc<T>,
    request_receiver: mpsc::Receiver<RequestChannelMessage>,
}

impl<T: Executor> Batcher<T> {
    pub fn new(
        batch_size: usize,
        executor: T,
        receiver: mpsc::Receiver<RequestChannelMessage>,
    ) -> Self {
        Self {
            batch_size,
            executor: Arc::new(executor),
            request_receiver: receiver,
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
                tokio::spawn(async move {
                    // move requests and response senders into here
                    let execution_result =
                        tokio::task::spawn_blocking(move || executor.execute(requests))
                            .await
                            .unwrap();

                    execution_result.into_iter().zip(response_senders).for_each(
                        |(response, response_sender)| {
                            response_sender.send(response).unwrap();
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
