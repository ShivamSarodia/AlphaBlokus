use tokio_util::sync::CancellationToken;

pub fn setup_cancel_token() -> CancellationToken {
    let cancel_token = CancellationToken::new();
    tokio::spawn({
        let cancel_token = cancel_token.clone();
        async move {
            tokio::signal::ctrl_c().await.unwrap();
            tracing::info!("Ctrl-C received, stopping...");
            cancel_token.cancel();
        }
    });
    cancel_token
}
