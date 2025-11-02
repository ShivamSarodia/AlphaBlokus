use crate::config::LoggingConfig;

/// Helper function to initialize the subscriber with a writer.
fn init_with_writer<W>(writer: W) -> tracing_appender::non_blocking::WorkerGuard
where
    W: std::io::Write + Send + 'static,
{
    let (non_blocking, guard) = tracing_appender::non_blocking(writer);
    tracing_subscriber::fmt().with_writer(non_blocking).init();
    guard
}

/// Initialize a non-blocking logger based on the provided configuration.
pub fn init_logger(config: &LoggingConfig) -> tracing_appender::non_blocking::WorkerGuard {
    match config {
        LoggingConfig::Console => init_with_writer(std::io::stdout()),
        LoggingConfig::File {
            directory,
            filename,
        } => init_with_writer(tracing_appender::rolling::hourly(directory, filename)),
    }
}
