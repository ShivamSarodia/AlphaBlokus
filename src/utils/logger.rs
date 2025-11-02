/// Initialize a non-blocking logger to stdout.
pub fn init_logger() -> tracing_appender::non_blocking::WorkerGuard {
    // guard must be held to flush logs on shutdown.
    let (non_blocking, guard) = tracing_appender::non_blocking(std::io::stdout());

    tracing_subscriber::fmt().with_writer(non_blocking).init();

    guard
}
