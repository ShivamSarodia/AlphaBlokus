use serde::Deserialize;

/// Configuration for observability (logging and metrics).
#[derive(Deserialize, Debug, Clone)]
pub struct ObservabilityConfig {
    pub logging: LoggingConfig,
    pub metrics: MetricsConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            logging: LoggingConfig::Console,
            metrics: MetricsConfig::None,
        }
    }
}

/// Configuration for logging behavior.
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LoggingConfig {
    /// Log to console (stdout).
    Console,
    /// Log to a file with hourly rotation.
    File { directory: String, filename: String },
}

/// Configuration for metrics.
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MetricsConfig {
    /// Publish Prometheus metrics.
    Prometheus,
    /// No metrics collection.
    None,
}
