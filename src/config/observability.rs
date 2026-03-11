use serde::Deserialize;

fn default_include_player_order_labels() -> bool {
    false
}

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
    Prometheus {
        #[serde(default = "default_include_player_order_labels")]
        include_player_order_labels: bool,
    },
    /// No metrics collection.
    None,
}

impl MetricsConfig {
    pub fn include_player_order_labels(&self) -> bool {
        match self {
            Self::Prometheus {
                include_player_order_labels,
            } => *include_player_order_labels,
            Self::None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_metrics_config_defaults_player_order_labels_to_false() {
        let config: MetricsConfig = toml::from_str(
            r#"
type = "prometheus"
"#,
        )
        .unwrap();

        assert!(!config.include_player_order_labels());
    }

    #[test]
    fn prometheus_metrics_config_parses_player_order_labels_flag() {
        let config: MetricsConfig = toml::from_str(
            r#"
type = "prometheus"
include_player_order_labels = true
"#,
        )
        .unwrap();

        assert!(config.include_player_order_labels());
    }
}
