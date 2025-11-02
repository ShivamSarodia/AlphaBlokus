use crate::config::MetricsConfig;
use metrics_exporter_prometheus::PrometheusBuilder;

/// Initialize the metrics system based on the provided configuration.
pub fn init_metrics(config: &MetricsConfig) {
    match config {
        MetricsConfig::Prometheus => {
            PrometheusBuilder::new()
                .install()
                .expect("install prometheus recorder");

            metrics::counter!("moves_made_total").absolute(0);
            metrics::counter!("games_started_total").absolute(0);
            metrics::counter!("games_finished_total").absolute(0);
        }
        MetricsConfig::None => {
            // No-op: metrics are disabled
        }
    }
}
