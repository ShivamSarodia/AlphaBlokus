use crate::config::MetricsConfig;
use anyhow::Result;
use metrics_exporter_prometheus::PrometheusBuilder;

/// Initialize the metrics system based on the provided configuration.
pub fn init_metrics(config: &MetricsConfig, run_name: Option<&str>) -> Result<()> {
    match config {
        MetricsConfig::Prometheus => {
            let mut builder = PrometheusBuilder::new();
            if let Some(run_name) = run_name {
                builder = builder.add_global_label("run_name", run_name);
            }
            builder.install()?;

            metrics::counter!("moves_made_total").absolute(0);
            metrics::counter!("games_started_total").absolute(0);
            metrics::counter!("games_finished_total").absolute(0);
            metrics::counter!("reload_executor.model_not_changed_total").absolute(0);
            metrics::counter!("reload_executor.model_reloaded_total").absolute(0);
            metrics::counter!("game_data_rows_published_total").absolute(0);
        }
        MetricsConfig::None => {
            // No-op: metrics are disabled
        }
    }
    Ok(())
}
