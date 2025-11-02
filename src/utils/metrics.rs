use metrics_exporter_prometheus::PrometheusBuilder;

/// Initialize the metrics system.
pub fn init_metrics() {
    PrometheusBuilder::new()
        .install()
        .expect("install prometheus recorder");

    metrics::counter!("moves_made_total").absolute(0);
    metrics::counter!("games_started_total").absolute(0);
    metrics::counter!("games_finished_total").absolute(0);
}
