use alpha_blokus::{config, gameplay::run_selfplay};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::utils;
use config::{LoadableConfig, SelfPlayConfig};

#[derive(Parser)]
#[command()]
#[command(group(
    clap::ArgGroup::new("config_source")
        .required(true)
        .multiple(false)
        .args(&["config", "config_url"])
))]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// URL to config file to load.
    #[arg(long, value_name = "URL")]
    config_url: Option<String>,
    /// Optional run name to attach as a label to all published metrics.
    #[arg(long, value_name = "NAME")]
    run_name: Option<String>,
}

fn main() -> Result<()> {
    utils::load_env()?;

    let cli = Cli::parse();

    let config = match (&cli.config, cli.config_url.as_deref()) {
        (Some(path), None) => SelfPlayConfig::from_file(path)?,
        (None, Some(url)) => SelfPlayConfig::from_url(url)?,
        _ => unreachable!("clap enforces exactly one config source"),
    };

    // Initialize observability based on config
    // Don't drop the guard to flush logs on shutdown.
    let _guard = utils::init_logger(&config.observability.logging);
    utils::init_metrics(&config.observability.metrics, cli.run_name.as_deref())?;

    tracing::info!("Starting self-play with config:\n\n{:#?}", config);

    config.game.load_move_profiles()?;

    run_selfplay(config)?;

    Ok(())
}
