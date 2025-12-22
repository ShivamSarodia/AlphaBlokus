use alpha_blokus::{config, gameplay::run_selfplay};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::utils;
use config::{LoadableConfig, SelfPlayConfig};

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

fn main() -> Result<()> {
    utils::load_env()?;

    let cli = Cli::parse();

    let config = SelfPlayConfig::from_file(&cli.config)?;

    // Initialize observability based on config
    // Don't drop the guard to flush logs on shutdown.
    let _guard = utils::init_logger(&config.observability.logging);
    utils::init_metrics(&config.observability.metrics)?;

    tracing::info!("Starting self-play with config: {:?}", config);

    config.game.load_move_profiles()?;

    run_selfplay(config)?;

    Ok(())
}
