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
    dotenvy::dotenv()?;

    let cli = Cli::parse();
    tracing::info!("Starting self-play with config: {:#?}", cli.config);

    // Don't drop the guard to flush logs on shutdown.
    let _guard = utils::init_logger();
    utils::init_metrics();

    let config = SelfPlayConfig::from_file(&cli.config)?;
    config.game.load_move_profiles()?;

    run_selfplay(config);

    Ok(())
}
