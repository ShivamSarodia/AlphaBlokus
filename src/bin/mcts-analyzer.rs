use alpha_blokus::{config, mcts_analyzer::run};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::utils;
use config::{LoadableConfig, MCTSAnalyzerConfig};

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

    let config = MCTSAnalyzerConfig::from_file(&cli.config)?;

    tracing::info!("Starting MCTS analyzer with config: {:?}", config);

    config.game.load_move_profiles()?;

    run(config);

    Ok(())
}
