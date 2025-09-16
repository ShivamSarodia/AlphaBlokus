use alpha_blokus::{config, gameplay::run_selfplay};
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use config::{LoadableConfig, SelfPlayConfig};

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("Starting self-play with config:\n\n{:#?}", cli.config);

    let config = SelfPlayConfig::from_file(&cli.config)?;
    config.game.load_move_profiles()?;

    run_selfplay(config);

    Ok(())
}
