use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::config::{LoadableConfig, PreprocessMovesConfig};
use alpha_blokus::move_profile;

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = PreprocessMovesConfig::from_file(&cli.config)?;
    println!("Running with config:\n\n{config:#?}");

    let move_profiles = move_profile::generate(&config.game)?;
    move_profile::save(move_profiles, &config.output_file)?;

    Ok(())
}
