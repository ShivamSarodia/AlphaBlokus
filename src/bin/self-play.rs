use alpha_blokus::config;
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
    let config = SelfPlayConfig::from_file(&cli.config)?;

    println!("Running with config:\n\n{config:#?}");
    Ok(())
}
