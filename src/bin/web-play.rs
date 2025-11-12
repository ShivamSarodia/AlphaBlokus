use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::utils;

#[derive(Parser)]
#[command()]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

fn main() -> Result<()> {
    utils::load_env()?;

    let _ = Cli::parse();

    Ok(())
}
