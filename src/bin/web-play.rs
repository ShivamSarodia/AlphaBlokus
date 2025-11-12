use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command()]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

fn main() -> Result<()> {
    dotenvy::dotenv().context("Failed to load .env file")?;

    let _ = Cli::parse();

    Ok(())
}
