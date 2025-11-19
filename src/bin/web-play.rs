use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use alpha_blokus::{
    config::{LoadableConfig, WebPlayConfig},
    utils,
    web::run,
};

#[derive(Parser)]
#[command()]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    utils::load_env()?;

    let Cli {
        config: config_path,
    } = Cli::parse();

    let config = WebPlayConfig::from_file(&config_path).context("Failed to load config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;

    run(config).await;
    Ok(())
}
