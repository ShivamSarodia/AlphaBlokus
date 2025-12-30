use alpha_blokus::{
    config::{self, GameConfig},
    inference::{Client, DefaultClient, Request},
    recorder::read_mcts_data_from_disk,
    utils,
};
use anyhow::{Context, Result};
use clap::Parser;
use config::{InferenceConfig, LoadableConfig};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

#[derive(Deserialize, Debug)]
struct ScrapConfig {
    game: GameConfig,
    inference: InferenceConfig,
}

impl LoadableConfig for ScrapConfig {}

fn main() -> Result<()> {
    utils::load_env()?;

    let cli = Cli::parse();

    let config = ScrapConfig::from_file(&cli.config).context("Failed to load config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;

    println!(
        "Starting compare executor results with config:\n\n{:#?}",
        config,
    );

    run(config)?;

    Ok(())
}

fn run(config: &'static ScrapConfig) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(async move {
        let cancel_token = utils::setup_cancel_token();

        let client = Arc::new(
            DefaultClient::from_inference_config(
                &config.inference,
                &config.game,
                cancel_token.clone(),
            )
            .await
            .context("Failed to create inference client")?,
        );

        let mcts_data_path = "/Users/shivamsarodia/Dev/AlphaBlokus/data/s3_mirrors/full/games/2025-12-14_14-56-12-305253677-44479286_10006.bin";
        let mcts_data = read_mcts_data_from_disk(mcts_data_path).context("Failed to read MCTS data from disk")?;
        let first_mcts_data = &mcts_data[0];

        println!("Board sum: {}", first_mcts_data.board.slices().iter().map(|slice| slice.count()).sum::<usize>());

        let response = client
            .evaluate(Request {
                board: first_mcts_data.board.clone(),
                valid_move_indexes: first_mcts_data.valid_moves.clone(),
            })
            .await
            .unwrap();
        println!("Response: {:?}", response);

        response.policy.iter().enumerate().for_each(|(i, response_policy)| {
            let children_visit_count = first_mcts_data.visit_counts[i];
            println!("({}\t) {}\t{}", i, response_policy, children_visit_count);
        });
        Ok(())
    })
}
