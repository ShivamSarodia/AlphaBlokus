use alpha_blokus::{
    config::{self, GameConfig, NUM_PLAYERS},
    game::Board,
    inference::{Client, DefaultClient, Request},
    utils,
};
use anyhow::{Context, Result};
use clap::Parser;
use config::{BenchmarkInferenceConfig, LoadableConfig};
use rand::Rng;
use std::{path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command()]
#[command(group(
    clap::ArgGroup::new("config_source")
        .required(true)
        .multiple(false)
        .args(&["config", "config_url"])
))]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// URL to config file to load.
    #[arg(long, value_name = "URL")]
    config_url: Option<String>,
}

fn main() -> Result<()> {
    utils::load_env()?;

    let cli = Cli::parse();

    let config = match (&cli.config, cli.config_url.as_deref()) {
        (Some(path), None) => BenchmarkInferenceConfig::from_file(path),
        (None, Some(url)) => BenchmarkInferenceConfig::from_url(url),
        _ => unreachable!("clap enforces exactly one config source"),
    }
    .context("Failed to load config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;

    println!("Starting inference benchmark with config:\n\n{:#?}", config,);

    run_benchmark_inference(config)?;

    Ok(())
}

fn run_benchmark_inference(config: &'static BenchmarkInferenceConfig) -> Result<()> {
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

        println!("Starting benchmark...");

        tokio::spawn({
            let cancel_token = cancel_token.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(config.duration_seconds)).await;
                cancel_token.cancel();
            }
        });

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..config.num_concurrent_threads {
            set.spawn({
                let client = Arc::clone(&client);
                let cancel_token = cancel_token.clone();

                async move {
                    let mut num_evaluations = 0;

                    loop {
                        let request = Request {
                            board: random_board(&config.game),
                            valid_move_indexes: random_valid_move_indexes(&config.game),
                        };
                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                break;
                            }
                            result = client.evaluate(request) => {
                                if let Err(err) = result {
                                    tracing::error!("Inference request failed: {}", err);
                                    continue;
                                }
                                num_evaluations += 1;
                            }
                        }
                    }
                    num_evaluations
                }
            });
        }

        let start_time = std::time::Instant::now();
        let mut total_num_evaluations = 0;
        while let Some(num_evaluations) = set.join_next().await {
            match num_evaluations {
                Ok(count) => total_num_evaluations += count,
                Err(err) => tracing::error!("Benchmark task failed: {}", err),
            }
        }
        println!(
            "Number of evaluations per second: {}",
            total_num_evaluations as f64 / start_time.elapsed().as_secs_f64()
        );
        Ok(())
    })
}

fn random_board(config: &'static GameConfig) -> Board {
    let mut board = Board::new(config);
    for player in 0..NUM_PLAYERS {
        for x in 0..config.board_size {
            for y in 0..config.board_size {
                if rand::rng().random_bool(0.5) {
                    board.slice_mut(player).set((x, y), true);
                }
            }
        }
    }
    board
}

fn random_valid_move_indexes(config: &'static GameConfig) -> Vec<usize> {
    let num_valid_moves = rand::rng().random_range(0..config.num_moves / 30);
    rand::seq::index::sample(&mut rand::rng(), config.num_moves, num_valid_moves).into_vec()
}
