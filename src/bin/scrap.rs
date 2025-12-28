use alpha_blokus::{
    config::{self, GameConfig},
    game::{State, move_data::move_index_to_player_pov},
    inference::{Client, DefaultClient, Request},
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

        let mut state = State::new(&config.game).unwrap();

        // Apply a move that does NOT point towards player 1 start.
        let valid_move_index = state.valid_moves().nth(1).unwrap();
        println!("move index: {:?}", valid_move_index);
        let profile = config.game.move_profiles().unwrap().get(valid_move_index);
        println!(
            "Piece orientation index: {:?}",
            profile.piece_orientation_index
        );
        println!("Center: {:?}", profile.center);

        state.apply_move(valid_move_index).unwrap();
        let board = state.board().clone();

        println!("Board after Player 0 move: {}", board);

        let pov_board = board.clone_with_player_pov(1);

        // This one points towards player 0 start
        let mut state_2 = state.clone();
        let valid_move_index_point_towards_player_0_start = state_2.valid_moves().next().unwrap();
        state_2
            .apply_move(valid_move_index_point_towards_player_0_start)
            .unwrap();
        println!(
            "Board after Player 1 move index {}: {}",
            valid_move_index_point_towards_player_0_start,
            state_2.board()
        );

        // This one does NOT point towards player 0 start.
        let mut state_3 = state.clone();
        let valid_move_index_not_point_towards_player_0_start =
            state_3.valid_moves().nth(1).unwrap();
        state_3
            .apply_move(valid_move_index_not_point_towards_player_0_start)
            .unwrap();
        println!(
            "Board after Player 1 move index {}: {}",
            valid_move_index_not_point_towards_player_0_start,
            state_3.board()
        );

        let player_pov_valid_move_indexes = state
            .valid_moves()
            .map(|move_index| {
                move_index_to_player_pov(
                    move_index,
                    state.player(),
                    config.game.move_profiles().unwrap(),
                )
            })
            .collect();

        let response = client
            .evaluate(Request {
                board: pov_board,
                valid_move_indexes: player_pov_valid_move_indexes,
            })
            .await
            .unwrap();
        println!("Response: {:?}", response);

        println!(
            "Policy of move index {} towards player 0 start: {:?}",
            valid_move_index_point_towards_player_0_start, response.policy[0],
        );
        println!(
            "Policy of move index {} not towards player 0 start: {:?}",
            valid_move_index_not_point_towards_player_0_start, response.policy[1],
        );

        Ok(())
    })
}
