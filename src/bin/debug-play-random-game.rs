use anyhow::Result;
use rand::prelude::IteratorRandom;

use alpha_blokus::config::GameConfig;
use alpha_blokus::game::{GameStatus, State};

fn main() -> Result<()> {
    let mut game_config = GameConfig {
        board_size: 10,
        num_moves: 6233,
        num_pieces: 21,
        num_piece_orientations: 91,
        move_data: None,
        move_data_file: "static/move_data_size_10.bin".to_string(),
    };
    println!("Running with config:\n\n{game_config:#?}");

    game_config.load_move_profiles()?;

    let mut state = State::new(&game_config);
    loop {
        let valid_moves = state.valid_moves();
        let random_move = valid_moves.choose(&mut rand::rng()).unwrap();
        let game_state = state.apply_move(random_move);
        println!("{}\n", state);
        if game_state == GameStatus::GameOver {
            break;
        }
    }
    Ok(())
}
