use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: i32,
    // Number of valid moves.
    pub num_moves: i32,
    // Number of pieces that can be played. (For standard Blokus, this is 21)
    pub num_pieces: i32,
    // Number of (piece, orientation) tuples that produce a unique shape. (For standard Blokus, this is 91)
    pub num_piece_orientations: i32,
    // Path to the file containing the static moves data.
    pub moves_file_path: String,
}

#[derive(Deserialize, Debug)]
pub struct SelfPlayConfig {
    pub game: GameConfig,
}

impl SelfPlayConfig {
    pub fn from_string(string: &str) -> Result<Config> {
        let self_play_config =
            toml::from_str::<SelfPlayConfig>(string).context("Failed to parse config")?;
        Ok(Config::SelfPlay(self_play_config))
    }

    pub fn from_file(path: &PathBuf) -> Result<Config> {
        let contents = std::fs::read_to_string(path).context(format!(
            "Failed to read config from path: {}",
            path.display()
        ))?;
        Self::from_string(&contents)
    }
}

#[derive(Debug)]
pub enum Config {
    SelfPlay(SelfPlayConfig),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_from_string() -> Result<()> {
        let Config::SelfPlay(config) = SelfPlayConfig::from_string(
            r#"
            [game]
            board_size = 20
            num_moves = 30433
            num_pieces = 21
            num_piece_orientations = 91
            moves_file_path = ""
        "#,
        )?;
        assert_eq!(config.game.board_size, 20);
        Ok(())
    }

    // TODO: Test from_file as well.
}
