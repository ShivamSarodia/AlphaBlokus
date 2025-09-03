use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

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

    pub fn from_file(path: &Path) -> Result<Config> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from path: {}", path.display()))?;
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
    fn loads_from_file() -> Result<()> {
        let path = Path::new("tests/fixtures/self_play_config.toml");
        let Config::SelfPlay(config) = SelfPlayConfig::from_file(path)?;
        assert_eq!(config.game.board_size, 20);
        Ok(())
    }
}
