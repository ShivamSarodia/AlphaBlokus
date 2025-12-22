use anyhow::{Context, Result};
use async_trait::async_trait;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

use crate::game::BoardSlice;
use crate::{
    agents::Agent,
    config::{GameConfig, PentobiConfig},
    game::State,
};

pub struct PentobiAgent {
    name: String,
    game_config: &'static GameConfig,
    child: Child,
}

impl PentobiAgent {
    pub fn build(
        pentobi_config: &'static PentobiConfig,
        game_config: &'static GameConfig,
    ) -> Result<Self> {
        let child = Command::new(&pentobi_config.binary_path)
            .arg("--level")
            .arg(pentobi_config.level.to_string())
            .arg("--book")
            .arg(&pentobi_config.opening_book)
            .arg("--noresign")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to spawn Pentobi at {:?}",
                    pentobi_config.binary_path
                )
            })?;

        Ok(Self {
            name: pentobi_config.name.clone(),
            game_config,
            child,
        })
    }

    async fn communicate(&mut self, input: &str) -> Result<String> {
        // Take stdin and stdout (ownership)
        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("Pentobi stdin is not available")?;
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("Pentobi stdout is not available")?;

        stdin.write_all(input.as_bytes()).await?;
        stdin.flush().await?;

        // Read lines until we hit a blank line (`b"\n"`)
        let mut reader = BufReader::new(stdout);
        let mut buf = String::new();
        let mut full_output = String::new();

        loop {
            buf.clear();
            let bytes_read = reader.read_line(&mut buf).await?;

            if bytes_read == 0 {
                // EOF – engine died or closed stdout unexpectedly
                break;
            }

            // Break if we hit a blank line
            if buf == "\n" || buf == "\r\n" {
                break;
            }

            full_output.push_str(&buf);
        }

        // If the output doesn't start with "= ", it's an unexpected response.
        if !full_output.starts_with("= ") {
            anyhow::bail!(
                "Unexpected response from Pentobi: {}",
                full_output.trim_end()
            );
        }

        Ok(full_output.split_at(2).1.to_string())
    }

    fn gtp_coordinates_to_move_index(&self, coordinates: &str) -> Result<usize> {
        let cells = coordinates
            .split(",")
            .map(|coord| {
                // Extract the column
                let (col_str, row_str) = coord.split_at(1);
                let col_char = col_str
                    .chars()
                    .nth(0)
                    .context("Empty coordinate from GTP")?;
                let x = (col_char as u8 - b'a') as usize;

                // Extract the row
                let row: usize = row_str.parse().context("Invalid row number from GTP")?;
                let y = self.game_config.board_size - row;

                Ok([x, y])
            })
            .collect::<Result<Vec<[usize; 2]>>>()?;

        let slice = BoardSlice::from_cells(self.game_config.board_size, &cells);
        let move_profiles = self.game_config.move_profiles()?;
        move_profiles
            .iter()
            .position(|profile| profile.occupied_cells == slice)
            .context("No matching move found for provided cells")
    }

    fn move_index_to_gtp_coordinates(&self, move_index: usize) -> Result<String> {
        let move_profiles = self.game_config.move_profiles()?;
        let slice = &move_profiles.get(move_index).occupied_cells;

        Ok(slice
            .to_cells()
            .iter()
            .map(|(x, y)| {
                let col_char = (b'a' + *x as u8) as char;
                let row = self.game_config.board_size - *y;
                format!("{}{}", col_char, row)
            })
            .collect::<Vec<String>>()
            .join(","))
    }
}

#[async_trait]
impl Agent for PentobiAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> Result<usize> {
        let message = &format!("genmove {}\n", state.player() + 1);
        let result = self.communicate(message.as_str()).await?;
        self.gtp_coordinates_to_move_index(result.trim_end())
    }

    async fn report_move(&mut self, state: &State, move_index: usize) -> Result<()> {
        let coordinates = self.move_index_to_gtp_coordinates(move_index)?;
        let message = format!("play {} {}\n", state.player() + 1, coordinates);
        self.communicate(message.as_str()).await?;
        Ok(())
    }
}
