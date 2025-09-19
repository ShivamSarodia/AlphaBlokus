use anyhow::Result;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use crate::{config::NUM_PLAYERS, game::Board};
use chrono::prelude::*;
use chrono_tz::US::Pacific;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use zstd::{Decoder, Encoder};

#[derive(Debug, Serialize, Deserialize)]
pub struct MCTSData {
    /// The player to move.
    pub player: usize,
    /// The current turn in the game.
    pub turn: u16,
    /// A unique identifier for this game.
    pub game_id: u64,
    /// The board, from the perspective of the player to move.
    pub board: Board,
    /// The valid move indices, from the perspective of the player to move.
    pub valid_moves: Vec<usize>,
    /// The same valid moves as above, but represented as
    /// (piece_orientation_index, center_x, center_y) for training purposes.
    pub valid_move_tuples: Vec<(usize, usize, usize)>,
    /// The visit counts of each child, in the same order as above.
    pub visit_counts: Vec<u32>,
    /// The result of this game, from the perspective of the player to move.
    /// This field will be 0.0 for a game that is in progress, and completed
    /// before writing when the game result is known.
    pub game_result: [f32; NUM_PLAYERS],
}

#[derive(Clone)]
pub struct Recorder {
    sender: mpsc::UnboundedSender<Vec<MCTSData>>,
}

impl Recorder {
    pub fn build_and_start(flush_row_count: usize, output_directory: String) -> Self {
        // To start, create the output directory if it doesn't exist.
        std::fs::create_dir_all(&output_directory).unwrap();

        // Start a consumer thread that flushes data to disk when there's enough data.
        let channel = mpsc::unbounded_channel();
        let mut receiver = channel.1;
        tokio::spawn(async move {
            let mut unflushed_mcts_data = Vec::new();
            while let Some(mcts_data_vec) = receiver.recv().await {
                unflushed_mcts_data.extend(mcts_data_vec);

                // When there's enough data, flush it to disk.
                if unflushed_mcts_data.len() >= flush_row_count {
                    // Flush the data.
                    let output_directory = output_directory.clone();
                    tokio::task::spawn_blocking(move || {
                        write_mcts_data_to_disk(unflushed_mcts_data, &output_directory).unwrap();
                    });
                    unflushed_mcts_data = Vec::new();
                }
            }

            // When the channel is closed, flush the remaining data one last time.
            tokio::task::spawn_blocking(move || {
                write_mcts_data_to_disk(unflushed_mcts_data, &output_directory).unwrap();
            });
        });

        // Return the recorder which contains the sender for pushing new data.
        Recorder { sender: channel.0 }
    }

    pub fn push_mcts_data(&self, mcts_data: Vec<MCTSData>) {
        self.sender.send(mcts_data).unwrap();
    }
}

/// Generate a filename for the MCTS data file based on the current time, a random number
/// to prevent collisions, and the number of rows of MCTS data.
fn generate_filename(num_rows: usize) -> String {
    // Get current UTC time
    let now_utc = Utc::now();

    // Convert to Pacific Time
    let now_pacific = now_utc.with_timezone(&Pacific);

    // Generate a random 8-digit number (leading zeros if needed)
    let rand_num: u32 = rand::rng().random_range(0..100_000_000);
    let rand_str = format!("{:08}", rand_num);

    // Build filename-safe string
    format!(
        "{}-{}_{num_rows}.bin",
        now_pacific.format("%Y-%m-%d_%H-%M-%S"),
        rand_str
    )
}

fn write_mcts_data_to_disk(mcts_data: Vec<MCTSData>, output_directory: &str) -> Result<()> {
    let filename = output_directory.to_string() + "/" + &generate_filename(mcts_data.len());

    let file = File::create(filename)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    rmp_serde::encode::write(&mut enc, &mcts_data)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    Ok(())
}

pub fn read_mcts_data_from_disk(filename: &str) -> Result<Vec<MCTSData>> {
    let file = File::open(filename)?;
    let buf = BufReader::new(file);
    let mut dec = Decoder::new(buf)?;
    let mcts_data: Vec<MCTSData> = rmp_serde::decode::from_read(&mut dec)?;
    Ok(mcts_data)
}

#[cfg(test)]
mod tests {
    use crate::testing;

    use super::*;

    fn create_mcts_data(game_id: u64) -> MCTSData {
        MCTSData {
            player: 0,
            turn: 0,
            game_id: game_id,
            board: Board::new(&testing::create_game_config()),
            valid_moves: vec![0, 1, 2, 3, 4],
            valid_move_tuples: vec![(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            visit_counts: vec![0, 0, 0, 0, 0],
            game_result: [0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn test_write_mcts_data_to_disk() {
        // Write some MCTS data to disk and confirm it ~works.
        let mcts_data = vec![create_mcts_data(0)];
        let directory = testing::create_tmp_directory();
        write_mcts_data_to_disk(mcts_data, &directory).unwrap();

        // Ensure there's a file written.
        let mut files = std::fs::read_dir(&directory).unwrap();
        let file = files.next().unwrap().unwrap();

        assert!(file.path().to_string_lossy().ends_with("_1.bin"));
        assert!(file.metadata().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn test_full_recorder() {
        let directory = testing::create_tmp_directory();
        let recorder = Recorder::build_and_start(3, directory.clone());

        let data_1 = create_mcts_data(1);
        let data_2 = create_mcts_data(2);
        let data_3 = create_mcts_data(3);
        let data_4 = create_mcts_data(4);

        // Push 1 and 2. At this point, no data should be written to disk.
        recorder.push_mcts_data(vec![data_1, data_2]);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        assert_eq!(0, std::fs::read_dir(&directory).unwrap().count());

        // Push 3 and 4.
        recorder.push_mcts_data(vec![data_3, data_4]);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The file should be written to disk with four rows of data.
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        let file = std::fs::read_dir(&directory)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert!(file.path().to_string_lossy().ends_with("_4.bin"));
        assert!(file.metadata().unwrap().len() > 0);
    }
}
