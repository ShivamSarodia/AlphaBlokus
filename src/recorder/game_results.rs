use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::mpsc;

use crate::config::NUM_PLAYERS;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameResultRow {
    pub agent_names: [String; NUM_PLAYERS],
    pub result: [f32; NUM_PLAYERS],
}

#[derive(Clone)]
pub struct GameResultRecorder {
    sender: mpsc::UnboundedSender<GameResultRow>,
}

impl GameResultRecorder {
    pub fn build_and_start(
        flush_row_count: usize,
        output_path: String,
    ) -> Result<(Self, tokio::task::JoinHandle<()>)> {
        if let Some(parent) = Path::new(&output_path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent directory for {}", output_path)
            })?;
        }

        let channel = mpsc::unbounded_channel();
        let mut receiver = channel.1;
        let background_task = tokio::spawn(async move {
            let mut pending_rows = Vec::new();
            while let Some(row) = receiver.recv().await {
                pending_rows.push(row);
                if pending_rows.len() >= flush_row_count {
                    if let Err(err) = write_rows(&pending_rows, &output_path).await {
                        tracing::error!("Failed to flush game result rows: {}", err);
                    }
                    pending_rows.clear();
                }
            }

            if let Err(err) = write_rows(&pending_rows, &output_path).await {
                tracing::error!("Failed to flush final game result rows: {}", err);
            }
        });

        Ok((Self { sender: channel.0 }, background_task))
    }

    pub fn disabled() -> (Self, tokio::task::JoinHandle<()>) {
        let channel = mpsc::unbounded_channel();
        let mut receiver = channel.1;
        let background_task = tokio::spawn(async move { while receiver.recv().await.is_some() {} });

        (Self { sender: channel.0 }, background_task)
    }

    pub fn push_game_result(&self, row: GameResultRow) -> Result<()> {
        self.sender.send(row)?;
        Ok(())
    }
}

async fn write_rows(rows: &[GameResultRow], output_path: &str) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let file = open_append_file(output_path).await?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        let encoded = serde_json::to_string(row)?;
        writer.write_all(encoded.as_bytes()).await?;
        writer.write_all(b"\n").await?;
    }
    writer.flush().await?;
    Ok(())
}

async fn open_append_file(output_path: &str) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .await
        .with_context(|| format!("Failed to open game result output file {}", output_path))
}

pub async fn read_game_results_from_disk(path: &str) -> Result<Vec<GameResultRow>> {
    let file = File::open(path)
        .await
        .with_context(|| format!("Failed to open game result file {}", path))?;
    let mut reader = BufReader::new(file).lines();
    let mut rows = Vec::new();
    while let Some(line) = reader.next_line().await? {
        rows.push(serde_json::from_str(&line)?);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    fn sample_row(name: &str, result: [f32; NUM_PLAYERS]) -> GameResultRow {
        GameResultRow {
            agent_names: [
                format!("{name}_a"),
                format!("{name}_b"),
                format!("{name}_c"),
                format!("{name}_d"),
            ],
            result,
        }
    }

    #[tokio::test]
    async fn test_game_result_recorder_flushes_after_threshold_and_on_drop() {
        let directory = testing::create_tmp_directory();
        let output_path = format!("{directory}/results.jsonl");
        let (recorder, background_task) =
            GameResultRecorder::build_and_start(3, output_path.clone()).unwrap();

        recorder
            .push_game_result(sample_row("one", [1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        recorder
            .push_game_result(sample_row("two", [0.5, 0.5, 0.0, 0.0]))
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!Path::new(&output_path).exists());

        recorder
            .push_game_result(sample_row("three", [0.25, 0.25, 0.25, 0.25]))
            .unwrap();

        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if Path::new(&output_path).exists() {
                break;
            }
        }

        let rows = read_game_results_from_disk(&output_path).await.unwrap();
        assert_eq!(rows.len(), 3);

        recorder
            .push_game_result(sample_row("four", [1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        drop(recorder);
        background_task.await.unwrap();

        let rows = read_game_results_from_disk(&output_path).await.unwrap();
        assert_eq!(rows.len(), 4);
    }
}
