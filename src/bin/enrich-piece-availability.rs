use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use alpha_blokus::config::{EnrichPieceAvailabilityConfig, LoadableConfig, NUM_PLAYERS};
use alpha_blokus::game::{Board, BoardSlice};
use alpha_blokus::recorder::{
    encode_mcts_data, read_mcts_data_from_disk, upload_encoded_mcts_data_to_s3_file,
};
use alpha_blokus::s3::{S3Uri, create_s3_client};
use alpha_blokus::utils;
use anyhow::{Context, Result, bail};
use clap::Parser;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const LOCAL_GAMES_MIRROR_DIR: &str =
    "/Users/shivamsarodia/Dev/AlphaBlokus/data/s3_mirrors/full_v2/games";

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to config file to load.
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NormalizedShape(Vec<(usize, usize)>);

fn main() -> Result<()> {
    utils::load_env()?;

    let cli = Cli::parse();
    let config = EnrichPieceAvailabilityConfig::from_file(&cli.config)
        .context("Failed to load enrich piece availability config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;

    println!("Running with config:\n\n{config:#?}");

    run(config)
}

fn run(config: &'static EnrichPieceAvailabilityConfig) -> Result<()> {
    let source_s3_uri = S3Uri::new(config.source_s3_path.clone())?;

    let destination_s3_directory = S3Uri::new(config.destination_s3_directory.clone())?;
    if destination_s3_directory.filename.is_some() {
        bail!("destination_s3_directory must be an S3 directory path");
    }

    let shape_to_piece_index = build_piece_shape_lookup(&config.game, config.game.num_pieces)
        .context("Failed to build piece-shape lookup from move profiles")?;
    let orientation_to_piece_index = build_orientation_to_piece_lookup(
        &config.game,
        config.game.num_piece_orientations,
        config.game.num_pieces,
    )
    .context("Failed to build orientation-to-piece lookup from move profiles")?;

    let runtime = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    runtime.block_on(async move {
        let source_file_uris = collect_source_file_uris(&source_s3_uri)
            .await
            .context("Failed to resolve source S3 files")?;
        if source_file_uris.is_empty() {
            bail!("No source files found to process");
        }
        println!("Found {} source file(s) to process", source_file_uris.len());

        let shape_to_piece_index = Arc::new(shape_to_piece_index);
        let orientation_to_piece_index = Arc::new(orientation_to_piece_index);
        let destination_s3_directory = Arc::new(destination_s3_directory);

        let max_parallel = source_file_uris.len().clamp(1, 16);
        let semaphore = Arc::new(Semaphore::new(max_parallel));
        let mut join_set = JoinSet::new();

        for source_file_uri in source_file_uris {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .context("Failed to acquire worker permit")?;
            let shape_to_piece_index = Arc::clone(&shape_to_piece_index);
            let orientation_to_piece_index = Arc::clone(&orientation_to_piece_index);
            let destination_s3_directory = Arc::clone(&destination_s3_directory);

            join_set.spawn(async move {
                let _permit = permit;
                process_one_source_file(
                    source_file_uri,
                    &destination_s3_directory,
                    &shape_to_piece_index,
                    &orientation_to_piece_index,
                    config.game.num_pieces,
                )
                .await
            });
        }

        let mut processed_files = 0usize;
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(Ok(())) => {
                    processed_files += 1;
                    if processed_files.is_multiple_of(100) {
                        println!("Processed {} file(s)...", processed_files);
                    }
                }
                Ok(Err(err)) => return Err(err),
                Err(err) => return Err(anyhow::anyhow!("Worker task failed: {err}")),
            }
        }

        println!("Successfully processed {} file(s)", processed_files);
        Ok(())
    })
}

async fn collect_source_file_uris(source_s3_uri: &S3Uri) -> Result<Vec<S3Uri>> {
    if let Some(filename) = source_s3_uri.filename.as_ref() {
        return Ok(vec![source_s3_uri.with_filename(filename.to_string())?]);
    }

    let client = create_s3_client().await?;
    let prefix = source_s3_uri.key();
    let mut continuation_token: Option<String> = None;
    let mut source_files = Vec::new();

    loop {
        let mut request = client
            .list_objects_v2()
            .bucket(source_s3_uri.bucket.clone());
        if !prefix.is_empty() {
            request = request.prefix(prefix.clone());
        }
        if let Some(token) = continuation_token.as_ref() {
            request = request.continuation_token(token.clone());
        }

        let response = request.send().await?;
        for object in response.contents() {
            let Some(key) = object.key() else {
                continue;
            };
            if !key.ends_with(".bin") {
                continue;
            }
            let uri = S3Uri::new(format!("s3://{}/{}", source_s3_uri.bucket, key))
                .with_context(|| format!("Failed to parse S3 key as URI: {key}"))?;
            source_files.push(uri);
        }

        if response.is_truncated().unwrap_or(false) {
            continuation_token = response.next_continuation_token().map(str::to_string);
        } else {
            break;
        }
    }

    source_files.sort_by_key(S3Uri::key);
    Ok(source_files)
}

async fn process_one_source_file(
    source_file_uri: S3Uri,
    destination_s3_directory: &S3Uri,
    shape_to_piece_index: &HashMap<NormalizedShape, usize>,
    orientation_to_piece_index: &[usize],
    num_pieces: usize,
) -> Result<()> {
    let source_filename = source_file_uri
        .filename
        .as_ref()
        .context("source_s3_path must point to a file, not a directory")?
        .to_string();
    let local_mirror_path = Path::new(LOCAL_GAMES_MIRROR_DIR).join(&source_filename);

    ensure_local_source_file(&source_file_uri, &local_mirror_path)
        .await
        .with_context(|| {
            format!(
                "Failed to ensure local source file for s3://{}/{}",
                source_file_uri.bucket,
                source_file_uri.key()
            )
        })?;

    let mut mcts_data = read_mcts_data_from_disk(&local_mirror_path.to_string_lossy())
        .with_context(|| {
            format!(
                "Failed to read game data from {}",
                local_mirror_path.display()
            )
        })?;

    for (row_index, row) in mcts_data.iter_mut().enumerate() {
        row.piece_availability =
            compute_piece_availability(&row.board, num_pieces, shape_to_piece_index).with_context(
                || format!("Failed to compute piece availability for row {row_index}"),
            )?;
        assert_valid_move_tuples_only_use_available_pieces_for_player_zero(
            row,
            &row.piece_availability,
            orientation_to_piece_index,
            num_pieces,
        )
        .with_context(|| {
            format!(
                "Valid move tuples include unavailable piece for row {row_index} (player={})",
                row.player
            )
        })?;
    }

    let body = tokio::task::spawn_blocking(move || encode_mcts_data(&mcts_data)).await??;
    let destination_file_uri = destination_s3_directory.with_filename(source_filename.clone())?;
    upload_encoded_mcts_data_to_s3_file(body, &destination_file_uri)
        .await
        .with_context(|| {
            format!(
                "Failed to upload enriched file to s3://{}/{}",
                destination_file_uri.bucket,
                destination_file_uri.key()
            )
        })?;

    println!(
        "Uploaded enriched file to s3://{}/{}",
        destination_file_uri.bucket,
        destination_file_uri.key()
    );

    Ok(())
}

async fn ensure_local_source_file(source_s3_uri: &S3Uri, local_path: &Path) -> Result<()> {
    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    if local_path.exists() {
        return Ok(());
    }

    println!(
        "Downloading source game data from s3://{}/{} to {}",
        source_s3_uri.bucket,
        source_s3_uri.key(),
        local_path.display()
    );

    let client = create_s3_client().await?;
    let response = client
        .get_object()
        .bucket(source_s3_uri.bucket.clone())
        .key(source_s3_uri.key())
        .send()
        .await?;

    let mut body = response.body.into_async_read();
    let mut file = tokio::fs::File::create(local_path)
        .await
        .with_context(|| format!("Failed to create local file {}", local_path.display()))?;
    tokio::io::copy(&mut body, &mut file)
        .await
        .with_context(|| {
            format!(
                "Failed to write downloaded data to {}",
                local_path.display()
            )
        })?;
    file.flush().await?;
    Ok(())
}

fn build_piece_shape_lookup(
    game_config: &alpha_blokus::config::GameConfig,
    num_pieces: usize,
) -> Result<HashMap<NormalizedShape, usize>> {
    let move_profiles = game_config.move_profiles()?;

    let mut shape_to_piece_index = HashMap::new();
    let mut seen_piece_index = vec![false; num_pieces];

    for move_profile in move_profiles.iter() {
        let normalized = normalize_cells(move_profile.occupied_cells.to_cells().as_slice())?;

        if let Some(existing_piece_index) =
            shape_to_piece_index.insert(normalized, move_profile.piece_index)
            && existing_piece_index != move_profile.piece_index
        {
            bail!(
                "Conflicting piece index mapping for normalized shape: {} vs {}",
                existing_piece_index,
                move_profile.piece_index
            );
        }

        if move_profile.piece_index >= num_pieces {
            bail!(
                "Move profile piece_index {} is out of range for num_pieces={}",
                move_profile.piece_index,
                num_pieces
            );
        }
        seen_piece_index[move_profile.piece_index] = true;
    }

    for (piece_index, seen) in seen_piece_index.into_iter().enumerate() {
        if !seen {
            bail!(
                "No move profiles found for piece_index {} (num_pieces={})",
                piece_index,
                num_pieces
            );
        }
    }

    Ok(shape_to_piece_index)
}

fn build_orientation_to_piece_lookup(
    game_config: &alpha_blokus::config::GameConfig,
    num_piece_orientations: usize,
    num_pieces: usize,
) -> Result<Vec<usize>> {
    let move_profiles = game_config.move_profiles()?;
    let mut lookup = vec![None::<usize>; num_piece_orientations];

    for move_profile in move_profiles.iter() {
        let orientation = move_profile.piece_orientation_index;
        let piece_index = move_profile.piece_index;
        if orientation >= num_piece_orientations {
            bail!(
                "Move profile piece_orientation_index {} is out of range for num_piece_orientations={}",
                orientation,
                num_piece_orientations
            );
        }
        if piece_index >= num_pieces {
            bail!(
                "Move profile piece_index {} is out of range for num_pieces={}",
                piece_index,
                num_pieces
            );
        }
        if let Some(existing_piece_index) = lookup[orientation]
            && existing_piece_index != piece_index
        {
            bail!(
                "Conflicting piece index for orientation {}: {} vs {}",
                orientation,
                existing_piece_index,
                piece_index
            );
        }
        lookup[orientation] = Some(piece_index);
    }

    lookup
        .into_iter()
        .enumerate()
        .map(|(orientation, maybe_piece)| {
            maybe_piece.ok_or_else(|| {
                anyhow::anyhow!(
                    "No piece index found for orientation {} (num_piece_orientations={})",
                    orientation,
                    num_piece_orientations
                )
            })
        })
        .collect()
}

fn assert_valid_move_tuples_only_use_available_pieces_for_player_zero(
    row: &alpha_blokus::recorder::MCTSData,
    availability: &[Vec<u8>],
    orientation_to_piece_index: &[usize],
    num_pieces: usize,
) -> Result<()> {
    if availability.len() != NUM_PLAYERS {
        bail!(
            "Expected availability to have {} players, got {}",
            NUM_PLAYERS,
            availability.len()
        );
    }
    if availability[0].len() != num_pieces {
        bail!(
            "Expected availability[0] to have num_pieces={} entries, got {}",
            num_pieces,
            availability[0].len()
        );
    }

    for (tuple_index, (piece_orientation_index, _, _)) in row.valid_move_tuples.iter().enumerate() {
        if *piece_orientation_index >= orientation_to_piece_index.len() {
            bail!(
                "valid_move_tuples[{}] has out-of-range piece_orientation_index {}",
                tuple_index,
                piece_orientation_index
            );
        }
        let piece_index = orientation_to_piece_index[*piece_orientation_index];
        if piece_index >= num_pieces {
            bail!(
                "Derived piece_index {} out of range for num_pieces={}",
                piece_index,
                num_pieces
            );
        }
        if availability[0][piece_index] == 0 {
            bail!(
                "valid_move_tuples[{}] uses piece_orientation_index {} -> piece_index {}, but piece is unavailable for player 0",
                tuple_index,
                piece_orientation_index,
                piece_index
            );
        }
    }

    Ok(())
}

fn compute_piece_availability(
    board: &Board,
    num_pieces: usize,
    shape_to_piece_index: &HashMap<NormalizedShape, usize>,
) -> Result<Vec<Vec<u8>>> {
    let mut availability = vec![vec![1u8; num_pieces]; NUM_PLAYERS];

    for (player, board_slice) in board.slices().iter().enumerate() {
        let components = connected_components(board_slice);
        let mut piece_seen_for_player = vec![false; num_pieces];
        for component in components {
            let normalized = normalize_cells(component.as_slice())?;
            let piece_index = *shape_to_piece_index.get(&normalized).ok_or_else(|| {
                anyhow::anyhow!(
                    "Unknown piece shape for player {}. Normalized cells: {:?}",
                    player,
                    normalized.0
                )
            })?;
            if piece_index >= num_pieces {
                bail!(
                    "Derived piece index {} is out of range for num_pieces={}",
                    piece_index,
                    num_pieces
                );
            }
            if piece_seen_for_player[piece_index] {
                bail!(
                    "Piece index {} appears more than once for player {} in one row",
                    piece_index,
                    player
                );
            }
            piece_seen_for_player[piece_index] = true;
            availability[player][piece_index] = 0;
        }
    }

    Ok(availability)
}

fn connected_components(board_slice: &BoardSlice) -> Vec<Vec<(usize, usize)>> {
    let size = board_slice.size();
    let mut visited = vec![false; size * size];
    let mut components = Vec::new();

    let index = |x: usize, y: usize| x + y * size;

    for y in 0..size {
        for x in 0..size {
            if !board_slice.get((x, y)) || visited[index(x, y)] {
                continue;
            }

            let mut component = Vec::new();
            let mut stack = vec![(x, y)];
            visited[index(x, y)] = true;

            while let Some((cx, cy)) = stack.pop() {
                component.push((cx, cy));

                if cx > 0 {
                    let nx = cx - 1;
                    if board_slice.get((nx, cy)) && !visited[index(nx, cy)] {
                        visited[index(nx, cy)] = true;
                        stack.push((nx, cy));
                    }
                }
                if cx + 1 < size {
                    let nx = cx + 1;
                    if board_slice.get((nx, cy)) && !visited[index(nx, cy)] {
                        visited[index(nx, cy)] = true;
                        stack.push((nx, cy));
                    }
                }
                if cy > 0 {
                    let ny = cy - 1;
                    if board_slice.get((cx, ny)) && !visited[index(cx, ny)] {
                        visited[index(cx, ny)] = true;
                        stack.push((cx, ny));
                    }
                }
                if cy + 1 < size {
                    let ny = cy + 1;
                    if board_slice.get((cx, ny)) && !visited[index(cx, ny)] {
                        visited[index(cx, ny)] = true;
                        stack.push((cx, ny));
                    }
                }
            }

            components.push(component);
        }
    }

    components
}

fn normalize_cells(cells: &[(usize, usize)]) -> Result<NormalizedShape> {
    if cells.is_empty() {
        bail!("Cannot normalize an empty shape");
    }

    let min_x = cells.iter().map(|(x, _)| *x).min().unwrap();
    let min_y = cells.iter().map(|(_, y)| *y).min().unwrap();

    let mut normalized = cells
        .iter()
        .map(|(x, y)| (x - min_x, y - min_y))
        .collect::<Vec<(usize, usize)>>();
    normalized.sort_unstable();
    Ok(NormalizedShape(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alpha_blokus::config::GameConfig;

    #[test]
    fn normalize_cells_is_translation_invariant() {
        let a = normalize_cells(&[(10, 11), (11, 11), (11, 12)]).unwrap();
        let b = normalize_cells(&[(0, 0), (1, 0), (1, 1)]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn compute_piece_availability_marks_unavailable_piece_indexes() {
        let game_config = GameConfig {
            board_size: 5,
            move_data_file: PathBuf::from("unused"),
            num_moves: 0,
            num_pieces: 4,
            num_piece_orientations: 0,
            move_data: None,
        };
        let mut board = Board::new(&game_config);
        board.slice_mut(0).set((1, 1), true);
        board.slice_mut(0).set((2, 1), true);
        board.slice_mut(1).set((4, 4), true);

        let mut shape_to_piece = HashMap::new();
        shape_to_piece.insert(normalize_cells(&[(0, 0), (1, 0)]).unwrap(), 2);
        shape_to_piece.insert(normalize_cells(&[(0, 0)]).unwrap(), 1);

        let availability =
            compute_piece_availability(&board, game_config.num_pieces, &shape_to_piece).unwrap();
        assert_eq!(availability[0], vec![1, 1, 0, 1]);
        assert_eq!(availability[1], vec![1, 0, 1, 1]);
        assert_eq!(availability[2], vec![1, 1, 1, 1]);
        assert_eq!(availability[3], vec![1, 1, 1, 1]);
    }

    #[test]
    fn valid_move_tuples_validation_detects_unavailable_piece_for_player_zero() {
        let row = alpha_blokus::recorder::MCTSData {
            player: 0,
            turn: 0,
            game_id: 1,
            board: Board::new(&GameConfig {
                board_size: 5,
                move_data_file: PathBuf::from("unused"),
                num_moves: 0,
                num_pieces: 4,
                num_piece_orientations: 3,
                move_data: None,
            }),
            valid_moves: vec![],
            valid_move_tuples: vec![(1, 0, 0)],
            visit_counts: vec![],
            game_result: [0.0; NUM_PLAYERS],
            q_value: [0.0; NUM_PLAYERS],
            piece_availability: vec![],
        };
        let availability = vec![
            vec![1, 0, 1, 1], // piece 1 unavailable for player 0
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
        ];
        let orientation_to_piece_index = vec![0, 1, 2];

        let result = assert_valid_move_tuples_only_use_available_pieces_for_player_zero(
            &row,
            &availability,
            &orientation_to_piece_index,
            4,
        );
        assert!(result.is_err());
    }
}
