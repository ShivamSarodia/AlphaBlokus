use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::{GameConfig, LoadableConfig, WebPlayConfig};
use crate::game::move_data::MoveProfile;
use crate::game::{BoardSlice, BoardSlice2D, State as BlokusState};

struct AppState {
    config: &'static WebPlayConfig,
    inner: Arc<Mutex<InnerAppState>>,
}

#[derive(Clone)]
struct InnerAppState {
    blokus_state: BlokusState,
}

impl AppState {
    pub fn new(config: &'static WebPlayConfig) -> Self {
        AppState {
            config,
            inner: Arc::new(Mutex::new(InnerAppState {
                blokus_state: BlokusState::new(&config.game),
            })),
        }
    }
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        AppState {
            config: self.config,
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Serialize)]
struct GameResponse {
    board_size: usize,
    pieces: Vec<PieceResponse>,
    board: Vec<BoardSlice2D>,
    current_player: usize,
}

#[derive(Deserialize)]
struct MoveRequest {
    cells: Vec<CellPayload>,
}

#[derive(Deserialize)]
struct CellPayload {
    row: usize,
    col: usize,
}

#[derive(Serialize)]
struct PieceResponse {
    id: usize,
    squares: usize,
    orientations: Vec<PieceOrientationResponse>,
}

#[derive(Serialize)]
struct PieceOrientationResponse {
    id: usize,
    width: u8,
    height: u8,
    cells: Vec<[u8; 2]>,
    valid: bool,
}

pub fn load_config(path: &std::path::Path) -> Result<&'static WebPlayConfig> {
    let config = WebPlayConfig::from_file(path).context("Failed to load config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;
    Ok(config)
}

pub async fn run(config: &'static WebPlayConfig) -> Result<()> {
    let app_state = AppState::new(config);
    let router = build_router(app_state);

    start_server(router).await
}

fn build_router(app_state: AppState) -> Router {
    Router::new()
        .route("/api/game", get(get_game_state))
        .route("/api/move", post(post_move))
        .with_state(app_state)
}

async fn start_server(router: Router) -> Result<()> {
    let addr: SocketAddr = ([0, 0, 0, 0], 8083).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Failed to bind TCP listener")?;
    println!("Started API server on http://{addr}");

    axum::serve(listener, router)
        .await
        .context("Server exited unexpectedly")
}

fn build_game_response(
    inner_app_state: &InnerAppState,
    config: &'static WebPlayConfig,
) -> GameResponse {
    let pieces = build_piece_response(&config.game, &inner_app_state.blokus_state);
    let board = inner_app_state
        .blokus_state
        .board()
        .slices()
        .iter()
        .map(|slice| slice.to_2d())
        .collect::<Vec<_>>();
    let current_player = inner_app_state.blokus_state.player();
    GameResponse {
        board_size: config.game.board_size,
        pieces,
        board,
        current_player,
    }
}

async fn get_game_state(State(app_state): State<AppState>) -> Json<GameResponse> {
    let inner = app_state.inner.lock().await;

    Json(build_game_response(&inner, app_state.config))
}

async fn post_move(
    State(app_state): State<AppState>,
    Json(request): Json<MoveRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    if request.cells.is_empty() {
        return Err(ApiError::InvalidMove("No cells provided".into()));
    }

    let board_size = app_state.config.game.board_size;
    let move_slice = build_slice_from_cells(&request.cells, board_size)?;

    let inner = app_state.inner.lock().await;
    let move_index = find_move_by_shape(&move_slice, &app_state.config.game)
        .ok_or_else(|| ApiError::UnknownMove("No matching move found for provided cells".into()))?;

    if !inner.blokus_state.is_valid_move(move_index) {
        return Err(ApiError::MoveNotAllowed(
            "Move is not valid in the current state".into(),
        ));
    }

    Ok(Json(build_game_response(&inner, app_state.config)))
}

fn compute_valid_orientations(state: &BlokusState, game_config: &'static GameConfig) -> Vec<bool> {
    let mut validity = vec![false; game_config.num_piece_orientations];

    for move_index in state.valid_moves() {
        let profile = game_config.move_profiles().get(move_index);
        validity[profile.piece_orientation_index] = true;
    }

    validity
}

fn build_piece_response(
    game_config: &'static GameConfig,
    state: &BlokusState,
) -> Vec<PieceResponse> {
    let validity = compute_valid_orientations(state, game_config);

    let mut pieces = (0..game_config.num_pieces)
        .map(|piece_index| PieceResponse {
            id: piece_index,
            squares: 0,
            orientations: Vec::new(),
        })
        .collect::<Vec<_>>();

    let mut orientation_shapes: Vec<Option<OrientationShape>> =
        vec![None; game_config.num_piece_orientations];

    for profile in game_config.move_profiles().iter() {
        let orientation_idx = profile.piece_orientation_index;
        if orientation_shapes[orientation_idx].is_none() {
            orientation_shapes[orientation_idx] = Some(OrientationShape::from_profile(
                profile,
                game_config.board_size,
            ));
        }
    }

    for orientation in orientation_shapes.into_iter().flatten() {
        let piece = &mut pieces[orientation.piece_index];
        if piece.squares == 0 {
            piece.squares = orientation.cells.len();
        }
        piece.orientations.push(orientation.to_response(&validity));
    }

    pieces
}

#[derive(Clone)]
struct OrientationShape {
    id: usize,
    piece_index: usize,
    width: u8,
    height: u8,
    cells: Vec<[u8; 2]>,
}

impl OrientationShape {
    fn from_profile(profile: &MoveProfile, board_size: usize) -> Self {
        let mut min_x = board_size;
        let mut min_y = board_size;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut cells = Vec::new();

        for x in 0..board_size {
            for y in 0..board_size {
                if profile.occupied_cells.get((x, y)) {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    cells.push((x, y));
                }
            }
        }

        if cells.is_empty() {
            return OrientationShape {
                id: profile.piece_orientation_index,
                piece_index: profile.piece_index,
                width: 0,
                height: 0,
                cells: Vec::new(),
            };
        }

        let width = (max_x - min_x + 1) as u8;
        let height = (max_y - min_y + 1) as u8;

        let normalized_cells = cells
            .into_iter()
            .map(|(x, y)| [(x - min_x) as u8, (y - min_y) as u8])
            .collect();

        OrientationShape {
            id: profile.piece_orientation_index,
            piece_index: profile.piece_index,
            width,
            height,
            cells: normalized_cells,
        }
    }

    fn to_response(&self, validity: &[bool]) -> PieceOrientationResponse {
        PieceOrientationResponse {
            id: self.id,
            width: self.width,
            height: self.height,
            cells: self.cells.clone(),
            valid: validity.get(self.id).copied().unwrap_or(false),
        }
    }
}

fn build_slice_from_cells(
    cells: &[CellPayload],
    board_size: usize,
) -> Result<BoardSlice, ApiError> {
    let mut slice = BoardSlice::new(board_size);
    for cell in cells {
        if cell.row >= board_size || cell.col >= board_size {
            return Err(ApiError::InvalidMove(format!(
                "Cell ({}, {}) is out of bounds",
                cell.row, cell.col
            )));
        }
        slice.set((cell.col, cell.row), true);
    }
    Ok(slice)
}

fn find_move_by_shape(slice: &BoardSlice, config: &'static GameConfig) -> Option<usize> {
    for (index, profile) in config.move_profiles().iter().enumerate() {
        if board_slices_equal(slice, &profile.occupied_cells) {
            return Some(index);
        }
    }
    None
}

fn board_slices_equal(a: &BoardSlice, b: &BoardSlice) -> bool {
    if a.size() != b.size() {
        return false;
    }

    let size = a.size();
    for x in 0..size {
        for y in 0..size {
            if a.get((x, y)) != b.get((x, y)) {
                return false;
            }
        }
    }
    true
}

enum ApiError {
    InvalidMove(String),
    UnknownMove(String),
    MoveNotAllowed(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::InvalidMove(message) => {
                (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
            }
            ApiError::UnknownMove(message) => {
                (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
            }
            ApiError::MoveNotAllowed(message) => {
                (StatusCode::CONFLICT, Json(json!({ "error": message }))).into_response()
            }
        }
    }
}
