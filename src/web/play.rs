use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::json;

use crate::{
    config::{GameConfig, LoadableConfig, WebPlayConfig},
    game::BoardSlice,
};

use super::{
    response,
    state::{AgentRunError, AppState},
};

#[derive(Deserialize)]
struct MoveRequest {
    cells: Vec<[u8; 2]>,
}

#[derive(Deserialize)]
struct AgentMoveRequest {
    agent: String,
}

type ApiResult<T> = Result<T, ApiError>;

pub fn load_config(path: &std::path::Path) -> Result<&'static WebPlayConfig> {
    let config = WebPlayConfig::from_file(path).context("Failed to load config")?;
    config
        .game
        .load_move_profiles()
        .context("Failed to load move profiles")?;
    Ok(config)
}

pub async fn run(config: &'static WebPlayConfig) -> Result<()> {
    let app_state = AppState::build(config).await?;
    let router = build_router(app_state);

    start_server(router).await
}

fn build_router(app_state: AppState) -> Router {
    Router::new()
        .route("/api/game", get(get_game_state))
        .route("/api/move", post(post_move))
        .route("/api/agent_move", post(post_agent_move))
        .route("/api/reset", post(post_reset))
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

async fn get_game_state(State(app_state): State<AppState>) -> Json<response::GameResponse> {
    let session = app_state.session().await;
    let agent_names = app_state.agent_names();
    Json(response::build_game_response(
        &session.state,
        app_state.config,
        agent_names,
        session.pending_agent(),
    ))
}

async fn post_move(
    State(app_state): State<AppState>,
    Json(request): Json<MoveRequest>,
) -> ApiResult<Json<response::GameResponse>> {
    let game_config = &app_state.config.game;
    let move_slice = build_slice_from_cells(&request.cells, game_config.board_size)?;
    let move_index = find_move(&move_slice, game_config)
        .ok_or_else(|| ApiError::UnknownMove("No matching move found for provided cells".into()))?;

    let mut session = app_state.session().await;
    if session.has_pending_agent() {
        return Err(ApiError::AgentBusy(
            "An agent is currently selecting a move".into(),
        ));
    }

    let agent_names = app_state.agent_names();
    {
        let game = &mut session.state;
        if !game.is_valid_move(move_index) {
            return Err(ApiError::MoveNotAllowed(
                "Move is not valid in the current state".into(),
            ));
        }
        game.apply_move(move_index);
    }

    Ok(Json(response::build_game_response(
        &session.state,
        app_state.config,
        agent_names,
        session.pending_agent(),
    )))
}

async fn post_agent_move(
    State(app_state): State<AppState>,
    Json(request): Json<AgentMoveRequest>,
) -> ApiResult<StatusCode> {
    match app_state.start_agent_move(&request.agent).await {
        Ok(_) => Ok(StatusCode::ACCEPTED),
        Err(error) => Err(match error {
            AgentRunError::AgentNotFound => ApiError::AgentNotFound(format!(
                "Agent '{}' is not defined in the server config",
                request.agent
            )),
            AgentRunError::AgentBusy => {
                ApiError::AgentBusy("An agent is already selecting a move".into())
            }
            AgentRunError::NoMovesAvailable => {
                ApiError::MoveNotAllowed("No valid moves are available".into())
            }
            AgentRunError::AgentFailed => {
                ApiError::AgentFailed("Failed to communicate with the agent task".into())
            }
        }),
    }
}

async fn post_reset(State(app_state): State<AppState>) -> ApiResult<Json<response::GameResponse>> {
    app_state.reset().await;
    let session = app_state.session().await;
    Ok(Json(response::build_game_response(
        &session.state,
        app_state.config,
        app_state.agent_names(),
        session.pending_agent(),
    )))
}

fn build_slice_from_cells(cells: &[[u8; 2]], board_size: usize) -> ApiResult<BoardSlice> {
    let mut slice = BoardSlice::new(board_size);

    for [row, col] in cells {
        let row = *row as usize;
        let col = *col as usize;
        if row >= board_size || col >= board_size {
            return Err(ApiError::InvalidMove(format!(
                "Cell ({}, {}) is out of bounds",
                row, col
            )));
        }
        slice.set((col, row), true);
    }
    Ok(slice)
}

fn find_move(slice: &BoardSlice, config: &'static GameConfig) -> Option<usize> {
    config
        .move_profiles()
        .iter()
        .position(|profile| slice == &profile.occupied_cells)
}

enum ApiError {
    InvalidMove(String),
    UnknownMove(String),
    MoveNotAllowed(String),
    AgentNotFound(String),
    AgentBusy(String),
    AgentFailed(String),
}

impl ApiError {
    fn into_parts(self) -> (StatusCode, String) {
        match self {
            ApiError::InvalidMove(message) | ApiError::UnknownMove(message) => {
                (StatusCode::BAD_REQUEST, message)
            }
            ApiError::MoveNotAllowed(message) | ApiError::AgentBusy(message) => {
                (StatusCode::CONFLICT, message)
            }
            ApiError::AgentNotFound(message) => (StatusCode::NOT_FOUND, message),
            ApiError::AgentFailed(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = self.into_parts();
        (status, Json(json!({ "error": message }))).into_response()
    }
}
