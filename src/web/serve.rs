use axum::{
    Json, Router,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;
use std::net::SocketAddr;

use crate::{
    config::WebPlayConfig,
    web::{
        get_game::get_game, post_agent_move::post_agent_move, post_human_move::post_human_move,
        post_reset::post_reset, state::AppState,
    },
};

// Boilerplate to run server.
pub async fn run(config: &'static WebPlayConfig) -> () {
    let app_state = AppState::build(config).await;
    let router = build_router(app_state);

    start_server(router).await
}

fn build_router(app_state: AppState) -> Router {
    Router::new()
        .route("/api/game", get(get_game))
        .route("/api/human_move", post(post_human_move))
        .route("/api/agent_move", post(post_agent_move))
        .route("/api/reset", post(post_reset))
        .with_state(app_state)
}

async fn start_server(router: Router) {
    let addr: SocketAddr = ([0, 0, 0, 0], 8083).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");
    println!("Started API server on http://{addr}");

    axum::serve(listener, router)
        .await
        .expect("Server exited unexpectedly");
}

// Shared types
pub enum ApiError {
    UnknownMove(String),
    MoveNotAllowed(String),
    AgentBusy(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::UnknownMove(message) => (StatusCode::BAD_REQUEST, message),
            ApiError::MoveNotAllowed(message) | ApiError::AgentBusy(message) => {
                (StatusCode::CONFLICT, message)
            }
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

pub(crate) type ApiResult<T> = Result<T, ApiError>;
