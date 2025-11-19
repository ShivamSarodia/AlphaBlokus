use axum::{Json, extract::State};
use serde::Deserialize;

use crate::web::{
    get_game::{GameResponse, build_game_response},
    serve::ApiResult,
    state::AppState,
};

#[derive(Deserialize)]
pub struct AgentMoveRequest {
    agent: String,
}

pub async fn post_agent_move(
    State(app_state): State<AppState>,
    Json(request): Json<AgentMoveRequest>,
) -> ApiResult<Json<GameResponse>> {
    app_state.start_agent_move(&request.agent).await?;

    Ok(Json(build_game_response(&app_state).await))
}
