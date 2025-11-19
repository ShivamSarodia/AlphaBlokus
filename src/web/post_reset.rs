use axum::{Json, extract::State};

use crate::web::{
    get_game::{GameResponse, build_game_response},
    serve::ApiResult,
    state::AppState,
};

pub async fn post_reset(State(app_state): State<AppState>) -> ApiResult<Json<GameResponse>> {
    app_state.reset().await;
    Ok(Json(build_game_response(&app_state).await))
}
