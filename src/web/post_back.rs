use axum::{Json, extract::State};

use crate::web::{
    get_game::{GameResponse, build_game_response},
    serve::ApiResult,
    state::AppState,
};

pub async fn post_back(State(app_state): State<AppState>) -> ApiResult<Json<GameResponse>> {
    {
        let mut session = app_state.session().await;
        session.blokus_states.pop();
    }
    Ok(Json(build_game_response(&app_state).await))
}
