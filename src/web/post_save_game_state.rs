use axum::{Json, extract::State};
use tokio::fs;

use crate::{
    game::SerializableState,
    web::{
        get_game::{GameResponse, build_game_response},
        serve::ApiResult,
        state::AppState,
    },
};

pub async fn post_save_game_state(
    State(app_state): State<AppState>,
) -> ApiResult<Json<GameResponse>> {
    let state = {
        let session = app_state.session().await;
        session.blokus_states.last().unwrap().clone()
    };

    let serializable_state = SerializableState::from_state(&state);
    let state_json =
        serde_json::to_string(&serializable_state).expect("Failed to serialize game state");
    fs::write("/tmp/state.json", state_json)
        .await
        .expect("Failed to write game state to file");

    Ok(Json(build_game_response(&app_state).await))
}
