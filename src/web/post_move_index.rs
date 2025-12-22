use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use super::state::AppState;
use crate::{
    game::BoardSlice,
    web::serve::{ApiError, ApiResult},
};

#[derive(Deserialize)]
pub struct MoveIndexRequest {
    cells: Vec<[usize; 2]>,
}

#[derive(Serialize)]
pub struct MoveIndexResponse {
    move_index: usize,
}

pub async fn post_move_index(
    State(app_state): State<AppState>,
    Json(request): Json<MoveIndexRequest>,
) -> ApiResult<Json<MoveIndexResponse>> {
    let slice = BoardSlice::from_cells(app_state.config.game.board_size, &request.cells);

    let move_index = app_state
        .config
        .game
        .move_profiles()
        .unwrap()
        .iter()
        .position(|profile| profile.occupied_cells == slice)
        .ok_or_else(|| ApiError::UnknownMove("No matching move found for provided cells".into()))?;

    Ok(Json(MoveIndexResponse { move_index }))
}
