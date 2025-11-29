use crate::web::{serve::ApiResult, state::AppState};
use axum::{
    Json,
    extract::{Path, State},
};
use serde::Serialize;

#[derive(Serialize)]
pub struct GetMoveCellsResponse {
    pub cells: Vec<(usize, usize)>,
}

pub async fn get_move_cells(
    State(state): State<AppState>,
    Path(move_index): Path<usize>,
) -> ApiResult<Json<GetMoveCellsResponse>> {
    let move_profile = state.config.game.move_profiles().get(move_index);
    Ok(Json(GetMoveCellsResponse {
        cells: move_profile.occupied_cells.to_cells(),
    }))
}
