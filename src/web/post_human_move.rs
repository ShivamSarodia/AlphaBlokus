use axum::{Json, extract::State};
use serde::Deserialize;

use crate::{
    game::BoardSlice,
    web::{
        get_game::{GameResponse, build_game_response},
        serve::{ApiError, ApiResult},
        state::AppState,
    },
};

#[derive(Deserialize, Debug)]
pub struct HumanMoveRequest {
    cells: Vec<[usize; 2]>,
}

pub async fn post_human_move(
    State(app_state): State<AppState>,
    Json(request): Json<HumanMoveRequest>,
) -> ApiResult<Json<GameResponse>> {
    tracing::info!("Received human move request: {:?}", request);

    // Find the move index for the provided cells.
    let slice = BoardSlice::from_cells(app_state.config.game.board_size, &request.cells);
    let move_index = app_state
        .config
        .game
        .move_profiles()
        .iter()
        .position(|profile| profile.occupied_cells == slice)
        .ok_or_else(|| ApiError::UnknownMove("No matching move found for provided cells".into()))?;

    tracing::info!("Identified move index: {:?}", move_index);

    // Report the move to all the agents to update their states.
    app_state.report_human_move(move_index).await?;

    // Apply the move to the current session. Be sure to drop the guard after.
    {
        let mut session = app_state.session().await;
        session.apply_human_move(move_index).await?;
    }

    tracing::info!("Applied human move to session");

    Ok(Json(build_game_response(&app_state).await))
}
