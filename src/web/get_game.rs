use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::config::{GameConfig, NUM_PLAYERS};
use crate::game::State as BlokusState;
use crate::game::move_data::{MoveProfile, move_index_to_player_pov};
use crate::game::{BoardSlice, BoardSlice2D};
use crate::inference;
use crate::web::state::AppState;

#[derive(Serialize)]
pub struct GameResponse {
    board_size: usize,
    pieces: Vec<PieceResponse>,
    board: Vec<BoardSlice2D>,
    last_moves: Vec<BoardSlice2D>,
    current_player: usize,
    agents: Vec<String>,
    pending_agent: Option<String>,
    game_over: bool,
    scores: Option<Vec<f32>>,
    tile_counts: Vec<usize>,
    network_value: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct PieceResponse {
    id: usize,
    squares: usize,
    available: bool,
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

pub async fn get_game(State(app_state): State<AppState>) -> Json<GameResponse> {
    Json(build_game_response(&app_state).await)
}

pub async fn build_game_response(app_state: &AppState) -> GameResponse {
    let (blokus_state, last_moves, pending_agent) = {
        let session = app_state.session().await;
        (
            session.blokus_states.last().unwrap().clone(),
            build_last_moves(&session.blokus_states, app_state.config.game.board_size),
            session.pending_agent.as_ref().map(|name| name.to_string()),
        )
    };

    let pieces = build_piece_response(&app_state.config.game, &blokus_state);
    let board = blokus_state
        .board()
        .slices()
        .iter()
        .map(|slice| slice.to_2d())
        .collect::<Vec<_>>();
    let current_player = blokus_state.player();
    let game_over = !blokus_state.any_valid_moves();
    let scores = game_over.then(|| blokus_state.result().to_vec());
    let tile_counts = blokus_state
        .board()
        .slices()
        .iter()
        .map(|slice| slice.count())
        .collect::<Vec<_>>();
    let network_value = match app_state.shared_evaluator() {
        Some(client) => {
            evaluate_network_value(&blokus_state, &app_state.config.game, client.as_ref()).await
        }
        None => None,
    };
    GameResponse {
        board_size: app_state.config.game.board_size,
        pieces,
        board,
        last_moves,
        current_player,
        agents: app_state.agent_names(),
        pending_agent,
        game_over,
        scores,
        tile_counts,
        network_value,
    }
}

async fn evaluate_network_value<T: inference::Client + ?Sized>(
    state: &BlokusState,
    game_config: &'static GameConfig,
    client: &T,
) -> Option<Vec<f32>> {
    let player = state.player();
    let move_profiles = game_config.move_profiles().ok()?;
    let valid_move_indexes = state
        .valid_moves()
        .map(|move_index| move_index_to_player_pov(move_index, player, move_profiles))
        .collect::<Vec<_>>();

    let mut value = client
        .evaluate(inference::Request {
            board: state.board().clone_with_player_pov(player as i32),
            valid_move_indexes,
            piece_availability: state.piece_availability_player_pov(player),
        })
        .await
        .ok()?
        .value;
    value.rotate_right(player);
    Some(value.to_vec())
}

fn build_last_moves(states: &[BlokusState], board_size: usize) -> Vec<BoardSlice2D> {
    let mut last_moves: [BoardSlice; NUM_PLAYERS] =
        std::array::from_fn(|_| BoardSlice::new(board_size));

    for window in states.windows(2) {
        let previous_state = &window[0];
        let next_state = &window[1];
        let player = previous_state.player();
        last_moves[player] = diff_player_slice(previous_state, next_state, player);
    }

    last_moves.into_iter().map(|slice| slice.to_2d()).collect()
}

fn diff_player_slice(
    previous_state: &BlokusState,
    next_state: &BlokusState,
    player: usize,
) -> BoardSlice {
    let previous_slice = previous_state.board().slice(player);
    let next_slice = next_state.board().slice(player);
    let mut diff = BoardSlice::new(previous_slice.size());

    for x in 0..previous_slice.size() {
        for y in 0..previous_slice.size() {
            if next_slice.get((x, y)) && !previous_slice.get((x, y)) {
                diff.set((x, y), true);
            }
        }
    }

    diff
}

// Kind of annoying code below that just computes the orientation shapes for each piece
// for the frontend to render for the human player.

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

fn build_piece_response(
    game_config: &'static GameConfig,
    state: &BlokusState,
) -> Vec<PieceResponse> {
    let validity = compute_valid_orientations(state, game_config);
    let player = state.player();

    let mut pieces = (0..game_config.num_pieces)
        .map(|piece_index| PieceResponse {
            id: piece_index,
            squares: 0,
            available: state.is_piece_available(player, piece_index),
            orientations: Vec::new(),
        })
        .collect::<Vec<_>>();

    let mut orientation_shapes: Vec<Option<OrientationShape>> =
        vec![None; game_config.num_piece_orientations];

    for profile in game_config.move_profiles().unwrap().iter() {
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

fn compute_valid_orientations(state: &BlokusState, game_config: &'static GameConfig) -> Vec<bool> {
    let mut validity = vec![false; game_config.num_piece_orientations];

    for move_index in state.valid_moves() {
        let profile = game_config.move_profiles().unwrap().get(move_index);
        validity[profile.piece_orientation_index] = true;
    }

    validity
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use std::sync::Mutex;

    use crate::testing::{create_game_config, create_half_game_config};

    struct MockInferenceClient {
        requests: Mutex<Vec<inference::Request>>,
        response: Option<inference::Response>,
        should_fail: bool,
    }

    impl inference::Client for MockInferenceClient {
        async fn evaluate(
            &self,
            request: inference::Request,
        ) -> anyhow::Result<inference::Response> {
            self.requests.lock().unwrap().push(request);
            if self.should_fail {
                Err(anyhow!("inference failed"))
            } else {
                Ok(self.response.clone().unwrap())
            }
        }
    }

    #[test]
    fn build_last_moves_keeps_latest_move_for_each_player() {
        let game_config = create_half_game_config();
        let mut states = vec![BlokusState::new(&game_config).unwrap()];

        for _ in 0..5 {
            let mut next_state = states.last().unwrap().clone();
            let move_index = next_state
                .first_valid_move()
                .expect("expected another valid move on half board test config");
            next_state.apply_move(move_index).unwrap();
            states.push(next_state);
        }

        let last_moves = build_last_moves(&states, game_config.board_size);
        assert_eq!(last_moves.len(), NUM_PLAYERS);

        let player_zero_last_move = BoardSlice::from_2d(&last_moves[0]);
        let player_one_last_move = BoardSlice::from_2d(&last_moves[1]);
        let player_two_last_move = BoardSlice::from_2d(&last_moves[2]);
        let player_three_last_move = BoardSlice::from_2d(&last_moves[3]);

        assert!(player_zero_last_move.count() > 0);
        assert!(player_one_last_move.count() > 0);
        assert!(player_two_last_move.count() > 0);
        assert!(player_three_last_move.count() > 0);

        let latest_player_zero_turn = &states[4];
        let latest_player_zero_state = &states[5];
        let expected_player_zero_last_move =
            diff_player_slice(latest_player_zero_turn, latest_player_zero_state, 0);

        assert_eq!(player_zero_last_move, expected_player_zero_last_move);
    }

    #[test]
    fn build_piece_response_marks_played_pieces_unavailable() {
        let game_config = create_half_game_config();
        let mut state = BlokusState::new(game_config).unwrap();
        let player = state.player();
        let move_index = state.first_valid_move().unwrap();
        let piece_index = game_config
            .move_profiles()
            .unwrap()
            .get(move_index)
            .piece_index;

        state.apply_move(move_index).unwrap();
        while state.player() != player {
            let move_index = state.first_valid_move().unwrap();
            state.apply_move(move_index).unwrap();
        }

        let pieces = build_piece_response(game_config, &state);

        assert!(!pieces[piece_index].available);
        assert!(!state.is_piece_available(state.player(), piece_index));
    }

    #[test]
    fn build_piece_response_keeps_unplayed_blocked_pieces_available() {
        let game_config = create_half_game_config();
        let mut state = BlokusState::new(game_config).unwrap();

        let target_piece = loop {
            let pieces = build_piece_response(game_config, &state);
            if let Some(piece) = pieces.iter().find(|piece| {
                piece.available
                    && piece
                        .orientations
                        .iter()
                        .all(|orientation| !orientation.valid)
            }) {
                break piece.id;
            }

            let Some(move_index) = state.first_valid_move() else {
                panic!("Expected to find an available but blocked piece before game over");
            };
            state.apply_move(move_index).unwrap();
        };

        let pieces = build_piece_response(game_config, &state);
        let piece = &pieces[target_piece];

        assert!(piece.available);
        assert!(
            piece
                .orientations
                .iter()
                .all(|orientation| !orientation.valid)
        );
    }

    #[tokio::test]
    async fn evaluate_network_value_rotates_back_to_universal_player_order() {
        let game_config = create_game_config();
        let mut state = BlokusState::new(game_config).unwrap();
        state.apply_move(state.first_valid_move().unwrap()).unwrap();

        let client = MockInferenceClient {
            requests: Mutex::new(Vec::new()),
            response: Some(inference::Response {
                value: [0.7, 0.1, 0.1, 0.1],
                policy: vec![],
            }),
            should_fail: false,
        };

        let value = evaluate_network_value(&state, game_config, &client)
            .await
            .unwrap();
        let mut expected = [0.7, 0.1, 0.1, 0.1];
        expected.rotate_right(state.player());

        assert_eq!(value, expected.to_vec());
    }

    #[tokio::test]
    async fn evaluate_network_value_returns_none_when_inference_fails() {
        let game_config = create_game_config();
        let state = BlokusState::new(game_config).unwrap();
        let client = MockInferenceClient {
            requests: Mutex::new(Vec::new()),
            response: None,
            should_fail: true,
        };

        let value = evaluate_network_value(&state, game_config, &client).await;

        assert!(value.is_none());
    }

    #[tokio::test]
    async fn evaluate_network_value_uses_current_player_pov_request() {
        let game_config = create_game_config();
        let mut state = BlokusState::new(game_config).unwrap();
        state.apply_move(state.first_valid_move().unwrap()).unwrap();

        let client = MockInferenceClient {
            requests: Mutex::new(Vec::new()),
            response: Some(inference::Response {
                value: [0.25; NUM_PLAYERS],
                policy: vec![],
            }),
            should_fail: false,
        };

        evaluate_network_value(&state, game_config, &client)
            .await
            .unwrap();
        let requests = client.requests.lock().unwrap();
        let request = requests.last().unwrap();
        let expected_valid_moves = state
            .valid_moves()
            .map(|move_index| {
                move_index_to_player_pov(
                    move_index,
                    state.player(),
                    game_config.move_profiles().unwrap(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(request.valid_move_indexes, expected_valid_moves);
        assert_eq!(
            request.piece_availability,
            state.piece_availability_player_pov(state.player())
        );
        assert_eq!(
            request.board,
            state.board().clone_with_player_pov(state.player() as i32)
        );
    }
}
