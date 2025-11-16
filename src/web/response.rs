use crate::config::{GameConfig, WebPlayConfig};
use crate::game::move_data::MoveProfile;
use crate::game::{BoardSlice2D, State as BlokusState};

use serde::Serialize;

#[derive(Serialize)]
pub struct GameResponse {
    board_size: usize,
    pieces: Vec<PieceResponse>,
    board: Vec<BoardSlice2D>,
    current_player: usize,
    agents: Vec<String>,
    pending_agent: Option<String>,
    game_over: bool,
    scores: Option<Vec<f32>>,
    tile_counts: Vec<usize>,
}

#[derive(Serialize)]
struct PieceResponse {
    id: usize,
    squares: usize,
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

pub fn build_game_response(
    state: &BlokusState,
    config: &'static WebPlayConfig,
    agent_names: Vec<String>,
    pending_agent: Option<&str>,
) -> GameResponse {
    let pieces = build_piece_response(&config.game, state);
    let board = state
        .board()
        .slices()
        .iter()
        .map(|slice| slice.to_2d())
        .collect::<Vec<_>>();
    let current_player = state.player();
    let game_over = !state.any_valid_moves();
    let scores = game_over.then(|| state.result().to_vec());
    let tile_counts = state
        .board()
        .slices()
        .iter()
        .map(|slice| slice.count())
        .collect::<Vec<_>>();
    GameResponse {
        board_size: config.game.board_size,
        pieces,
        board,
        current_player,
        agents: agent_names,
        pending_agent: pending_agent.map(|name| name.to_string()),
        game_over,
        scores,
        tile_counts,
    }
}

fn build_piece_response(
    game_config: &'static GameConfig,
    state: &BlokusState,
) -> Vec<PieceResponse> {
    let validity = compute_valid_orientations(state, game_config);

    let mut pieces = (0..game_config.num_pieces)
        .map(|piece_index| PieceResponse {
            id: piece_index,
            squares: 0,
            orientations: Vec::new(),
        })
        .collect::<Vec<_>>();

    let mut orientation_shapes: Vec<Option<OrientationShape>> =
        vec![None; game_config.num_piece_orientations];

    for profile in game_config.move_profiles().iter() {
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
        let profile = game_config.move_profiles().get(move_index);
        validity[profile.piece_orientation_index] = true;
    }

    validity
}

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
