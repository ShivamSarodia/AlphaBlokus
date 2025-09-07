// Structure representing a board state, with each player
// on a different slice of the provided values. Commented out because
// it's not yet used.
//
// pub struct Board {
//     slices: [BoardSlice; NUM_PLAYERS],
//     size: usize,
// }
//
// impl Board {
//     pub fn new(game_config: &GameConfig) -> Self {
//         return Board {
//             slices: std::array::from_fn(|_| BoardSlice::new(game_config.board_size)),
//             size: game_config.board_size,
//         };
//     }
// }
//
// pub struct State {
//     board: Board,
//     moves_enabled: MultiPlayerMovesArray<bool>,
//     moves_ruled_out: MultiPlayerMovesArray<bool>,
// }
