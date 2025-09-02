pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: i32,
    // Number of valid moves.
    pub num_moves: i32,
    // Number of pieces that can be played. (For standard Blokus, this is 21)
    pub num_pieces: i32,
    // Number of (piece, orientation) tuples that produce a unique shape. (For standard Blokus, this is 91)
    pub num_piece_orientations: i32,
    // Path to the file containing the static moves data.
    pub moves_file_path: String,
}

pub struct SelfPlayConfig {
    pub game_config: GameConfig,
}
