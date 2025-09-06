use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct GameConfig {
    // Size of one side of the Blokus board.
    pub board_size: usize,
    // Number of valid moves.
    pub num_moves: usize,
    // Number of pieces that can be played. (For standard Blokus, this is 21)
    pub num_pieces: usize,
    // Number of (piece, orientation) tuples that produce a unique shape. (For standard Blokus, this is 91)
    pub num_piece_orientations: usize,
    // Path to the file containing the static moves data.
    pub moves_file_path: String,
}

impl GameConfig {
    pub fn board_area(&self) -> usize {
        self.board_size * self.board_size
    }
}
