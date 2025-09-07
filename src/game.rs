use crate::config::GameConfig;
use serde::{Deserialize, Serialize};
use std::fmt;

// List of one-value-per-move.
#[derive(Serialize, Deserialize)]
pub struct MovesArray<T> {
    values: Vec<T>,
}

impl<T: Clone> MovesArray<T> {
    pub fn new_with(value: T, game_config: &GameConfig) -> Self {
        MovesArray {
            values: vec![value; game_config.num_moves],
        }
    }
}

impl<T> MovesArray<T> {
    pub fn new_from_vec(values: Vec<T>, game_config: &GameConfig) -> Self {
        if values.len() != game_config.num_moves {
            panic!(
                "Number of values ({}) does not match num_moves ({})",
                values.len(),
                game_config.num_moves
            );
        }
        MovesArray { values }
    }

    pub fn get(&self, index: usize) -> &T {
        &self.values[index]
    }
}

// Structure representing one MovesArray per player.
// pub struct MultiPlayerMovesArray<T>([MovesArray<T>; NUM_PLAYERS]);
//
// impl<T: Clone> MultiPlayerMovesArray<T> {
//     pub fn new_with(value: T, game_config: &GameConfig) -> Self {
//         return MultiPlayerMovesArray(std::array::from_fn(|_| {
//             MovesArray::new_with(value.clone(), &game_config)
//         }));
//     }
// }

// Structure representing an board_size x board_size slice of a board, like
// a single player's pieces on a board.
#[derive(PartialEq, Eq, Clone, Deserialize, Serialize)]
pub struct BoardSlice {
    cells: Vec<bool>,
    size: usize,
}

impl BoardSlice {
    pub fn new(size: usize) -> Self {
        BoardSlice {
            cells: vec![false; size * size],
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn index(&self, at: (usize, usize)) -> usize {
        if at.0 >= self.size {
            panic!("Attempting to get board value outside x bounds: {}", at.0);
        }
        if at.1 >= self.size {
            panic!("Attempting to get board value outside y bounds: {}", at.1);
        }
        at.0 + at.1 * self.size
    }

    pub fn get(&self, at: (usize, usize)) -> bool {
        self.cells[self.index(at)]
    }

    pub fn get_padded(&self, at: (i32, i32)) -> bool {
        if at.0 < 0 || at.1 < 0 || at.0 >= self.size as i32 || at.1 >= self.size as i32 {
            return false;
        }
        self.get((at.0 as usize, at.1 as usize))
    }

    pub fn set(&mut self, at: (usize, usize), value: bool) {
        let i = self.index(at);
        self.cells[i] = value;
    }

    pub fn count(&self) -> usize {
        self.cells.iter().filter(|&&value| value).count()
    }

    /// Rotate the given board by the given number of turns. Each turn
    /// is a 90deg rotation in the direction of play.
    pub fn rotate(&self, turns: i32) -> Self {
        // Get turns % 4 in the range 0 to 3.
        let k = turns.rem_euclid(4);
        let mut rotated_board = self.clone();

        // Short circuit if no turns.
        if k == 0 {
            return rotated_board;
        }

        for x in 0..self.size {
            for y in 0..self.size {
                let src_coordinates = match k {
                    1 => (y, self.size - 1 - x),
                    2 => (self.size - 1 - x, self.size - 1 - y),
                    3 => (self.size - 1 - y, x),
                    _ => unreachable!(),
                };
                rotated_board.set((x, y), self.get(src_coordinates));
            }
        }

        rotated_board
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        for x in 0..self.size {
            for y in 0..self.size {
                if self.get((x, y)) && other.get((x, y)) {
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for BoardSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print column letters header
        f.write_str("\n   ")?;
        for x in 0..self.size {
            let col_letter = (b'A' + x as u8) as char;
            f.write_str(&format!("{:2}", col_letter))?;
        }
        f.write_str("\n")?;

        // Print top border
        f.write_str(" ┌")?;
        f.write_str(&"─".repeat(self.size * 2 + 1))?;
        f.write_str("┐\n")?;

        // Print rows with letters
        for y in 0..self.size {
            // Print row letter
            let row_letter = (b'A' + y as u8) as char;
            f.write_str(&format!("{}│ ", row_letter))?;

            for x in 0..self.size {
                if self.get((x, y)) {
                    f.write_str("■ ")?;
                } else {
                    f.write_str("· ")?;
                }
            }
            f.write_str("│\n")?;
        }

        // Print bottom border
        f.write_str(" └")?;
        f.write_str(&"─".repeat(self.size * 2 + 1))?;
        f.write_str("┘")?;
        Ok(())
    }
}

// Structure representing a board state, with each player
// on a different slice of the provided values.
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
