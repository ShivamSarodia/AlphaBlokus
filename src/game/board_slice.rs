use serde::{Deserialize, Serialize};
use std::fmt;

use crate::game::display::{BoardDisplay, BoardDisplayColor, BoardDisplayLayer, BoardDisplayShape};

// Structure representing an board_size x board_size slice of a board, like
// a single player's pieces on a board.
#[derive(Debug, PartialEq, Eq, Clone, Deserialize, Serialize)]
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

    /// For all cells in `other` that are true, set the corresponding cell in self to true.
    pub fn add(&mut self, other: &Self) {
        for x in 0..self.size {
            for y in 0..self.size {
                if other.get((x, y)) {
                    self.set((x, y), true);
                }
            }
        }
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
        let display = BoardDisplay::new(vec![BoardDisplayLayer {
            color: BoardDisplayColor::Black,
            shape: BoardDisplayShape::Square,
            board_slice: self,
        }]);
        f.write_str(&display.render())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        let mut board_slice = BoardSlice::new(5);
        board_slice.set((0, 0), true);
        println!("{}", board_slice);
    }
}
