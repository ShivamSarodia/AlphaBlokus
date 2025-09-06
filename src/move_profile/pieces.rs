use anyhow::{Result, bail};

#[derive(Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Coord {
    pub x: usize,
    pub y: usize,
    pub board_size: usize,
}

impl Coord {
    pub fn rotate(&mut self, rotation: i32) {
        (self.x, self.y) = match rotation % 4 {
            0 => (self.x, self.y),
            1 => (self.y, self.board_size - 1 - self.x),
            2 => (self.board_size - 1 - self.x, self.board_size - 1 - self.y),
            3 => (self.board_size - 1 - self.y, self.x),
            _ => unreachable!(),
        }
    }

    pub fn flip(&mut self, flip: bool) {
        if flip {
            (self.x, self.y) = (self.x, self.board_size - 1 - self.y)
        }
    }

    pub fn translate(&mut self, v: (i32, i32)) {
        let new_x = (self.x as i32) + v.0;
        let new_y = (self.y as i32) + v.1;

        if new_x < 0
            || new_x >= (self.board_size as i32)
            || new_y < 0
            || new_y >= (self.board_size as i32)
        {
            panic!("Coordinate out of range")
        }

        (self.x, self.y) = (new_x as usize, new_y as usize)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Piece {
    pub coords: Vec<Coord>,
}

impl Piece {
    pub fn rotate(&mut self, rotation: i32) {
        for c in &mut self.coords {
            c.rotate(rotation)
        }
    }

    pub fn flip(&mut self, flip: bool) {
        if flip {
            for c in &mut self.coords {
                c.flip(flip)
            }
        }
    }

    pub fn translate(&mut self, v: (i32, i32)) {
        for c in &mut self.coords {
            c.translate(v)
        }
    }

    pub fn top_right(&self) -> Coord {
        Coord {
            x: self.coords.iter().map(|c| c.x).max().unwrap(),
            y: self.coords.iter().map(|c| c.y).max().unwrap(),
            board_size: self.coords[0].board_size,
        }
    }

    pub fn bottom_left(&self) -> Coord {
        Coord {
            x: self.coords.iter().map(|c| c.x).min().unwrap(),
            y: self.coords.iter().map(|c| c.y).min().unwrap(),
            board_size: self.coords[0].board_size,
        }
    }

    /// Translate the piece so the bottom left coordinate is at (0,0).
    /// Note that (0,0) is not always occupied by the translated piece,
    /// e.g. the + piece doesn't translate to occupy (0,0).
    pub fn recenter(&mut self) {
        let bottom_left = self.bottom_left();
        self.translate((-(bottom_left.x as i32), -(bottom_left.y as i32)))
    }

    pub fn sort(&mut self) {
        self.coords.sort()
    }
}

pub fn piece_list(expected_piece_count: usize, board_size: usize) -> Result<Vec<Piece>> {
    let make_coord = |x, y| Coord { x, y, board_size };

    let pieces = vec![
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(3, 0),
                make_coord(4, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(1, 1),
                make_coord(2, 1),
                make_coord(3, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(0, 2),
                make_coord(1, 0),
                make_coord(2, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(1, 1),
                make_coord(1, 2),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(2, 1),
                make_coord(2, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(2, 1),
                make_coord(3, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(3, 0),
                make_coord(1, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(2, 1),
                make_coord(2, 2),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(1, 2),
                make_coord(2, 2),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(1, 0),
                make_coord(2, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(1, 1),
                make_coord(0, 1),
                make_coord(1, 0),
                make_coord(2, 1),
                make_coord(1, 2),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(1, 2),
                make_coord(2, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(1, 1),
                make_coord(2, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(3, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(0, 1),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(0, 1),
                make_coord(1, 1),
                make_coord(1, 0),
            ],
        },
        Piece {
            coords: vec![
                make_coord(0, 0),
                make_coord(1, 0),
                make_coord(2, 0),
                make_coord(1, 1),
            ],
        },
        Piece {
            coords: vec![make_coord(0, 0), make_coord(1, 0), make_coord(2, 0)],
        },
        Piece {
            coords: vec![make_coord(0, 0), make_coord(1, 0), make_coord(0, 1)],
        },
        Piece {
            coords: vec![make_coord(0, 0), make_coord(1, 0)],
        },
        Piece {
            coords: vec![make_coord(0, 0)],
        },
    ];
    if pieces.len() != expected_piece_count {
        bail!("Number of pieces does not match config: {}", pieces.len())
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_piece() -> Piece {
        return piece_list(21, 20).unwrap()[3].clone();
    }

    #[test]
    fn rotate() {
        let mut piece = get_piece();

        // Rotate by 1
        piece.rotate(1);
        assert_eq!(
            piece,
            Piece {
                coords: vec![
                    Coord {
                        x: 0,
                        y: 19,
                        board_size: 20
                    },
                    Coord {
                        x: 0,
                        y: 18,
                        board_size: 20
                    },
                    Coord {
                        x: 0,
                        y: 17,
                        board_size: 20
                    },
                    Coord {
                        x: 1,
                        y: 18,
                        board_size: 20
                    },
                    Coord {
                        x: 2,
                        y: 18,
                        board_size: 20
                    },
                ]
            }
        );

        // Rotate the rest of the way.
        piece.rotate(3);
        assert_eq!(piece, get_piece())
    }

    #[test]
    fn flip() {
        let mut piece = get_piece();

        piece.flip(false);
        assert_eq!(piece, get_piece());

        piece.flip(true);
        assert_eq!(
            piece,
            Piece {
                coords: vec![
                    Coord {
                        x: 0,
                        y: 19,
                        board_size: 20
                    },
                    Coord {
                        x: 1,
                        y: 19,
                        board_size: 20
                    },
                    Coord {
                        x: 2,
                        y: 19,
                        board_size: 20
                    },
                    Coord {
                        x: 1,
                        y: 18,
                        board_size: 20
                    },
                    Coord {
                        x: 1,
                        y: 17,
                        board_size: 20
                    },
                ]
            }
        );

        piece.flip(true);
        assert_eq!(piece, get_piece());
    }

    #[test]
    fn translate_and_recenter() {
        let mut piece = get_piece();

        piece.translate((1, 1));
        assert_eq!(
            piece,
            Piece {
                coords: vec![
                    Coord {
                        x: 1,
                        y: 1,
                        board_size: 20
                    },
                    Coord {
                        x: 2,
                        y: 1,
                        board_size: 20
                    },
                    Coord {
                        x: 3,
                        y: 1,
                        board_size: 20
                    },
                    Coord {
                        x: 2,
                        y: 2,
                        board_size: 20
                    },
                    Coord {
                        x: 2,
                        y: 3,
                        board_size: 20
                    },
                ]
            }
        );

        piece.recenter();
        assert_eq!(piece, get_piece());
    }

    #[test]
    fn sort() {
        let mut piece = get_piece();

        // Sort the piece and save into another variable.
        piece.sort();
        let sorted_piece = piece.clone();

        // Swap coordinates in the piece so it's no longer sorted.
        (piece.coords[3], piece.coords[0]) = (piece.coords[0].clone(), piece.coords[3].clone());
        assert_ne!(sorted_piece, piece);

        // Re-sort the piece
        piece.sort();
        assert_eq!(sorted_piece, piece);
    }
}
