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
    coords: Vec<Coord>,
    center: Coord,
}

impl Piece {
    pub fn new(mut coords: Vec<Coord>) -> Self {
        // We select the best coordinate to represent the piece's "center".
        // For each dimension, we try to average the max and min coordinates. If
        // there's an integer result, we just use that. If not, we prefer a coordinate
        // which is occupied by this piece.

        // Start by sorting the coordinates for consistency and center-picking process.
        coords.sort();

        // Find the possible options for the center. E.g. for a piece that is contained in
        // a 3x3 square, there's one unambiguous center. But for a piece that is contained in
        // a 2x2 square, there's four possible centers.
        let max_x = coords.iter().map(|c| c.x).max().unwrap();
        let max_y = coords.iter().map(|c| c.y).max().unwrap();
        let min_x = coords.iter().map(|c| c.x).min().unwrap();
        let min_y = coords.iter().map(|c| c.y).min().unwrap();

        let avg_x = (max_x + min_x) / 2;
        let center_x_options = if (max_x + min_x) % 2 == 0 {
            vec![avg_x]
        } else {
            vec![avg_x, avg_x + 1]
        };

        let avg_y = (max_y + min_y) / 2;
        let center_y_options = if (max_y + min_y) % 2 == 0 {
            vec![avg_y]
        } else {
            vec![avg_y, avg_y + 1]
        };

        // Default to the first option we have.
        let mut center = Coord {
            x: center_x_options[0],
            y: center_y_options[0],
            board_size: coords[0].board_size,
        };

        for coord in coords.iter() {
            // If one of the piece coordinates is a center option, use that instead.
            if center_x_options.contains(&coord.x) && center_y_options.contains(&coord.y) {
                center = coord.clone();
                break;
            }

            // If we never break, we'll fall back to the sensible default above.
        }

        Self { coords, center }
    }

    pub fn coords(&self) -> &Vec<Coord> {
        &self.coords
    }

    pub fn center(&self) -> &Coord {
        &self.center
    }

    pub fn rotate(&mut self, rotation: i32) {
        for c in &mut self.coords {
            c.rotate(rotation)
        }
        self.center.rotate(rotation);
        self.sort();
    }

    pub fn flip(&mut self, flip: bool) {
        if flip {
            for c in &mut self.coords {
                c.flip(flip);
            }
            self.center.flip(flip);
        }
        self.sort();
    }

    pub fn translate(&mut self, v: (i32, i32)) {
        for c in &mut self.coords {
            c.translate(v)
        }
        self.center.translate(v);
        self.sort();
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
        self.translate((-(bottom_left.x as i32), -(bottom_left.y as i32)));
        self.sort();
    }

    // This function is private because every other function that mutates the coordinates
    // list is expected to sort before returning the piece.
    fn sort(&mut self) {
        self.coords.sort();
    }
}

pub fn piece_list(expected_piece_count: usize, board_size: usize) -> Result<Vec<Piece>> {
    let make_coord = |x, y| Coord { x, y, board_size };

    let pieces = vec![
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(2, 0),
            make_coord(3, 0),
            make_coord(4, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(1, 1),
            make_coord(2, 1),
            make_coord(3, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(0, 2),
            make_coord(1, 0),
            make_coord(2, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(1, 1),
            make_coord(1, 2),
            make_coord(2, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(2, 1),
            make_coord(2, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(2, 1),
            make_coord(3, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(2, 0),
            make_coord(3, 0),
            make_coord(1, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(2, 1),
            make_coord(2, 2),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(1, 2),
            make_coord(2, 2),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(1, 0),
            make_coord(2, 0),
        ]),
        Piece::new(vec![
            make_coord(1, 1),
            make_coord(0, 1),
            make_coord(1, 0),
            make_coord(2, 1),
            make_coord(1, 2),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(1, 2),
            make_coord(2, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(1, 1),
            make_coord(2, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(2, 0),
            make_coord(3, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(2, 0),
            make_coord(0, 1),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(0, 1),
            make_coord(1, 1),
            make_coord(1, 0),
        ]),
        Piece::new(vec![
            make_coord(0, 0),
            make_coord(1, 0),
            make_coord(2, 0),
            make_coord(1, 1),
        ]),
        Piece::new(vec![make_coord(0, 0), make_coord(1, 0), make_coord(2, 0)]),
        Piece::new(vec![make_coord(0, 0), make_coord(1, 0), make_coord(0, 1)]),
        Piece::new(vec![make_coord(0, 0), make_coord(1, 0)]),
        Piece::new(vec![make_coord(0, 0)]),
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
            Piece::new(vec![
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
            ])
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
            Piece::new(vec![
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
                    x: 1,
                    y: 18,
                    board_size: 20
                },
                Coord {
                    x: 1,
                    y: 17,
                    board_size: 20
                },
                Coord {
                    x: 2,
                    y: 19,
                    board_size: 20
                },
            ])
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
            Piece::new(vec![
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
            ])
        );

        piece.recenter();
        assert_eq!(piece, get_piece());
    }
}
