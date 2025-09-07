use crate::game::BoardSlice;

pub struct BoardDisplayLayer<'a> {
    pub color: &'a str,
    pub board_slice: &'a BoardSlice,
}

pub struct BoardDisplay<'a> {
    layers: Vec<BoardDisplayLayer<'a>>,
}

impl<'a> BoardDisplay<'a> {
    pub fn new(layers: Vec<BoardDisplayLayer<'a>>) -> Self {
        BoardDisplay { layers }
    }

    pub fn render(&self) -> String {
        let size = self.layers[0].board_slice.size();

        let mut result = String::new();

        result.push_str("\n   ");
        for x in 0..size {
            let col_letter = (b'A' + x as u8) as char;
            result.push_str(&format!("{:2}", col_letter));
        }
        result.push('\n');

        // Print top border
        result.push_str(" ┌");
        result.push_str(&"─".repeat(size * 2 + 1));
        result.push_str("┐\n");

        // Print rows with letters
        for y in 0..size {
            // Print row letter
            let row_letter = (b'A' + y as u8) as char;
            result.push_str(&format!("{}│ ", row_letter));

            for x in 0..size {
                // For the first layer where the cell is occupied,
                // produce output for that layer.

                let mut cell_representation = "· ";
                for layer in &self.layers {
                    if layer.board_slice.get((x, y)) {
                        cell_representation = match layer.color {
                            "black" => "⬛",
                            "red" => "🟥",
                            "blue" => "🟦",
                            "green" => "🟩",
                            "yellow" => "🟨",
                            _ => "⬛",
                        };
                        break;
                    }
                }
                result.push_str(cell_representation);
            }
            result.push_str("│\n");
        }

        // Print bottom border
        result.push_str(" └");
        result.push_str(&"─".repeat(size * 2 + 1));
        result.push('┘');

        result
    }
}
