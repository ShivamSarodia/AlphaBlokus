use crate::config::GameConfig;
use serde::{Deserialize, Serialize};

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

// Structure representing one MovesArray per player. Commented out
// because it's not yet used.
// pub struct MultiPlayerMovesArray<T>([MovesArray<T>; NUM_PLAYERS]);
//
// impl<T: Clone> MultiPlayerMovesArray<T> {
//     pub fn new_with(value: T, game_config: &GameConfig) -> Self {
//         return MultiPlayerMovesArray(std::array::from_fn(|_| {
//             MovesArray::new_with(value.clone(), &game_config)
//         }));
//     }
// }
