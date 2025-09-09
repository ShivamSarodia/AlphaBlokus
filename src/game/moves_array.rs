use crate::config::GameConfig;
use bit_set::BitSet;
use serde::{Deserialize, Serialize};

// List of one-value-per-move.
#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MovesBitSet {
    values: BitSet,
    num_moves: usize,
}

impl MovesBitSet {
    pub fn new(num_moves: usize) -> Self {
        MovesBitSet {
            values: BitSet::with_capacity(num_moves),
            num_moves,
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.values.contains(index)
    }

    pub fn insert(&mut self, index: usize) {
        self.values.insert(index);
    }

    pub fn num_moves(&self) -> usize {
        self.num_moves
    }

    pub fn add_mut(&mut self, other: &Self) {
        self.values.union_with(&other.values);
    }

    pub fn subtract_iter<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = usize> + 'a {
        self.values.difference(&other.values)
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
