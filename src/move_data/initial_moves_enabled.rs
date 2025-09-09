use crate::{
    config::NUM_PLAYERS,
    game::{MovesArray, MovesBitSet},
    move_data::MoveProfile,
};

pub fn compute_initial_moves_enabled(
    move_profiles: &MovesArray<MoveProfile>,
    num_moves: usize,
    board_size: usize,
) -> [MovesBitSet; NUM_PLAYERS] {
    let mut moves_enabled = std::array::from_fn(|_| MovesBitSet::new(num_moves));

    for i in 0..num_moves {
        let move_profile = move_profiles.get(i);
        if move_profile.occupied_cells.get((0, 0)) {
            moves_enabled[0].insert(i);
        }
        if move_profile.occupied_cells.get((0, board_size - 1)) {
            moves_enabled[1].insert(i);
        }
        if move_profile
            .occupied_cells
            .get((board_size - 1, board_size - 1))
        {
            moves_enabled[2].insert(i);
        }
        if move_profile.occupied_cells.get((board_size - 1, 0)) {
            moves_enabled[3].insert(i);
        }
    }

    moves_enabled
}
