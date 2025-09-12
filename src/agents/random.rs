use crate::agents::Agent;
use crate::game::State;
use rand::prelude::IteratorRandom;

pub struct RandomAgent {}

impl Agent for RandomAgent {
    fn choose_move(&self, state: &State) -> usize {
        state.valid_moves().choose(&mut rand::rng()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::testing;

    use super::*;

    #[test]
    fn test_choose_move() {
        let agent = RandomAgent {};
        let state = State::new(testing::create_game_config());
        let move_index = agent.choose_move(&state);
        assert!(state.valid_moves().contains(&move_index));
    }
}
