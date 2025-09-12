use crate::agents::Agent;
use crate::game::State;
use rand::prelude::IteratorRandom;

#[derive(Default)]
pub struct RandomAgent {}

impl Agent for RandomAgent {
    async fn choose_move(&self, state: &State<'_>) -> usize {
        state.valid_moves().choose(&mut rand::rng()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::testing;

    use super::*;

    #[tokio::test]
    async fn test_choose_move() {
        let agent = RandomAgent {};
        let state = State::new(testing::create_game_config());
        let move_index = agent.choose_move(&state).await;
        assert!(state.valid_moves().contains(&move_index));
    }
}
