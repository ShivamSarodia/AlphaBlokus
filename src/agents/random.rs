use crate::agents::Agent;
use crate::config::{GameConfig, RandomConfig};
use crate::game::State;
use async_trait::async_trait;
use rand::prelude::IteratorRandom;

pub struct RandomAgent {
    pub name: String,
    from_largest: bool,
    game_config: &'static GameConfig,
}

impl RandomAgent {
    pub fn new(random_config: &RandomConfig, game_config: &'static GameConfig) -> Self {
        Self {
            name: random_config.name.clone(),
            from_largest: random_config.from_largest,
            game_config,
        }
    }
}

#[async_trait]
impl Agent for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> usize {
        if !self.from_largest {
            return state.valid_moves().choose(&mut rand::rng()).unwrap();
        }

        let mut largest_moves = Vec::new();
        let mut max_cells = 0usize;
        for move_index in state.valid_moves() {
            let count = self
                .game_config
                .move_profiles()
                .get(move_index)
                .occupied_cells
                .count();
            if count > max_cells {
                max_cells = count;
                largest_moves.clear();
                largest_moves.push(move_index);
            } else if count == max_cells {
                largest_moves.push(move_index);
            }
        }

        largest_moves.into_iter().choose(&mut rand::rng()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::testing;

    use super::*;

    #[tokio::test]
    async fn test_choose_move() {
        let config = testing::create_game_config();
        let random_config = crate::config::RandomConfig {
            name: "test_random".to_string(),
            from_largest: false,
        };
        let mut agent = RandomAgent::new(&random_config, config);
        let state = State::new(&config);
        let move_index = agent.choose_move(&state).await;
        assert!(state.valid_moves().contains(&move_index));
    }

    #[tokio::test]
    async fn test_choose_move_from_largest() {
        let config = testing::create_game_config();
        let random_config = crate::config::RandomConfig {
            name: "test_random_largest".to_string(),
            from_largest: true,
        };
        let mut agent = RandomAgent::new(&random_config, config);
        let state = State::new(&config);

        let chosen_move = agent.choose_move(&state).await;
        let chosen_count = config
            .move_profiles()
            .get(chosen_move)
            .occupied_cells
            .count();

        let max_count = state
            .valid_moves()
            .map(|idx| config.move_profiles().get(idx).occupied_cells.count())
            .max()
            .unwrap();
        assert_eq!(chosen_count, max_count);
    }
}
