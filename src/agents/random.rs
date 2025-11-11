use crate::agents::Agent;
use crate::config::RandomConfig;
use crate::game::State;
use async_trait::async_trait;
use rand::prelude::IteratorRandom;

pub struct RandomAgent {
    pub name: String,
}

impl RandomAgent {
    pub fn new(random_config: &RandomConfig) -> Self {
        Self {
            name: random_config.name.clone(),
        }
    }
}

#[async_trait]
impl Agent for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> usize {
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
        let config = testing::create_game_config();
        let random_config = crate::config::RandomConfig {
            name: "test_random".to_string(),
        };
        let mut agent = RandomAgent::new(&random_config);
        let state = State::new(&config);
        let move_index = agent.choose_move(&state).await;
        assert!(state.valid_moves().contains(&move_index));
    }
}
