use std::sync::Arc;

use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::agents::Agent;
use crate::config::{GameConfig, PolicySamplingConfig};
use crate::game::State;
use crate::game::move_data::move_index_to_player_pov;
use crate::inference;
use async_trait::async_trait;

pub struct PolicySamplingAgent<T: inference::Client + Send + Sync> {
    pub name: String,
    temperature: f32,
    game_config: &'static GameConfig,
    inference_client: Arc<T>,
}

impl<T: inference::Client + Send + Sync> PolicySamplingAgent<T> {
    pub fn new(
        policy_sampling_config: &'static PolicySamplingConfig,
        game_config: &'static GameConfig,
        inference_client: Arc<T>,
    ) -> Self {
        Self {
            name: policy_sampling_config.name.clone(),
            temperature: policy_sampling_config.temperature,
            game_config,
            inference_client,
        }
    }
}

#[async_trait]
impl<T: inference::Client + Send + Sync> Agent for PolicySamplingAgent<T> {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> usize {
        let player = state.player();
        let valid_moves: Vec<usize> = state.valid_moves().collect();
        let player_pov_move_indexes: Vec<usize> = valid_moves
            .iter()
            .map(|&move_index| {
                move_index_to_player_pov(move_index, player, self.game_config.move_profiles())
            })
            .collect();

        let inference_result = self
            .inference_client
            .evaluate(inference::Request {
                board: state.board().clone_with_player_pov(player as i32),
                valid_move_indexes: player_pov_move_indexes,
            })
            .await;

        let array_index = if self.temperature.abs() < f32::EPSILON {
            inference_result
                .policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx)
                .unwrap()
        } else {
            let unnormalized_weights: Vec<f32> = inference_result
                .policy
                .iter()
                .map(|&p| p.powf(1.0 / self.temperature))
                .collect();

            let normalization_factor = unnormalized_weights.iter().sum::<f32>();
            let weights: Vec<f32> = unnormalized_weights
                .iter()
                .map(|x| x / normalization_factor)
                .collect();

            let dist = WeightedIndex::new(&weights).unwrap();
            dist.sample(&mut rand::rng())
        };

        valid_moves[array_index]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::inference;
    use crate::testing;

    struct MockInferenceClient {
        pub requests: Mutex<Vec<inference::Request>>,
        pub policy: Vec<f32>,
    }

    impl inference::Client for MockInferenceClient {
        async fn evaluate(&self, request: inference::Request) -> inference::Response {
            self.requests.lock().unwrap().push(request);

            inference::Response {
                value: [0.0; 4],
                policy: self.policy.clone(),
            }
        }
    }

    #[tokio::test]
    async fn test_choose_move_with_temperature_zero() {
        let game_config = testing::create_game_config();
        let policy_sampling_config: &'static PolicySamplingConfig =
            Box::leak(Box::new(PolicySamplingConfig {
                name: "test_policy_sampling".to_string(),
                inference_config_name: "unused".to_string(),
                temperature: 0.0,
            }));

        let mut state = State::new(game_config);
        state.apply_move(state.first_valid_move().unwrap());

        let valid_move_count = state.valid_moves().count();
        let preferred_index = 1;
        let mut policy = vec![0.0; valid_move_count];
        policy[preferred_index] = 1.0;

        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
            policy,
        });

        let mut agent = PolicySamplingAgent::new(
            policy_sampling_config,
            game_config,
            Arc::clone(&mock_client),
        );

        let move_index = agent.choose_move(&state).await;
        assert!(state.is_valid_move(move_index));

        // With temperature 0, the agent should pick the move with highest probability.
        let expected_move_index = state.valid_moves().nth(preferred_index).unwrap();
        assert_eq!(move_index, expected_move_index);

        // Ensure the inference request used player POV move indexes.
        let request = mock_client.requests.lock().unwrap().pop().unwrap();
        let expected_valid_moves: Vec<usize> = state
            .valid_moves()
            .map(|move_index| {
                move_index_to_player_pov(move_index, state.player(), game_config.move_profiles())
            })
            .collect();
        assert_eq!(request.valid_move_indexes, expected_valid_moves);
    }

    #[tokio::test]
    async fn test_choose_move_with_temperature_sampling() {
        let game_config = testing::create_game_config();
        let policy_sampling_config: &'static PolicySamplingConfig =
            Box::leak(Box::new(PolicySamplingConfig {
                name: "test_policy_sampling_temp".to_string(),
                inference_config_name: "unused".to_string(),
                temperature: 1.5,
            }));

        let state = State::new(game_config);
        let valid_move_count = state.valid_moves().count();
        let mut policy = vec![1.0 / valid_move_count as f32; valid_move_count];
        policy[0] = 0.4;
        policy[1] = 0.3;
        policy[2] = 0.3;

        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
            policy,
        });

        let mut agent = PolicySamplingAgent::new(
            policy_sampling_config,
            game_config,
            Arc::clone(&mock_client),
        );

        for _ in 0..5 {
            let move_index = agent.choose_move(&state).await;
            assert!(state.is_valid_move(move_index));
        }
    }
}
