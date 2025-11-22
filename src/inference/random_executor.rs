use crate::{inference, inference::batcher::Executor};
use std::time::Duration;

use rand::Rng;

/// The random executor just returns random values and policies.
pub struct RandomExecutor {
    sleep_duration: Duration,
}

impl RandomExecutor {
    pub fn build(sleep_duration: Duration) -> Self {
        Self { sleep_duration }
    }
}

impl Executor for RandomExecutor {
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
        std::thread::sleep(self.sleep_duration);
        requests
            .into_iter()
            .map(|request| {
                let unnormalized_value = std::array::from_fn(|_| rand::rng().random::<f32>());
                let value_sum = unnormalized_value.iter().sum::<f32>();

                let unnormalized_policy = (0..request.valid_move_indexes.len())
                    .map(|_| rand::rng().random::<f32>())
                    .collect::<Vec<f32>>();
                let policy_sum = unnormalized_policy.iter().sum::<f32>();

                inference::Response {
                    value: unnormalized_value.map(|x| x / value_sum),
                    policy: unnormalized_policy
                        .into_iter()
                        .map(|x| x / policy_sum)
                        .collect(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Board;
    use crate::testing;

    #[test]
    fn test_execute() {
        let executor = RandomExecutor::build(Duration::from_millis(1));
        let results = executor.execute(vec![inference::Request {
            board: Board::new(&testing::create_game_config()),
            valid_move_indexes: vec![0, 1, 2],
        }]);

        assert_eq!(results.len(), 1);
        assert!((results[0].value.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((results[0].policy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
