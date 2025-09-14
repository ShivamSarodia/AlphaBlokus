use std::sync::{Arc, Mutex};

use ndarray::Axis;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::{
    config::{GameConfig, NUM_PLAYERS},
    game::MovesArray,
    inference,
    inference::batcher::Executor,
};

/// The ORT executor runs inference using the ORT library. This executor isn't
/// optimized for performance, but rather delivers a straightforward inference
/// implementation for local development.
pub struct OrtExecutor {
    session: Arc<Mutex<Session>>,
    game_config: Arc<GameConfig>,
}

impl OrtExecutor {
    pub fn build(model_path: &str, game_config: Arc<GameConfig>) -> Self {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        Self {
            session: Arc::new(Mutex::new(session)),
            game_config,
        }
    }
}

impl Executor for OrtExecutor {
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response> {
        let mut input_array = ndarray::Array4::<f32>::zeros((
            requests.len(),
            NUM_PLAYERS,
            self.game_config.board_size,
            self.game_config.board_size,
        ));

        for (batch_index, request) in requests.iter().enumerate() {
            for player in 0..NUM_PLAYERS {
                for x in 0..self.game_config.board_size {
                    for y in 0..self.game_config.board_size {
                        if request.board.slice(player).get((x, y)) {
                            input_array[[batch_index, player, x, y]] = 1.0;
                        }
                    }
                }
            }
        }
        let ort_inputs = ort::inputs!["board" => Tensor::from_array(input_array).unwrap()];
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort_inputs).unwrap();

        let values = outputs["value"].try_extract_array().unwrap();
        let policies = outputs["policy"].try_extract_array().unwrap();

        values
            .axis_iter(Axis(0))
            .zip(policies.axis_iter(Axis(0)))
            .map(|(value, policy)| {
                let value_slice = value.as_slice().unwrap();
                let value = <[f32; 4]>::try_from(value_slice).expect("value must have length 4");

                let policy_slice = policy.as_slice().unwrap();
                let policy = MovesArray::<f32>::try_from(policy_slice, &self.game_config)
                    .expect("policy must have length 4");

                inference::Response { value, policy }
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
        let executor = OrtExecutor::build(
            "static/networks/trivial_net.onnx",
            testing::create_game_config(),
        );

        let mut board_1 = Board::new(&testing::create_game_config());
        board_1.slice_mut(0).set((0, 0), true);
        let request_1 = inference::Request { board: board_1 };

        let mut board_1_copy = Board::new(&testing::create_game_config());
        board_1_copy.slice_mut(0).set((0, 0), true);
        let request_1_copy = inference::Request {
            board: board_1_copy,
        };

        let mut board_2 = Board::new(&testing::create_game_config());
        board_2.slice_mut(1).set((2, 2), true);
        let request_2 = inference::Request { board: board_2 };

        let results = executor.execute(vec![request_1, request_1_copy, request_2]);

        assert_eq!(results.len(), 3);
        // The first and second results should match, because the inputs were
        // identical.
        assert_eq!(results[0].value, results[1].value);
        assert_eq!(results[0].policy, results[1].policy);

        // The first and third results should not match, because the inputs were
        // different.
        assert_ne!(results[0].value, results[2].value);
        assert_ne!(results[0].policy, results[2].policy);
    }
}
