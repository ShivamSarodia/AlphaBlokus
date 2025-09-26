use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::Axis;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use anyhow::{Context, Result};

use crate::{
    config::{GameConfig, NUM_PLAYERS},
    inference,
    inference::batcher::Executor,
    inference::softmax::softmax_inplace,
};

/// The ORT executor runs inference using the ORT library. This executor isn't
/// optimized for performance, but rather delivers a straightforward inference
/// implementation for local development.
pub struct OrtExecutor {
    session: Arc<Mutex<Session>>,
    game_config: &'static GameConfig,
}

impl OrtExecutor {
    pub fn build(model_path: &Path, game_config: &'static GameConfig) -> Result<Self> {
        println!(
            "Building ORT executor with model path: {}",
            model_path.display()
        );
        let session = Session::builder()
            .context("Failed to create ORT session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set ORT optimization level")?
            .commit_from_file(model_path)
            .with_context(|| {
                format!("Failed to commit ORT session from {}", model_path.display())
            })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            game_config,
        })
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
            .zip(
                requests
                    .into_iter()
                    .map(|request| request.valid_move_indexes),
            )
            .map(|((value, policy), valid_move_indexes)| {
                let value_slice = value.as_slice().unwrap();
                let mut value =
                    <[f32; 4]>::try_from(value_slice).expect("value must have length 4");
                softmax_inplace(&mut value);

                let mut policy = valid_move_indexes
                    .iter()
                    // If the network returns each policy as a 91 x N x N array rather than
                    // a flattened array, we can just use the move profiles to look up the
                    // (piece_orientation_index, center_x, center_y) for each valid move index
                    // and use that tuple to identify which value from the policy array to
                    // pull in here.
                    .map(|&index| *policy.get(index).unwrap())
                    .collect::<Vec<f32>>();
                softmax_inplace(&mut policy);

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
    use std::path::Path;

    #[test]
    fn test_execute() {
        let executor = OrtExecutor::build(
            Path::new("static/networks/trivial_net_tiny.onnx"),
            testing::create_game_config(),
        )
        .unwrap();

        let mut board_1 = Board::new(&testing::create_game_config());
        board_1.slice_mut(0).set((0, 0), true);
        let request_1 = inference::Request {
            board: board_1,
            valid_move_indexes: vec![0, 1, 2],
        };

        let mut board_1_copy = Board::new(&testing::create_game_config());
        board_1_copy.slice_mut(0).set((0, 0), true);
        let request_1_copy = inference::Request {
            board: board_1_copy,
            valid_move_indexes: vec![2, 1, 0],
        };

        let mut board_2 = Board::new(&testing::create_game_config());
        board_2.slice_mut(1).set((2, 2), true);
        let request_2 = inference::Request {
            board: board_2,
            valid_move_indexes: vec![0, 1, 2],
        };

        let results = executor.execute(vec![request_1, request_1_copy, request_2]);

        assert_eq!(results.len(), 3);

        // Confirm the first and second values match but the policies are reversals of
        // each other, because the valid moves input was reversed.
        assert_eq!(results[0].value, results[1].value);
        assert_eq!(
            results[0].policy,
            results[1]
                .policy
                .iter()
                .copied()
                .rev()
                .collect::<Vec<f32>>()
        );

        // Confirm the first and third results do not match, because the inputs were
        // different.
        assert_ne!(results[0].value, results[2].value);
        assert_ne!(results[0].policy[0], results[2].policy[0]);
        assert_ne!(results[0].policy[1], results[2].policy[1]);
        assert_ne!(results[0].policy[2], results[2].policy[2]);
    }
}
