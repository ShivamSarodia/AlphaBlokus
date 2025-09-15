use std::sync::Arc;

use log::trace;

use crate::agents::Agent;
use crate::agents::mcts::node::Node;
use crate::config::{GameConfig, MCTSConfig};
use crate::game::{GameStatus, State};
use crate::inference;
use itertools::Itertools;

pub struct MCTSAgent<T: inference::Client> {
    mcts_config: Arc<MCTSConfig>,
    game_config: Arc<GameConfig>,
    inference_client: Arc<T>,
}

impl<T: inference::Client> MCTSAgent<T> {
    pub fn new(
        mcts_config: Arc<MCTSConfig>,
        game_config: Arc<GameConfig>,
        inference_client: Arc<T>,
    ) -> Self {
        Self {
            mcts_config,
            game_config,
            inference_client,
        }
    }

    async fn rollout_once(&self, state: &State<'_>, search_root: &mut Node) {
        trace!("Rolling out once from state: {}", state);

        let mut moves_played = Vec::new();

        let value = {
            let mut current_state = state.clone();
            // This reborrows search_root as mutable to avoid moving it, since we'll
            // need it again below when backpropagating the value.
            let mut current_node = &mut *search_root;

            loop {
                trace!(
                    "Rollout traversal iteration. Moves played: {:?}, current state: {}",
                    moves_played, current_state,
                );

                // Select the next child node to explore.
                let move_index = current_node.select_move_by_ucb();

                // Play and record the selected move.
                let game_status = current_state.apply_move(move_index);
                moves_played.push(move_index);

                // If the game is now over, we just assign values based on the final state.
                if game_status == GameStatus::GameOver {
                    trace!(
                        "Iteration terminated because game is over. Current state: {}",
                        current_state
                    );
                    break current_state.result();
                }

                // Try to find an existing child node for the selected move.
                if current_node.has_child(move_index) {
                    trace!(
                        "Proceeding to next iteration: found existing child node for move index: {}",
                        move_index
                    );
                    current_node = current_node.get_child_mut(move_index).unwrap();
                } else {
                    trace!(
                        "Expanding new node: no existing child node for move index: {}",
                        move_index
                    );
                    let new_node = Node::build_and_expand(
                        &current_state,
                        self.inference_client.as_ref(),
                        Arc::clone(&self.mcts_config),
                        Arc::clone(&self.game_config),
                        false,
                    )
                    .await;
                    let value = new_node.get_value_as_universal_pov();
                    current_node.add_child(move_index, new_node);
                    trace!("Added new node to parent node. Terminating iteration.");
                    break value;
                }
            }
        };

        trace!("Backpropagating through moves played: {:?}", moves_played);

        // Now, backpropagate the value we just learned up the tree.
        let mut node: Option<&mut Node> = Some(&mut *search_root);
        for &move_index in moves_played.iter() {
            node.as_deref_mut()
                .unwrap()
                .increment_child_value_sum(move_index, value);
            node.as_deref_mut()
                .unwrap()
                .increment_child_visit_count(move_index);
            node = node.unwrap().get_child_mut(move_index);
        }
    }
}

impl<T: inference::Client> Agent for MCTSAgent<T> {
    async fn choose_move(&self, state: &State<'_>) -> usize {
        // Create a new node to represent the root of the search tree. Start by expanding the
        // node immediately.
        let mut search_root = Node::build_and_expand(
            state,
            self.inference_client.as_ref(),
            Arc::clone(&self.mcts_config),
            Arc::clone(&self.game_config),
            true,
        )
        .await;

        // Run the rollouts, which formulates the search tree.
        for _ in 0..self.mcts_config.num_rollouts {
            self.rollout_once(state, &mut search_root).await;
        }

        search_root.select_move_to_play(state)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[tokio::test]
    async fn test_rotations_and_rolls() {
        let mcts_config = Arc::new(MCTSConfig {
            // For this test, perform just one rollout so that there's only one substantial
            // call to the inference client evaluation.
            num_rollouts: 1,
            total_dirichlet_noise_alpha: 1.0,
            root_dirichlet_noise_fraction: 0.0,
            ucb_exploration_factor: 1.0,
            temperature_turn_cutoff: 0,
            move_selection_temperature: 0.0,
        });

        // Generate a slightly larger game config than the standard 5x5 for testing
        // so that there's no concern about one move blocking the others.
        let mut raw_game_config = GameConfig {
            board_size: 10,
            num_moves: 6233,
            num_pieces: 21,
            num_piece_orientations: 91,
            move_data: None,
            move_data_file: "static/move_data_size_10.bin".to_string(),
        };
        raw_game_config.load_move_profiles().unwrap();
        let game_config = Arc::new(raw_game_config);

        struct MockInferenceClient {
            pub requests: RefCell<Vec<inference::Request>>,
        }

        impl inference::Client for MockInferenceClient {
            async fn evaluate(&self, request: inference::Request) -> inference::Response {
                // Push the requests onto the vector.
                self.requests.borrow_mut().push(request.clone());

                // Return a response where the current player's value is 1.0 and the other
                // players values are 0.0. It also returns a policy where move 1 is preferred.
                let value = [1.0, 0.0, 0.0, 0.0];
                let mut policy = vec![0.0; request.valid_move_indexes.len()];
                let index_to_prefer = request
                    .valid_move_indexes
                    .iter()
                    .position(|&move_index| move_index == 1008)
                    .unwrap();
                policy[index_to_prefer] = 1.0;

                inference::Response { value, policy }
            }
        }

        let mock_client = Arc::new(MockInferenceClient {
            requests: RefCell::new(Vec::new()),
        });
        let agent = MCTSAgent::new(
            mcts_config,
            Arc::clone(&game_config),
            Arc::clone(&mock_client),
        );

        let mut state = State::new(&game_config);
        let move_index_0 = agent.choose_move(&state).await;
        let move_profile_0 = game_config.move_profiles().get(move_index_0);

        // For the player 0 state, the move that's played should match the one preferred by the
        // policy.
        assert_eq!(
            move_index_0,
            state.valid_moves().collect::<Vec<usize>>()[10]
        );

        // On the first request, the valid move indexes and board should just match the state's
        // valid moves and board.
        let request_0 = mock_client.requests.borrow()[0].clone();
        assert_eq!(
            request_0.valid_move_indexes,
            state.valid_moves().collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 0 board: {}", request_0.board);
        assert_eq!(request_0.board, *state.board());

        // Now, apply the move and run a second rollout on the new state.
        mock_client.requests.borrow_mut().clear();
        state.apply_move(move_index_0);
        let move_index_1 = agent.choose_move(&state).await;
        let move_profile_1 = game_config.move_profiles().get(move_index_1);

        // On the second request, the valid move indexes should match the first request
        // because from the player's own perspective, the legal moves are the same in both
        // cases.
        let request_1 = mock_client.requests.borrow()[0].clone();
        assert_eq!(
            request_0
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
            request_1
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 1 board: {}", request_1.board);
        assert_eq!(request_1.board.slice(0).count(), 0);
        assert_eq!(request_1.board.slice(1).count(), 0);
        assert_eq!(request_1.board.slice(2).count(), 0);
        assert_eq!(request_1.board.slice(3).count(), 5);
        assert_eq!(request_1.board.slice(3).get((0, 9)), true);

        // Confirm that the piece selected is the same as in the first rollout.
        assert_eq!(move_profile_0.piece_index, move_profile_1.piece_index);

        mock_client.requests.borrow_mut().clear();
        state.apply_move(move_index_1);
        let move_index_2 = agent.choose_move(&state).await;
        let move_profile_2 = game_config.move_profiles().get(move_index_2);

        let request_2 = mock_client.requests.borrow()[0].clone();
        assert_eq!(
            request_0
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
            request_2
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 2 board: {}", request_2.board);
        assert_eq!(request_2.board.slice(0).count(), 0);
        assert_eq!(request_2.board.slice(1).count(), 0);
        assert_eq!(request_2.board.slice(2).count(), 5);
        assert_eq!(request_2.board.slice(2).get((9, 9)), true);
        assert_eq!(request_2.board.slice(3).count(), 5);
        assert_eq!(request_2.board.slice(3).get((0, 9)), true);

        // Something is VERY wrong!!
        // Look at state at Turn 1 vs Turn 2. It's clear that the second player
        // corner is in top right. However, the request 1 board is not correct
        // for that -- the request 1 board is showing the POV from the bottom
        // left corner instead of the top right.

        assert_eq!(move_profile_0.piece_index, move_profile_2.piece_index);

        // Confirm the board looks right
        // Confirm the valid move indexes look right
        // first_request.vali

        // state.apply_move(move_index);
        // assert_eq!(state.player(), 1);

        // Now, we're on to player 2.

        // TODO: Still need to confirm the *values* rotate ok.
    }
}
